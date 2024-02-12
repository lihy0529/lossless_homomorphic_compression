import torch
import api
import threading
import time
import torch.distributed as dist   
import math

# The number of GPU machines.
world_size = 1
# The rank of the current GPU machine. You should run this program on the four machines simultaneously with different machine_num.
machine_num = 0
# The number of GPUs in each GPU machine.
num_of_gpu = 2

data_size = 2**27
sparsity = 0.95
compression_ratio = 10

block_size = 1024
rank0_ip = "192.168.11.10"
port = 12345

print("\nThis is a demo program to evaluate the aggregation throuput of our lossless homomorphic algorithm. ")
print("The program will run two rounds of aggregation, the first round is on the baseline NCCL algorithm, and the second round is on our lossless homomorphic algorithm. ")
print("\nThe meaning of parameters are shown below: ")
print("world_size: The number of GPU machines. For example, world_size = 4 means the program uses four GPU machines. \033[93mYou should run all machines simultaneously with different machine_num.\033[0m ")
print("machine_num: The rank of the current GPU machine. For example, if world_size = 4, machine_num is in {0, 1, 2, 3}. ")
print("num_of_gpu: The number of GPUs in each GPU machine. For example, if num_of_gpu = 2, the program uses cuda:0 and cuda:1 on all GPU machines. ")
print("data_size: The number of floating-point parameters of the generated data to be aggregated (each parameter takes 32 bits). ")
print("sparsity: The sparsity of the data. ")
print("compression_ratio: The compression ratio (the compressed data size divided by the original data size) of our lossless homomorphic algorithm. ")
print("block_size: The block size of the data. We set the block size to 1024, which is the max number of thread in one GPU block. ")
print("rank0_ip: The IP address of the rank 0 machine. ")
print("port: The port number of the rank 0 machine. Usually you don't need to change this parameter. If you encounter the port conflict, you can change this parameter. \n"
      + "\tHowever, this usually happens when you fail to kill the previous running program.")
print("\033[93mWhen your prcess is stuck, the reason is usually that (1) You set a wrong rank0_ip or machine_num so that NCCL cannot initialize. Note that all GPU machines share the same\n"
      +"rank0_ip and have different machine_num. (2) You haven't run all programs on all the GPU machines. \033[0m")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(0)

barrier = threading.Barrier(num_of_gpu)

def aggregate(compression_ratio, gpu_num):
    device = torch.device('cuda:' + str(gpu_num))
    unit_size = shm[0].numel() // num_of_gpu
    [shm[i][i * unit_size:(i+1) * unit_size].add_(shm[gpu_num][i * unit_size:(i+1) * unit_size].to('cuda:' + str(i))) for i in range(num_of_gpu) if i != gpu_num]
    barrier.wait()
    local_data = shm[gpu_num][gpu_num * unit_size: (gpu_num+1) * unit_size]
    if compression_ratio == 0:
        # baseline
        for i in range(num_of_gpu):
            if world_size == 1: break
            if i == gpu_num: dist.all_reduce(local_data, op=dist.ReduceOp.SUM, async_op=False)
            barrier.wait()
    else:
        # our method
        grid_size = local_data.numel()//block_size
        compressed_r = math.ceil(grid_size / compression_ratio)
        index = torch.zeros(local_data.numel()//2, dtype=torch.uint8, device=device)
        count_sketch = torch.zeros(compressed_r * block_size, dtype=torch.float32, device=device)
        count_mapping = torch.zeros(compressed_r * block_size, dtype=torch.uint8, device=device)
        api.torch_launch_create_index_4_bit(local_data, index, grid_size, block_size)

        for i in range(num_of_gpu):
            if world_size == 1: break
            if i == gpu_num: dist.all_reduce(index, op=dist.ReduceOp.SUM, async_op=False)
            barrier.wait()
        api.torch_launch_read_index_4_bit(local_data, index, grid_size, block_size)
        api.torch_launch_compress_float_32(local_data, count_sketch, count_mapping, compressed_r, grid_size, block_size)

        for i in range(num_of_gpu):
            if world_size == 1: break
            if i == gpu_num: dist.all_reduce(count_sketch, op=dist.ReduceOp.SUM, async_op=False)
            barrier.wait()
        flag = torch.ones(1, dtype=torch.int32, device=device)
        while flag[0] != 0:
            flag[0] = 0
            api.torch_launch_decompress_float_32(local_data, count_sketch, count_mapping, compressed_r, grid_size, block_size, flag)
        
        api.torch_launch_estimate_float_32(local_data, count_sketch, compressed_r, grid_size, block_size)
    
    [shm[i][gpu_num * unit_size:(gpu_num+1) * unit_size].copy_(local_data.to('cuda:' + str(i))) for i in range(num_of_gpu) if i != gpu_num]
    barrier.wait()


def thread_function(shm, data_size, sparsity, compression_ratio, gpu_num):
    device = torch.device('cuda:' + str(gpu_num) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    data = torch.rand(data_size, dtype=torch.float32, device=device)
    data[data < sparsity] = 0
    if gpu_num == 0: print("\n\nOriginal_data: ", data[0:20])
    for i in range(2):
        shm[gpu_num] = data.clone()
        if i == 0 and gpu_num == 0: print("\n\n\033[92mNCCL Baseline:\033[0m ")
        if i == 1 and gpu_num == 0: print("\n\n\033[92mOur Method: \033[0m")
        
        barrier.wait()
        torch.cuda.synchronize(gpu_num)
        begin_aggregation = time.time_ns()
        aggregate(compression_ratio = compression_ratio * i, gpu_num = gpu_num)
        torch.cuda.synchronize(gpu_num)
        end_aggregation = time.time_ns()
        aggregate_time = end_aggregation-begin_aggregation
        if gpu_num == 0: 
            print("Aggregated data: ", shm[gpu_num][0:20])
            print("Aggregation throughput (Gbps): \033[92m", data_size * 32 / aggregate_time, "\033[0m")
    if gpu_num == 0: print("\033[93mNote that the aggregated data of NCCL baseline and our method should be similar. Otherwise it may because the compressed data size is too high. \033[0m")




if __name__ == "__main__":
    # Initialize the process group of NCCL. 
    if world_size != 1: dist.init_process_group(backend='nccl', rank = machine_num, world_size= world_size, init_method="tcp://{}:{}".format(rank0_ip, port))
    
    threads = []
    shm = [0 for i in range(num_of_gpu)]
    for gpu_num in range(num_of_gpu):
        threads.append(threading.Thread(target=thread_function, args=(shm, data_size, sparsity, compression_ratio, gpu_num,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()