# Accelerating Distributed Deep Learning using Lossless Homomorphic Compression

As deep neural networks (DNNs) grow in complexity and size, the resultant increase in communication overhead during distributed training has become a significant bottleneck, challenging the scalability of distributed training systems. Existing solutions, while aiming to mitigate this bottleneck through worker-level compression and in-network aggregation, fall short due to their inability to efficiently reconcile the trade-offs between compression effectiveness and computational overhead, hindering overall performance and scalability. In this paper, we introduce a novel compression algorithm that effectively merges worker-level compression with in-network aggregation. Our solution is both homomorphic, allowing for efficient in-network aggregation without CPU/GPU processing, and lossless, ensuring no compromise on training accuracy. Theoretically optimal in compression and computational efficiency, our approach is empirically validated across diverse DNN models such as NCF, LSTM, VGG19, and BERT-base, demonstrating up to a 6.33x improvement in aggregation throughput and a 3.74x increase in per-iteration training speed.

This repository introduces our lossless homomorphic compression algorithm and includes a demo program (benchmark.py) that utilizes NCCL AllReduce.

## How to start

To test our benchmark, follow these steps:

1.  Ensure you have a GPU cluster setup where each GPU machine is equipped with Python, PyTorch (ensure PyTorch's CUDA version is compatible with your GPU driver), and NCCL. For any issues installing NCCL, a practical workaround is to use Docker with the following commands:
    ```
    sudo docker pull nvcr.io/nvidia/pytorch:22.09-py3

    sudo docker run -d --privileged -v $HOME:/workspace --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm --pid=host --net=host --ulimit memlock=100000000000:100000000000 --name=lossless_homomorphic_compression --shm-size=2gb --interactive --cap-add=IPC_LOCK --tty nvcr.io/nvidia/pytorch:22.09-py3

    sudo docker exec -it lossless_homomorphic_compression bash
    ```
    If you require RDMA support, ensure you have the necessary NICs and drivers, and use these commands to enable RDMA in Docker:
    ```
    sudo docker pull nvcr.io/nvidia/pytorch:22.09-py3
    
    sudo docker run -d --privileged -v $HOME:/workspace --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --rm --pid=host --net=host --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/issm0 --device=/dev/infiniband/umad0 --device=/dev/infiniband/rdma_cm --ulimit memlock=100000000000:100000000000 --name=lossless_homomorphic_compression --shm-size=2gb --interactive --cap-add=IPC_LOCK --tty nvcr.io/nvidia/pytorch:22.09-py3

    sudo docker exec -it lossless_homomorphic_compression bash
    ``` 

2. To set up our compression algorithm library, navigate to your workspace and run:
    ```
    git clone https://github.com/lihy0529/lossless_homomorphic_compression.git
    cd lossless_homomorphic_compression/lossless_homomorphic_compression_api
    pip install .
    ```
    Afterwards, to evaluate our algorithm's performance, execute:
    ```
    cd ..
    python benchmark.py
    ```
    to see the performance of our algorithm.