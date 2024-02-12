//The following inline functions are MurmurHash3_x86_32 implementation, see https://github.com/jwerle/murmurhash.c

__device__ inline static unsigned ROTL32 ( unsigned x, char r ) {return (x << r) | (x >> (32 - r));}
__device__ inline unsigned fmix32 ( unsigned h ) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
__device__ inline void MurmurHash3_x86_32 ( const void * key, int len,
       unsigned seed, void * out ) {
    const unsigned char * data = (const unsigned char*)key;
    const int nblocks = len / 4;
 
    unsigned h1 = seed;
 
    const unsigned c1 = 0xcc9e2d51;
    const unsigned c2 = 0x1b873593;

    const unsigned * blocks = (const unsigned *)(data + nblocks*4);
 
    for(int i = -nblocks; i; i++) {
        unsigned k1 = blocks[i];
 
        k1 *= c1;
        k1 = ROTL32(k1,15);
        k1 *= c2;
 
        h1 ^= k1;
        h1 = ROTL32(h1,13);
        h1 = h1*5+0xe6546b64;
    }
    const unsigned char * tail = (const unsigned char*)(data + nblocks*4);
    unsigned k1 = 0;
    switch(len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = ROTL32(k1,15); k1 *= c2; h1 ^= k1;
    };
    h1 ^= len;
    h1 = fmix32(h1);
    *(unsigned*)out = h1;
}

/*
    The two functions below (create_index_4_bit, read_index_4_bit) create and read the indexing structure

    Because NCCL AllReduce does not support bitwise-OR operation, we use 4 bits to represent the index. So that we can 
    use the inherent SUM function to aggregate the index. This may cause additional index overhead. Users can modify 
    the code to use 1 bit to represent each mask to reduce the index overhead when using other aggregation APIs (See the 
    annotated codes below the two functions).

    Here is the example of the two functions:

    gradient: [x, 0, y, 0, z, 0]

   __global__ void create_index_4_bit: Get the bitmap of the gradient

    input: 
        *data: The pointer to the gradient, i.e., [x, 0, y, 0, z, 0]
        *index: The pointer to the index, i.e., [0, 0, 0, 0, 0, 0]
    
    expected result:
        *index: [1, 0, 1, 0, 1, 0] (Each number is represented by 4 bits, 1 means the corresponding gradient is not zero, 
            0 means the corresponding gradient is zero. The length of the array is 6, so the whole array is 24 bits.)

    __global__ void read_index_4_bit: Read the bitmap of the gradient and incorporate it into the gradient. The reason why
        we incorporate the index into the gradient is to reduce the memory access overhead.
    
    input:
        *data: The pointer to the gradient, e.g., [x, 0, y, 0, z, 0]
        *index: The pointer to the aggregated index, e.g., [1, 0, 1, 0, 1, 0]
        
    expected result:
        *data: [x|1, 0, y|1, 0, z|1, 0], '|' represents bitwise-OR (Each number is represented by 32 bits, 1 means the 
        corresponding gradient is not zero, 0 means the corresponding gradient is zero. The length of the array is 6, so 
        the whole array is 192 bits.)
 */

__global__ void create_index_4_bit(float* data, char* index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_2 = (tid << 1);
    index[tid] = (data[tid_2] != 0) + ((data[tid_2+1] != 0)<<4);
}

__global__ void read_index_4_bit(int* data, char* index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
        data[tid] = (data[tid] & 0xfffffffe)| ((index[tid >> 1] & (0xf << (4*(tid & 0x1)))) != 0);
}

// The following two functions are the implementation of the 1-bit indexing structure. The usage is similar to the 4-bit
__global__ void create_index_1_bit(float* data, char* index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_2 = (tid << 3);
    index[tid] = (data[tid_2] != 0) + ((data[tid_2+1] != 0)<<1) + ((data[tid_2+2] != 0)<<2) + ((data[tid_2+3] != 0)<<3) +
        ((data[tid_2+4] != 0)<<4) + ((data[tid_2+5] != 0)<<5) + ((data[tid_2+6] != 0)<<6) + ((data[tid_2+7] != 0)<<7);
}

__global__ void read_index_1_bit(int* data, char* index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
        data[tid] = (data[tid] & 0xfffffffe) | ((index[tid >> 3] & (0x1 << (tid & 0x7))) != 0);
}

/*
    The following two functions are the implementation of the compression and decompression.

    __global__ void compress_float_32: compresses the gradient into the count sketch and generate an intermediate count mapping.

    input:
        *data: The pointer to the gradient, e.g., [x'=(x|1), 0, y'=(y|1), 0, z'=(z|1), 0]. We incorporate the index into the gradient 
            to reduce the memory access overhead. Note that x' is similar to x.
        *count_sketch: The pointer to the count sketch, e.g., [0, 0, 0, 0, 0]
        *count_mapping: The pointer to the count mapping, e.g., [0, 0, 0, 0, 0]
        compressed_r: The compressed dimension, e.g., 5 (The length of the count sketch and count mapping is 5)
    
    expected result:
        *count_sketch: [x'+z', z'-y', x'-y', y'-x', -z'] (See Figure 1 of our paper for reference.)
        *count_mapping: [2, 2, 2, 2, 1] (2 means the corresponding counter of the count sketch is mapped by two parameters, 
            1 means the corresponding counter of the count sketch is mapped by one parameter.)

    __global__ void decompress_float_32: decompresses the gradient from the count sketch and the count mapping.

    input:
        *data: The pointer to the gradient, e.g., [?|1, ?|0, ?|1, ?|0, ?|1, ?|0]. '?' represents the value is undetermined.
            1 means the corresponding gradient is not zero, 0 means the corresponding gradient is zero.
        *count_sketch: The pointer to the count sketch, e.g., [x'+z', z'-y', x'-y', y'-x', -z'] (See Figure 1 of our paper for reference.)
        *count_mapping: The pointer to the count mapping, e.g., [2, 2, 2, 2, 1]
        compressed_r: The compressed dimension, e.g., 5 (The length of the count sketch and count mapping is 5)
        *flag: Indecate whether the peeling process terminates. If the peeling process terminates, the value of flag will be 
            -1. Otherwise, the value of flag >= 0 will be the number of the peeled parameters in this iteration.
    
    expected result:
        *data: [x'&0xfffffffe, 0, y'&0xfffffffe, 0, z&0xfffffffe, 0] (The value of flag will be -1)
        details:
            iteration 1:
                *data: [?|1, 0, ?|1, 0, z'&0xfffffffe, 0]
                *count_sketch: [x', -y', x'-y', y'-x', 0]
                *count_mapping: [1, 1, 2, 2, 0]
                *flag: 1 (z is peeled)
            iteration 2:
                *data: [x'&0xfffffffe, 0, y'&0xfffffffe, 0, z&0xfffffffe, 0]
                *count_sketch: [0, 0, 0, 0, 0]
                *count_mapping: [0, 0, 0, 0, 0]
                *flag: 2 (x and y are peeled)
            iteration 3:
                *data: [x'&0xfffffffe, 0, y'&0xfffffffe, 0, z&0xfffffffe, 0]
                *count_sketch: [0, 0, 0, 0, 0]
                *count_mapping: [0, 0, 0, 0, 0]
                *flag: 0 (The peeling process terminates)
*/

__global__ void compress_float_32(float* data, float* count_sketch, int* count_mapping, int compressed_r) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned hash_val_base[3] = {blockIdx.x, blockIdx.x, blockIdx.x};
    int sign [3] = {};
    MurmurHash3_x86_32(&hash_val_base[0], 4, 0, &hash_val_base[0]);
    sign[0] = (hash_val_base[0] & 0x80000000) ? -1 : 1;
    hash_val_base[0] = (hash_val_base[0] % compressed_r) * blockDim.x + threadIdx.x;
    MurmurHash3_x86_32(&hash_val_base[1], 4, 1, &hash_val_base[1]);
    sign[1] = (hash_val_base[1] & 0x80000000) ? -1 : 1;
    hash_val_base[1] = (hash_val_base[1] % compressed_r) * blockDim.x + threadIdx.x;
    MurmurHash3_x86_32(&hash_val_base[2], 4, 2, &hash_val_base[2]);
    sign[2] = (hash_val_base[2] & 0x80000000) ? -1 : 1;
    hash_val_base[2] = (hash_val_base[2] % compressed_r) * blockDim.x + threadIdx.x;
    atomicAdd(&count_sketch[hash_val_base[0]], sign[0] * data[index]);
    atomicAdd(&count_sketch[hash_val_base[1]], sign[1] * data[index]);
    atomicAdd(&count_sketch[hash_val_base[2]], sign[2] * data[index]);
    atomicAdd(&count_mapping[hash_val_base[0]>>2], (((int*)data)[index]&0x1) << ((hash_val_base[0]&0x3) << 3));
    atomicAdd(&count_mapping[hash_val_base[1]>>2], (((int*)data)[index]&0x1) << ((hash_val_base[1]&0x3) << 3));
    atomicAdd(&count_mapping[hash_val_base[2]>>2], (((int*)data)[index]&0x1) << ((hash_val_base[2]&0x3) << 3));
}

__global__ void decompress_float_32(float* data, float* count_sketch, int* count_mapping, int compressed_r, int* flag) {
    int index = (blockIdx.x * blockDim.x + threadIdx.x);
    if((((int*)data) [index] & 0x1) == 0)return;
    unsigned hash_val_base[3] = {blockIdx.x, blockIdx.x, blockIdx.x};
    int sign [3] = {};
    MurmurHash3_x86_32(&hash_val_base[0], 4, 0, &hash_val_base[0]);
    sign[0] = (hash_val_base[0] & 0x80000000) ? -1 : 1;
    hash_val_base[0] = (hash_val_base[0] % compressed_r) * blockDim.x + threadIdx.x;
    MurmurHash3_x86_32(&hash_val_base[1], 4, 1, &hash_val_base[1]);
    sign[1] = (hash_val_base[1] & 0x80000000) ? -1 : 1;
    hash_val_base[1] = (hash_val_base[1] % compressed_r) * blockDim.x + threadIdx.x;
    MurmurHash3_x86_32(&hash_val_base[2], 4, 2, &hash_val_base[2]);
    sign[2] = (hash_val_base[2] & 0x80000000) ? -1 : 1;
    hash_val_base[2] = (hash_val_base[2] % compressed_r) * blockDim.x + threadIdx.x;
    for (int i = 0; i < 3; i++){
        if(((count_mapping[hash_val_base[i] >> 2] >> ((hash_val_base[i]&0x3) << 3)) & 0xff) == 1){
            data[index] = count_sketch[hash_val_base[i]] * sign[i];
            ((int*)data)[index] = ((int*)data)[index] & 0xfffffffe;
            atomicAdd(&count_sketch[hash_val_base[0]], -sign[0] * data[index]);
            atomicAdd(&count_sketch[hash_val_base[1]], -sign[1] * data[index]);
            atomicAdd(&count_sketch[hash_val_base[2]], -sign[2] * data[index]);
            atomicAdd(&count_mapping[hash_val_base[0]>>2], -(1 << ((hash_val_base[0]&0x3) << 3)));
            atomicAdd(&count_mapping[hash_val_base[1]>>2], -(1 << ((hash_val_base[1]&0x3) << 3)));
            atomicAdd(&count_mapping[hash_val_base[2]>>2], -(1 << ((hash_val_base[2]&0x3) << 3)));
            if (flag[0] != -1)atomicAdd(&flag[0], 1);
            return;
        }
    }
}

/*
    __global__ void estimate_float_32: estimate the gradient from the count sketch after the peeling process terminates.
    For each unpeeled parameter, we estimate the gradient by the median of the three corresponding counters in the count sketch.
*/

__global__ void estimate_float_32(float* data, float* count_sketch, int compressed_r){
    int index = (blockIdx.x * blockDim.x + threadIdx.x);
    if((((int*)data) [index] & 0x1) == 0)return;
    unsigned hash_val_base[3] = {blockIdx.x, blockIdx.x, blockIdx.x};
    int sign [3] = {};
    MurmurHash3_x86_32(&hash_val_base[0], 4, 0, &hash_val_base[0]);
    sign[0] = (hash_val_base[0] & 0x80000000) ? -1 : 1;
    hash_val_base[0] = (hash_val_base[0] % compressed_r) * blockDim.x + threadIdx.x;
    MurmurHash3_x86_32(&hash_val_base[1], 4, 1, &hash_val_base[1]);
    sign[1] = (hash_val_base[1] & 0x80000000) ? -1 : 1;
    hash_val_base[1] = (hash_val_base[1] % compressed_r) * blockDim.x + threadIdx.x;
    MurmurHash3_x86_32(&hash_val_base[2], 4, 2, &hash_val_base[2]);
    sign[2] = (hash_val_base[2] & 0x80000000) ? -1 : 1;
    hash_val_base[2] = (hash_val_base[2] % compressed_r) * blockDim.x + threadIdx.x;
    float ans_0 = count_sketch[hash_val_base[0]] * sign[0];
    float ans_1 = count_sketch[hash_val_base[1]] * sign[1];
    float ans_2 = count_sketch[hash_val_base[2]] * sign[2];
    ((int*) &ans_0)[0] = ((int*)(&ans_0))[0] & 0xfffffffe;
    ((int*) &ans_1)[0] = ((int*)(&ans_1))[0] & 0xfffffffe;
    ((int*) &ans_2)[0] = ((int*)(&ans_2))[0] & 0xfffffffe;
    if ((ans_2 - ans_0)*(ans_0 - ans_1) > 0) {
        data[index] = count_sketch[hash_val_base[0]] * sign[0];
        ((int*)data)[index] = ((int*)data)[index] & 0xfffffffe;
    }
    else if ((ans_0 - ans_1)*(ans_1 - ans_2) > 0) {
        data[index] = count_sketch[hash_val_base[1]] * sign[1];
        ((int*)data)[index] = ((int*)data)[index] & 0xfffffffe;
    }
    else{
        data[index] = count_sketch[hash_val_base[2]] * sign[2];
        ((int*)data)[index] = ((int*)data)[index] & 0xfffffffe;
    } 
}

/*
    The following functions are the launch functions of the above kernels. Users can call these functions to launch the 
    corresponding kernels.

    The parameters of the launch functions are the same as the parameters of the corresponding kernels. Users can refer to 
    the comments of the corresponding kernels for the detailed explanation of the parameters.
*/

void launch_create_index_4_bit(float* data, char* index, int grid_size, int block_size) {
    create_index_4_bit<<<grid_size / 2, block_size>>>(data, index);
}

void launch_read_index_4_bit(int* data, char* index, int grid_size, int block_size) {
    read_index_4_bit<<<grid_size, block_size>>>(data, index);
}

void launch_create_index_1_bit(float* data, char* index, int grid_size, int block_size) {
    create_index_1_bit<<<grid_size / 8, block_size>>>(data, index);
}

void launch_read_index_1_bit(int* data, char* index, int grid_size, int block_size) {
    read_index_1_bit<<<grid_size, block_size>>>(data, index);
}

void launch_compress_float_32(float* data, float* count_sketch, int* count_mapping, int compressed_r, int grid_size, int block_size) {
    compress_float_32<<<grid_size, block_size>>>(data, count_sketch, count_mapping, compressed_r);
}

void launch_decompress_float_32(float* data, float* count_sketch, int* count_mapping, int compressed_r, int grid_size, int block_size, int* flag) {
    decompress_float_32<<<grid_size, block_size>>>(data, count_sketch, count_mapping, compressed_r, flag);
}

void launch_estimate_float_32(float* data, float* count_sketch, int compressed_r, int grid_size, int block_size) {
    estimate_float_32<<<grid_size, block_size>>>(data, count_sketch, compressed_r);
}