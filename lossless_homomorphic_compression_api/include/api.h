void launch_create_index_4_bit(float* data, char* index, int grid_size, int block_size);
void launch_read_index_4_bit(int* data, char* index, int grid_size, int block_size);
void launch_create_index_1_bit(float* data, char* index, int grid_size, int block_size);
void launch_read_index_1_bit(int* data, char* index, int grid_size, int block_size);
void launch_compress_float_32(float* data, float* count_sketch, int* count_mapping, int compressed_r, int grid_size, int block_size);
void launch_decompress_float_32(float* data, float* count_sketch, int* count_mapping, int compressed_r, int grid_size, int block_size, int* flag);
void launch_estimate_float_32(float* data, float* count_sketch, int compressed_r, int grid_size, int block_size);