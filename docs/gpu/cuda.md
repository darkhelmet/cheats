# CUDA Programming

A comprehensive reference for CUDA C/C++ GPU programming, covering kernels, memory management, synchronization, and optimization techniques.

## Quick Start

### Installation & Setup
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Compile CUDA program
nvcc program.cu -o program

# Compile with debugging info
nvcc -g -G program.cu -o program

# Compile for specific architecture
nvcc -arch=sm_80 program.cu -o program
```

### Basic Program Structure
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel function (runs on GPU)
__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    // Host code
    int n = 1024;
    size_t size = n * sizeof(float);
    
    // Allocate host memory
    float* h_data = (float*)malloc(size);
    
    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, size);
    
    // Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    myKernel<<<gridSize, blockSize>>>(d_data, n);
    
    // Copy result back
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
```

## Core Concepts

### Execution Model
```cuda
// Grid -> Blocks -> Threads hierarchy
__global__ void kernel() {
    // Thread indices
    int tid = threadIdx.x;                    // Thread within block
    int bid = blockIdx.x;                     // Block within grid
    int gid = bid * blockDim.x + tid;         // Global thread index
    
    // Multi-dimensional indexing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * width + col;
}

// Launch configuration
dim3 gridSize(16, 16);      // 16x16 blocks
dim3 blockSize(32, 32);     // 32x32 threads per block
kernel<<<gridSize, blockSize>>>();
```

### Function Qualifiers
```cuda
__global__ void kernelFunction() {    // Runs on GPU, called from host
    // Kernel code
}

__device__ void deviceFunction() {    // Runs on GPU, called from GPU
    // Device function code
}

__host__ void hostFunction() {        // Runs on CPU (default)
    // Host function code
}

__host__ __device__ void bothFunction() {  // Can run on both
    // Code that works on both host and device
}
```

### Variable Qualifiers
```cuda
__global__ void kernel() {
    __shared__ float sharedArray[256];    // Shared memory
    __constant__ float constValue;        // Constant memory
    __device__ float deviceGlobal;        // Global device memory
    
    int localVar;                         // Register/local memory
}
```

## Memory Management

### Basic Memory Operations
```cuda
// Device memory allocation
float* d_array;
size_t size = n * sizeof(float);
cudaError_t err = cudaMalloc(&d_array, size);

// Memory copy operations
cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);

// Memory initialization
cudaMemset(d_array, 0, size);

// Free memory
cudaFree(d_array);
```

### Asynchronous Memory Operations
```cuda
// Create streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Asynchronous operations
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream1>>>(d_data);
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream1);

// Synchronization
cudaStreamSynchronize(stream1);
cudaDeviceSynchronize();  // Wait for all operations

// Cleanup
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### Pinned Memory
```cuda
// Allocate pinned memory (faster transfers)
float* h_pinned;
cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);

// Or use cudaMallocHost
cudaMallocHost(&h_pinned, size);

// Free pinned memory
cudaFreeHost(h_pinned);
```

### Unified Memory (Managed Memory)
```cuda
// Allocate unified memory
float* unified_data;
cudaMallocManaged(&unified_data, size);

// Use directly in kernel
kernel<<<grid, block>>>(unified_data);

// Prefetch to device (optional optimization)
cudaMemPrefetchAsync(unified_data, size, 0);  // Device 0

// Free unified memory
cudaFree(unified_data);
```

## Memory Types & Hierarchy

### Global Memory
```cuda
// Allocated with cudaMalloc
// Accessible by all threads
// Highest latency, largest capacity
__global__ void kernel(float* global_mem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    global_mem[idx] = idx;  // Global memory access
}
```

### Shared Memory
```cuda
__global__ void sharedMemoryExample(float* input, float* output) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];  // Static allocation
    
    // Dynamic shared memory (specified at kernel launch)
    extern __shared__ float dynamic_tile[];
    
    int tid = threadIdx.x;
    
    // Load data to shared memory
    tile[threadIdx.y][threadIdx.x] = input[...];
    
    // Synchronize threads in block
    __syncthreads();
    
    // Use shared memory data
    float result = tile[threadIdx.y][threadIdx.x] * 2.0f;
    output[...] = result;
}

// Launch with dynamic shared memory
int sharedMemSize = TILE_SIZE * TILE_SIZE * sizeof(float);
kernel<<<grid, block, sharedMemSize>>>(input, output);
```

### Constant Memory
```cuda
// Declare constant memory (at file scope)
__constant__ float const_array[1024];

// Copy to constant memory
float host_array[1024];
cudaMemcpyToSymbol(const_array, host_array, sizeof(host_array));

__global__ void useConstant() {
    float value = const_array[threadIdx.x];  // Fast broadcast read
}
```

### Texture Memory
```cuda
// Texture object (modern approach)
texture<float, 2, cudaReadModeElementType> tex;

__global__ void textureKernel() {
    float x = blockIdx.x * blockDim.x + threadIdx.x;
    float y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Read from texture (with interpolation)
    float value = tex2D(tex, x + 0.5f, y + 0.5f);
}

// Setup texture (host code)
cudaArray* cuArray;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaMallocArray(&cuArray, &channelDesc, width, height);
cudaBindTextureToArray(tex, cuArray, channelDesc);
```

## Kernel Launch & Configuration

### Launch Configuration
```cuda
// 1D configuration
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
kernel<<<gridSize, blockSize>>>(data);

// 2D configuration
dim3 blockSize(16, 16);     // 256 threads per block
dim3 gridSize((width + 15) / 16, (height + 15) / 16);
kernel2D<<<gridSize, blockSize>>>(data);

// With shared memory
int sharedMemSize = blockSize.x * blockSize.y * sizeof(float);
kernel<<<gridSize, blockSize, sharedMemSize>>>(data);

// With streams
kernel<<<gridSize, blockSize, sharedMemSize, stream>>>(data);
```

### Dynamic Parallelism
```cuda
__global__ void parentKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && needsProcessing(data[idx])) {
        // Launch child kernel from device
        childKernel<<<1, 32>>>(data, idx);
        
        // Synchronize child kernel
        cudaDeviceSynchronize();
    }
}

__global__ void childKernel(float* data, int offset) {
    int idx = threadIdx.x;
    // Process data[offset + idx]
}
```

### Occupancy Calculation
```cuda
int blockSize;      // The launch configurator will suggest a block size
int minGridSize;    // The minimum grid size needed to achieve max occupancy
int gridSize;       // The actual grid size

// Calculate optimal block size
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

// Round up according to array size
gridSize = (n + blockSize - 1) / blockSize;

myKernel<<<gridSize, blockSize>>>(data);
```

## Thread Synchronization

### Block-Level Synchronization
```cuda
__global__ void syncExample() {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    
    // Load data
    shared_data[tid] = input[blockIdx.x * blockDim.x + tid];
    
    // Wait for all threads in block to finish loading
    __syncthreads();
    
    // Now all threads can safely access shared_data
    float result = shared_data[255 - tid];  // Reverse access
    
    // Another sync before writing results
    __syncthreads();
    
    output[blockIdx.x * blockDim.x + tid] = result;
}
```

### Warp-Level Synchronization
```cuda
#include <cooperative_groups.h>

__global__ void warpExample() {
    auto warp = cooperative_groups::tiled_partition<32>(
        cooperative_groups::this_thread_block());
    
    int lane_id = warp.thread_rank();
    int value = lane_id;
    
    // Warp-level reduction
    for (int delta = 16; delta > 0; delta /= 2) {
        value += warp.shfl_down_sync(0xffffffff, value, delta);
    }
    
    if (lane_id == 0) {
        // Thread 0 of each warp has the sum
        atomicAdd(result, value);
    }
}
```

### Atomic Operations
```cuda
// Basic atomic operations
__global__ void atomicsExample(int* counter, float* sum, int* histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Atomic increment
    int old_val = atomicAdd(counter, 1);
    
    // Atomic floating point operations (CC >= 6.0)
    atomicAdd(sum, data[idx]);
    
    // Atomic compare and swap
    int expected = 0;
    int desired = idx;
    int old = atomicCAS(&flag, expected, desired);
    
    // Histogram example
    int bin = (int)(data[idx] * NUM_BINS);
    atomicAdd(&histogram[bin], 1);
}

// Custom atomic operations using CAS
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
```

### Cooperative Groups
```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void cooperativeKernel() {
    // Thread block group
    thread_block block = this_thread_block();
    
    // Warp-sized tile
    auto tile32 = tiled_partition<32>(block);
    
    // Grid group (requires cooperative launch)
    grid_group grid = this_grid();
    
    // Synchronize at different levels
    block.sync();     // Block-level sync
    tile32.sync();    // Warp-level sync
    grid.sync();      // Grid-level sync (cooperative kernels only)
}

// Launch cooperative kernel
cudaLaunchCooperativeKernel((void*)cooperativeKernel, gridSize, blockSize, args);
```

## Error Handling

### Basic Error Checking
```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

// Check kernel launch errors
kernel<<<grid, block>>>(data);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### Advanced Error Handling
```cuda
// Error checking function
void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
}

// Async error checking
cudaError_t asyncCheckErrors() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Async kernel error: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    // Check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Sync error: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}
```

## Performance Optimization

### Memory Coalescing
```cuda
// Bad: Non-coalesced access
__global__ void badAccess(float* data, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / width;
    int col = idx % width;
    
    // Strided access pattern (bad)
    float value = data[col * width + row];  // Column-major access
}

// Good: Coalesced access
__global__ void goodAccess(float* data, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Sequential access pattern (good)
    float value = data[row * width + col];  // Row-major access
}
```

### Shared Memory Bank Conflicts
```cuda
// Bad: Bank conflicts
__global__ void badSharedAccess() {
    __shared__ float shared[32][32];
    int tid = threadIdx.x;
    
    // All threads access same bank (conflict)
    float value = shared[0][tid];
}

// Good: No bank conflicts
__global__ void goodSharedAccess() {
    __shared__ float shared[32][33];  // Padding to avoid conflicts
    int tid = threadIdx.x;
    
    // Diagonal access pattern
    float value = shared[tid][(tid + offset) % 32];
}
```

### Loop Unrolling
```cuda
// Manual unrolling for small, known loop counts
__global__ void unrolledKernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    // Unroll small loops
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
        sum += data[idx * 4 + i];
    }
    
    data[idx] = sum;
}
```

### Memory Bandwidth Optimization
```cuda
// Vectorized memory access
__global__ void vectorizedAccess(float4* input, float4* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float4 data = input[idx];  // Load 4 floats at once
        
        // Process each component
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;
        
        output[idx] = data;  // Store 4 floats at once
    }
}
```

### Instruction-Level Optimizations
```cuda
__global__ void optimizedKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = data[idx];
        
        // Use fast math functions
        float result = __sinf(x) + __cosf(x);  // Fast sin/cos
        
        // Fast division (less accurate)
        result = __fdividef(result, 3.14159f);
        
        // Fast reciprocal square root
        result *= __frsqrt_rn(result);
        
        data[idx] = result;
    }
}
```

## Common Patterns

### Reduction
```cuda
// Efficient block-level reduction
__global__ void blockReduce(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Matrix Multiplication (Tiled)
```cuda
#define TILE_SIZE 16

__global__ void matmulTiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        if (row < N && tile * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && tile * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Scan (Prefix Sum)
```cuda
__global__ void scanBlock(float* input, float* output, int n) {
    __shared__ float temp[2 * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    temp[2 * tid] = (2 * idx < n) ? input[2 * idx] : 0;
    temp[2 * tid + 1] = (2 * idx + 1 < n) ? input[2 * idx + 1] : 0;
    
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < 2 * BLOCK_SIZE; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Clear last element
    if (tid == 0) temp[2 * BLOCK_SIZE - 1] = 0;
    __syncthreads();
    
    // Down-sweep phase
    for (int stride = BLOCK_SIZE; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE) {
            float t = temp[index];
            temp[index] += temp[index - stride];
            temp[index - stride] = t;
        }
        __syncthreads();
    }
    
    // Write results
    if (2 * idx < n) output[2 * idx] = temp[2 * tid];
    if (2 * idx + 1 < n) output[2 * idx + 1] = temp[2 * tid + 1];
}
```

## Debugging & Profiling

### CUDA-GDB
```bash
# Compile with debugging info
nvcc -g -G -O0 program.cu -o program

# Run with cuda-gdb
cuda-gdb ./program

# Common commands
(cuda-gdb) break main
(cuda-gdb) break kernel_name
(cuda-gdb) run
(cuda-gdb) cuda thread (0,0,0)  # Switch to specific thread
(cuda-gdb) cuda block (0,0)     # Switch to specific block
(cuda-gdb) print variable_name
(cuda-gdb) continue
```

### Profiling with nvprof/nsys
```bash
# Legacy profiling (nvprof)
nvprof ./program

# Modern profiling (Nsight Systems)
nsys profile --stats=true ./program

# Detailed metrics
nsys profile --stats=true --force-overwrite true -o profile ./program

# GPU metrics
ncu --set full ./program
```

### Printf Debugging
```cuda
__global__ void debugKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Debug specific threads
    if (blockIdx.x == 0 && threadIdx.x < 5) {
        printf("Block %d, Thread %d: data[%d] = %f\n",
               blockIdx.x, threadIdx.x, idx, data[idx]);
    }
    
    // Conditional debugging
    if (data[idx] < 0) {
        printf("Negative value at index %d: %f\n", idx, data[idx]);
    }
}
```

## Device Management

### Device Properties
```cuda
void queryDevice() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
        printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  Memory clock rate: %d MHz\n", prop.memoryClockRate / 1000);
    }
}

void setDevice() {
    int device = 0;  // Use device 0
    cudaSetDevice(device);
    
    // Verify device was set
    int currentDevice;
    cudaGetDevice(&currentDevice);
    printf("Using device %d\n", currentDevice);
}
```

### Multi-GPU Programming
```cuda
void multiGPU() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    // Allocate data on each device
    float** d_data = new float*[deviceCount];
    cudaStream_t* streams = new cudaStream_t[deviceCount];
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaMalloc(&d_data[dev], dataSize);
        cudaStreamCreate(&streams[dev]);
        
        // Launch work on this device
        kernel<<<grid, block, 0, streams[dev]>>>(d_data[dev]);
    }
    
    // Synchronize all devices
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
        cudaFree(d_data[dev]);
        cudaStreamDestroy(streams[dev]);
    }
}
```

## Quick Reference

### Memory Copy Directions
```cuda
cudaMemcpyHostToDevice    // CPU → GPU
cudaMemcpyDeviceToHost    // GPU → CPU
cudaMemcpyDeviceToDevice  // GPU → GPU
cudaMemcpyHostToHost      // CPU → CPU (rarely used)
```

### Built-in Variables
```cuda
// Thread indexing
threadIdx.x, threadIdx.y, threadIdx.z  // Thread index within block
blockIdx.x, blockIdx.y, blockIdx.z     // Block index within grid
blockDim.x, blockDim.y, blockDim.z     // Block dimensions
gridDim.x, gridDim.y, gridDim.z        // Grid dimensions

// Warp information
warpSize                                // Usually 32
```

### Mathematical Functions
```cuda
// Standard math functions (slower, double precision available)
sin(), cos(), tan(), log(), exp(), sqrt(), pow()

// Fast math functions (faster, single precision only)
__sinf(), __cosf(), __tanf(), __logf(), __expf()
__fsqrt_rn(), __fdividef()

// Integer functions
__clz()      // Count leading zeros
__popc()     // Population count (number of set bits)
__brev()     // Bit reverse
```

### Warp Shuffle Operations
```cuda
// Warp shuffle functions (CC >= 3.0)
__shfl_sync(mask, var, srcLane)        // Direct lane access
__shfl_up_sync(mask, var, delta)       // Shift up
__shfl_down_sync(mask, var, delta)     // Shift down  
__shfl_xor_sync(mask, var, laneMask)   // XOR-based shuffle

// Example: Warp reduction
int value = threadIdx.x;
for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
}
```

### Common Gotchas & Best Practices

#### Memory Access Patterns
- **Always aim for coalesced memory access** - threads in a warp should access consecutive memory addresses
- **Use shared memory for repeated data access** - 100x faster than global memory
- **Avoid bank conflicts in shared memory** - pad arrays or use different access patterns
- **Prefer AoS over SoA for coalesced access** in most cases

#### Thread Management
- **Choose block sizes that are multiples of warp size (32)** for best efficiency
- **Aim for high occupancy but not at all costs** - sometimes fewer blocks with more shared memory is better
- **Use `__syncthreads()` carefully** - all threads in the block must reach it
- **Avoid divergent branches within warps** when possible

#### Performance Tips
- **Use fast math functions** (`__sinf`, `__expf`, etc.) when precision allows
- **Minimize register usage** to increase occupancy
- **Use texture memory for spatial locality** in read-only data
- **Consider using unified memory** for easier development, but profile performance
- **Profile your code** with nsys/ncu to identify bottlenecks

#### Error Handling
- **Always check CUDA API return values** using error checking macros
- **Use `cudaGetLastError()`** after kernel launches to catch errors
- **Synchronize before error checking** for kernel execution errors
- **Use CUDA-GDB** for debugging device code when needed

This cheat sheet covers the essential CUDA programming concepts and patterns. For the most current information, always refer to the [official CUDA documentation](https://docs.nvidia.com/cuda/).