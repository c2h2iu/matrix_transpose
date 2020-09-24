#include <iostream>
#include <vector>
#include <string>
#include <taskflow/taskflow.hpp>
#include <algorithm>
#include <numeric>


int column;
int row;
int dimension{0};


/***
   first argument is the approach
   0: cpu_outplace, 1: cpu_inplace, 2: gpu_outplace, 3: gpu_inplace
   second argument is row
   third argument is column
***/


void matrix_transpose_cpu_outplace(){
    unsigned long int N = row * column;
    std::vector<int> h_A(N, 0);
    std::vector<int> h_B(N, 0);


    for(unsigned long int i = 0; i < N; ++i)    h_A[i] = (i/column);

   
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < column; ++j){
	    h_B[i + j * row] = h_A[i * column + j];
	}
    }
   
    /***        
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < column; ++j){
	    std::cout << h_A[i*column + j] << ' ' ; 
	}
	std::cout << '\n';
    }

    std::cout << "\n\n";

    for(int i = 0; i < column; ++i){
        for(int j = 0; j < row; ++j){
	    std::cout << h_B[j + i * row] << ' ' ; 
	}
	std::cout << '\n';
    }
    ***/
}



void matrix_transpose_cpu_inplace(){
    unsigned long int N = row * column;
    std::vector<int> h_A(N, 0);
    std::vector<bool> visited(N, false);
    std::vector<unsigned long int>heads;
    int temp = 0;
    int* p_h_A = h_A.data();
    
    for(unsigned long int i = 0; i < N; ++i)    *(p_h_A + i) = (i/column);

    /***
    for(int i = 0; i < N; ++i){
	std::cout << *(p_h_A + i) << ' ' ;
        if((i+1)%column == 0)    std::cout << '\n';
    }

    std::cout << "\n\n";
    ***/

    unsigned long int index = 1, cycleHead = 0, next = 0;
    while(index < (N-1)){
        cycleHead = index;
        temp = *(p_h_A + index);
	heads.push_back(index);

	do{
            next = (index*row)%(N-1);
	    std::swap(*(p_h_A + next), temp);
	    visited[index] = true;
	    index = next;
	}while(index != cycleHead);

	for(index = 1; index < (N-1) && visited[index]; ++index);
    }
    
    /*** 
    for(int i = 0; i < N; ++i){
        std::cout << *(p_h_A + i) << ' ' ;
        if((i+1)%row == 0)    std::cout << '\n';	
    }
    ***/
    for(unsigned long int i = 0; i < heads.size(); ++i)
        std::cout << heads[i] << '\n';
}


__global__ void kernel_outplace(int* d_A, int* d_B, const int row, const int column){
    int d_row = threadIdx.y + blockIdx.y * blockDim.y;
    int d_col = threadIdx.x + blockIdx.x * blockDim.x;
    
    if((d_row < row) && (d_col < column))
        d_B[d_col * row + d_row] = d_A[d_row * column + d_col];
}


__global__ void kernel_inplace(int* d_A, const int row, const int column){
    int d_row = threadIdx.y + blockIdx.y * blockDim.y;
    int d_col = threadIdx.x + blockIdx.x * blockDim.x;


}




void matrix_transpose_gpu(const int approach){
    unsigned long int N = row * column;
    unsigned long int size = N * sizeof(int);

    int gcd = std::gcd(row, column);
    int parameter_a = row/gcd;
    int parameter_b = column/gcd;

    std::vector<int> h_A(N, 0);
    std::vector<int> h_B(N, 1);

    for(unsigned long int i = 0; i < N; ++i)    h_A[i] = (i/column);
        
    int* d_A;
    int* d_B;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((column-1)/16+1, (row-1)/16+1);

    if(approach == 2)    kernel_outplace<<<dimGrid, dimBlock>>>(d_A, d_B, row, column);
    if(approach == 3){    
        kernel_inplace<<<dimGrid, dimBlock>>>(d_A, row, column);
    }
	
    cudaMemcpy(h_B.data(), d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    /***        
    for(int i = 0; i < N; ++i){    
        std::cout << h_A[i] << ' ';
	if((i+1) % column == 0)    std::cout << '\n';
    }
    std::cout << "\n\n\n";
   
    for(int i = 0; i < N; ++i){    
        std::cout << h_B[i] << ' ';
	if((i+1)%row == 0)    std::cout << '\n';
    }
    ***/
}



void matrix_multiplication_parallel_for(){
    std::vector<std::vector<int>> h_A(dimension, std::vector<int>(dimension, 1));
    std::vector<std::vector<int>> h_B(dimension, std::vector<int>(dimension, 1));
    std::vector<std::vector<int>> h_C(dimension, std::vector<int>(dimension, 0));

    tf::Taskflow taskflow;
    tf::Executor executor;

    tf::Task task = taskflow.for_each_index(0, dimension, 1, [&] (int i){
        for(int j = 0; j < dimension; ++j){
            for(int k = 0; k < dimension; ++k){
                h_C[i][j] += h_A[i][k] * h_B[k][j];
            }
        }
    });
    executor.run(taskflow).wait();
}


__global__ void kernel_global_memory(int* d_A, int* d_B, int* d_C, const int dimension){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int result = 0;

    for(int i = 0; i < dimension; ++i)
        result += d_A[y * dimension + i] * d_B[i * dimension + x];

    d_C[y * dimension + x] = result;
}


__global__ void kernel_shared_memory(int* d_A, int* d_B, int* d_C, const int dimension){
    __shared__ int s_d_A[32][32];
    __shared__ int s_d_B[32][32];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int result = 0;

    for(int i = 0; i < dimension/32; ++i){
        s_d_A[threadIdx.y][threadIdx.x] = d_A[y * dimension + (i * 32 + threadIdx.x)];
	s_d_B[threadIdx.y][threadIdx.x] = d_B[(i * 32 + threadIdx.y) * dimension + x];

	__syncthreads();

	for(int i = 0; i < 32; ++i)
	    result += s_d_A[threadIdx.y][i] * s_d_B[i][threadIdx.x];
	__syncthreads();
    }

    d_C[y * dimension + x] = result;
}



void matrix_multiplication_gpu(const int approach){
    int size = dimension * dimension * sizeof(int);
    std::vector<int> h_A(dimension*dimension, 1);
    std::vector<int> h_B(dimension*dimension, 1);
    std::vector<int> h_C(dimension*dimension, 0);

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(32, 32);
    dim3 dimGrid(dimension/32, dimension/32);

    if(approach == 1)    kernel_global_memory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, dimension);
    if(approach == 2)    kernel_shared_memory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, dimension);

    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    /***
    for(int i = 0; i < dimension*dimension; ++i){
	if(i%dimension == 0)    std::cout << '\n';
        std::cout << h_C[i] << ' ';
    }
    ***/
}



void matrix_multiplication_taskflow_gpu(){
    tf::Taskflow taskflow;
    tf::Executor executor;
    std::vector<int> h_A, h_B, h_C;
    int* d_A;
    int* d_B;
    int* d_C;

    auto allocate_a = taskflow.emplace([&](){
	h_A.resize(dimension*dimension, dimension+dimension);
        TF_CHECK_CUDA(cudaMalloc(&d_A, dimension*dimension*sizeof(int)), "failed to allocate a");	
    }).name("allocate_a");

    auto allocate_b = taskflow.emplace([&](){
	h_B.resize(dimension*dimension, dimension+dimension);
        TF_CHECK_CUDA(cudaMalloc(&d_B, dimension*dimension*sizeof(int)), "failed to allocate b");	
    }).name("allocate_b");

    auto allocate_c = taskflow.emplace([&](){
	h_C.resize(dimension*dimension, dimension+dimension);
        TF_CHECK_CUDA(cudaMalloc(&d_C, dimension*dimension*sizeof(int)), "failed to allocate c");	
    }).name("allocate_c");

    auto cudaFlow = taskflow.emplace([&](tf::cudaFlow& cf){
        auto copy_da = cf.copy(d_A, h_A.data(), dimension*dimension).name("HostToDevice_a");
        auto copy_db = cf.copy(d_B, h_B.data(), dimension*dimension).name("HostToDevice_b");
        auto copy_hc = cf.copy(h_C.data(), d_C, dimension*dimension).name("DeviceToHost_c");
    
	dim3 dimGrid(dimension/32, dimension/32);
	dim3 dimBlock(32, 32);

	auto kmatmul = cf.kernel(dimGrid, dimBlock, 0, kernel_global_memory, d_A, d_B, d_C, dimension).name("matmul");

	kmatmul.succeed(copy_da, copy_db).precede(copy_hc);
    }).name("cudaFlow");

    auto free = taskflow.emplace([&](){
        TF_CHECK_CUDA(cudaFree(d_A), "failed to free d_A");
        TF_CHECK_CUDA(cudaFree(d_B), "failed to free d_B");
        TF_CHECK_CUDA(cudaFree(d_C), "failed to free d_C");	
    }).name("free");

    cudaFlow.succeed(allocate_a, allocate_b, allocate_c).precede(free);
    executor.run(taskflow).wait();
}



int main(int argc, char* argv[]){ 
    int approach = std::stoi(argv[1]);
    row = std::stoi(argv[2]);
    column = std::stoi(argv[3]);

    if(approach == 0)    matrix_transpose_cpu_outplace();
    if(approach == 1)    matrix_transpose_cpu_inplace();
    if(approach == 2)    matrix_transpose_gpu(approach);
    if(approach == 3)    matrix_transpose_gpu(approach);


    return 0;
}

