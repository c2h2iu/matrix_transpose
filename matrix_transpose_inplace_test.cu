#include <iostream>
#include <vector>
#include <string>
#include <taskflow/taskflow.hpp>
#include <algorithm>
#include <numeric>



__global__ void kernel_1(int* d_A, const int row, const int column, const int b){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int temp[2][4];
   
    if(tid < column){
        for(int i = 0; i < row; ++i){
            temp[threadIdx.x][i] = d_A[( (i + tid/b)%row )*column + tid];
	}
	for(int i = 0; i < row; ++i){
            d_A[i*column+tid] = temp[threadIdx.x][i];
	}
    }
}


__global__ void kernel_2(int* d_A, const int row, const int column, const int b){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int temp[2][8];

    if(tid < row){
        for(int j = 0; j < column; ++j){
            temp[threadIdx.x][((tid + j/b)%row + (j * row))%column] = d_A[tid * column + j];
	}
	for(int j = 0; j < column; ++j){
            d_A[tid * column + j] = temp[threadIdx.x][j];
	}
    }
}



__global__ void kernel_3(int* d_A, const int row, const int column, const int a){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int temp[2][4];

    if(tid < column){
        for(int i = 0; i < row; ++i){
            temp[threadIdx.x][i] = d_A[((tid + i*column - (i/a))%row)*column + tid];
	}
	for(int i = 0; i < row; ++i){
            d_A[i * column + tid] = temp[threadIdx.x][i];
	}
    }
}


void gpu(){
    int row = 4, column = 8;

    int N = row * column;
    int size = N * sizeof(int);

    int mygcd = std::gcd(row, column);
    int parameter_a = row/mygcd;
    int parameter_b = column/mygcd;

    std::vector<int> h_A(N, 0);
    std::vector<int> h_B(N, 0);

    for(int i = 0; i < N; ++i){
	if(i/column == 0)    h_A[i] = i * 4;
	else    h_A[i] = (i/column) + (i%column) * 4;
	    
    }

    int* d_A;

    cudaMalloc((void **)&d_A, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    

    kernel_1<<<column/parameter_b, parameter_b>>>(d_A, row, column, parameter_b);
    cudaMemcpy(h_B.data(), d_A, size, cudaMemcpyDeviceToHost);

    /***
    for(int i = 0; i < N; ++i){
        std::cout << h_B[i] << ' ';
        if((i+1)%column == 0)    std::cout << '\n';
    }
    std::cout << '\n';
    ***/

    kernel_2<<<row/parameter_b, parameter_b>>>(d_A, row, column, parameter_b);
    cudaMemcpy(h_B.data(), d_A, size, cudaMemcpyDeviceToHost);

    /***
    for(int i = 0; i < N; ++i){
	std::cout << h_B[i] << ' ';
        if((i+1)%column == 0)    std::cout << '\n';
    }
    std::cout << '\n';
    ***/
    
    kernel_3<<<column/parameter_b, parameter_b>>>(d_A, row, column, parameter_a);
    cudaMemcpy(h_B.data(), d_A, size, cudaMemcpyDeviceToHost);

    /***
    for(int i = 0; i < N; ++i){
	std::cout << h_B[i] << ' ';
        if((i+1)%column == 0)    std::cout << '\n';
    }
    std::cout << '\n';
    ***/

    cudaFree(d_A);
    
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




int main(int argc, char* argv[]){ 

    gpu();

    return 0;
}

