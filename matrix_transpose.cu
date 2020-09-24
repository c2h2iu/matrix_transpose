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
