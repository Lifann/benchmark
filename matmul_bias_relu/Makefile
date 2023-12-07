matmul_bias_relu: matmul_bias_relu.cu
	nvcc -std=c++14 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -o matmul_bias_relu matmul_bias_relu.cu

clean:
	rm matmul_bias_relu
