# RSMI

### LibTorch
https://pytorch.org/get-started/locally/

### change makefile

Use CPU version or GPU according to you LibTorch library.

```
# TYPE = CPU
TYPE = GPU

ifeq ($(TYPE), GPU)
	INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
else
	INCLUDE = -I/home/liuguanli/Documents/libtorch/include -I/home/liuguanli/Documents/libtorch/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch/lib
endif
```

### how to generate datasets

[dataset demo](./datasets/uniform_10000_1_2_.csv)

### how to choose CPU or GPU version

### how to test

### Paper Link

### 
