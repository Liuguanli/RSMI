# RSMI

### LibTorch
https://pytorch.org/get-started/locally/

### change makefile and CPU or GPU version

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

comment *#define use_gpu* to use CPU version

```C++
#ifndef use_gpu
#define use_gpu
.
.
.
#endif  // use_gpu
```


### About datasets

[dataset demo](./datasets/uniform_10000_1_2_.csv)

### how to test



### Paper Link

