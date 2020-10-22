# RSMI

- [ ] change path for dataset, inserted points ... 
- [ ] mkdir for model saving
- [ ] explain trained model 
- [ ] recommend libtorch version
- [ ] add comments
- [ ] switch GPU and CPU 



###  How to use

#### 1. Download LibTorch
https://pytorch.org/get-started/locally/

#### 2. Change makefile

Choose CPU or GPU

Replace *home/liuguanli/Documents* with your own path.

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
#### 3. Change Exp.cpp

comment *#define use_gpu* to use CPU version

```C++
#ifndef use_gpu
#define use_gpu
.
.
.
#endif  // use_gpu
```


#### Prepare datasets

[dataset demo](./datasets/uniform_10000_1_2_.csv)

#### Run

```bash
make clean
make -f Makefile
./Exp -c 10000 -d uniform -s 1
```

### Our Paper

> Jianzhong Qi, Guanli Liu, Christian S. Jensen, Lars Kulik: [Effectively Learning Spatial Indices](http://www.vldb.org/pvldb/vol13/p2341-qi.pdf). Proc. VLDB Endow. 13(11): 2341-2354 (2020)

```tex
@article{DBLP:journals/pvldb/QiLJK20,
  author    = {Jianzhong Qi and
               Guanli Liu and
               Christian S. Jensen and
               Lars Kulik},
  title     = {Effectively Learning Spatial Indices},
  journal = {{PVLDB}}
  volume    = {13},
  number    = {11},
  pages     = {2341--2354},
  year      = {2020},
  url       = {http://www.vldb.org/pvldb/vol13/p2341-qi.pdf},
}
```

