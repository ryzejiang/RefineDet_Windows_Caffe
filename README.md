# RefineDet  Windows  Caffe  VisualStudio-15
在 Windows10 VisualStudio-15 上配置成功，能够直接使用 Ubuntu 下训练好的模型进行估计，CPU和GPU均可。
**使用方法等用空了再补**

## Windows Setup
### Requirements
 - Visual Studio 2013 (**Only with Python2.7**) or 2015 (**with Python 2.7 and Python 3.5**)
 - [CMake](https://cmake.org/) 3.4 or higher (Visual Studio and [Ninja](https://ninja-build.org/) generators are supported)

### Dependencies
 - Python for the pycaffe interface. Anaconda Python 2.7 or 3.5 x64 (or Miniconda)
 - Matlab for the matcaffe interface.
 - CUDA 7.5 or 8.0 (use CUDA 8 if using Visual Studio 2015)
 - cuDNN v5
 We assume that `cmake.exe` and `python.exe` are on your `PATH`.

### Building only for CPU

### Building only for GPU

## 可能对你有帮助的博客链接
[Caffe-ssd 在 windows 下的配置，及 python 调用](https://blog.csdn.net/Chris_zhangrx/article/details/83317721)
[Windows 下用 build_win.cmd 直接编译CPU版caffe](https://blog.csdn.net/Chris_zhangrx/article/details/79096015)
[Windows 下用 build_win.cmd 直接编译GPU版caffe](https://blog.csdn.net/Chris_zhangrx/article/details/83339684)
[Windows下 Pycaffe 的配置与使用](https://blog.csdn.net/Chris_zhangrx/article/details/79210288)