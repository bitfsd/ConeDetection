# paddle inference ros

## Introduction

This package is based on `Python3` and implements `Paddle Inference` on ROS, which makes `Paddle Inference` a node for real-time detection. It can help developers deploy CV model written by `PaddlePaddle` in Ubuntu environment.

## Structure

```
-|ConeDetection				(ros package)
 |-install_scripts
   -cv_bridge/
   -install_ppyolo.sh
 |src/ppyolo
   |-config
     -ppyolo.yaml			(configeration file)
   |-launch
     -ppyolo.launch			(launch file)
   |-scripts
     -camera.py				(for test)
     -download_model.sh		(for test)
     -pp_infer.py			(main code)
```

## Paddle Inference

`Paddle Inference ` is the original reasoning library of the propeller, which provides high-performance reasoning ability.

Because the training operator is directly based on `PaddlePaddle`,  `Paddle Influence` can generally support all models trained by the `PaddlePaddle`.

`Paddle Inference` has rich functional features and excellent performance. It has carried out in-depth adaptation and Optimization for different application scenarios on different platforms, so as to achieve high throughput and low delay, and ensure that the propeller model can be trained and used on the server side and deployed quickly.

## High performance implementation of Paddle Inference

* High-performance **CPU / GPU kernel.**
* **Subgraph integration tensorrt** speeds up GPU inference.

## Usage

### 1. Environment Preparation

- ubuntu 18.04
- ROS Melodic
- python3.6.9（Default）
- paddlepaddle-gpu 2.1.1+ （URL：[paddle-inference prebulided whl](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/hardware_info_cn.html#paddle-inference) Or build from source：[Jetson Nano——基于python API部署Paddle Inference GPU预测库（2.1.1）](https://blog.csdn.net/qq_45779334/article/details/118611953)）
- CUDA
- CuDnn
- TensorRT

**This package is tested on linux with cuda10.2 cudnn8.1 tensorRT7 and** [paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.2-cp36-cp36m-linux_x86_64.whl)

### 2. Build cv_bridge for python3

On Ubuntu 18.04 ROS Melodic device:

```bash
$ cd install_scripts
$ bash install_ppyolo.sh
```

**Check：**

```
$ python3
```

```
import cv_bridge
from cv_bridge.boost.cv_bridge_boost import getCvType
```

**Successfully installed：**

```
Python 3.6.9 (default, Jan 26 2021, 15:33:00) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv_bridge
>>> from cv_bridge.boost.cv_bridge_boost import getCvType
>>> 
```

### 3. Build ConeDetection

```
$ cd ConeDetection/
$ catkin build
```

### 4. Run test node

**Download pre-trained yolov3 model:**

```
$ cd src/paddle_inference_ros/scripts $$ ./download_model.sh
```

**Run in three Terminal:**

```
$ roscore
$ rosrun ppyolo camera.py
$ roslaunch ppyolo pp_infer.py
```

## **5、Train your own datasets**
Please refer to the project on AI Studio: https://aistudio.baidu.com/aistudio/projectdetail/3428082  
AI Studio platform provides free V100 GPU, which can satisfy most detection missions.  
