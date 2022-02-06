# paddle inference ros
## **功能包介绍**
该功能包基于python3将原生的`Paddle Inference`嵌入ROS中的工具，使`Paddle Inference`能够作为一个节点实现对ROS中图像信息进行实时检测功能，可以帮助开发者在Ubuntu18.04 ROS Melodic环境中中使用`Paddle inference`部署基于飞桨的CV模型。

## **功能包架构**
```
* ConeDetection					(ros 功能包)
	* install_scripts
       * cv_bridge/
       * install_ppyolo.sh
    * src/ppyolo
       * config
         * ppyolo.yaml			(参数文件)
       * launch
         * ppyolo.launch		(启动文件)
       * scripts
         * camera.py			(测试代码)
         * download_model.sh	(测试模型)
         * pp_infer.py			(主程序)
```



## **Paddle Inference介绍**
`Paddle Inference`是飞桨的原生推理库，提供高性能的推理能力。
由于直接基于飞桨的训练算子，因此`Paddle Inference`可以通用支持飞桨训练出的所有模型。
`Paddle Inference`功能特性丰富，性能优异，针对不同平台不同的应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。

### **Paddle Inference的高性能实现**
- 内置高性能的**CPU/GPU Kernel**
- **子图集成TensorRT**加快GPU推理速度

## **1、环境准备**
- ubuntu 18.04
- ROS Melodic
- python3.6.9（系统默认）
- paddlepaddle-gpu 2.1.1+ （下载地址：[paddle-inference prebulided whl](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/hardware_info_cn.html#paddle-inference)或者参照博客：[Jetson Nano——基于python API部署Paddle Inference GPU预测库（2.1.1）](https://blog.csdn.net/qq_45779334/article/details/118611953)进行安装）
- CUDA
- TensorRT

**该功能包在Linux下，cuda10.2 cudnn8.1 tensorRT7环境下使用**[paddlepaddle-cp36m.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.2-cp36-cp36m-linux_x86_64.whl)进行测试

## **2、编译python3的cv_bridge**
在Ubuntu18.04 ROS Melodic设备上：
```bash
$ cd install_scripts
$ bash install_ppyolo.sh
```
**检查是否安装成功：**
```
$ python3
```
```
import cv_bridge
from cv_bridge.boost.cv_bridge_boost import getCvType
```
**如果显示如下，则表明安装成功：**

```
Python 3.6.9 (default, Jan 26 2021, 15:33:00) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv_bridge
>>> from cv_bridge.boost.cv_bridge_boost import getCvType
>>> 
```

## **3、编译ConeDetection功能包**
```
$ cd ConeDetection/
$ catkin build
```

## **4、运行节点**
**下载yolo_v3目标检测模型**

```
$ cd src/paddle_inference_ros/scripts $$ ./download_model.sh
```
**分别使用三个终端运行：**

```
$ roscore
$ rosrun ppyolo camera.py
$ roslaunch ppyolo pp_infer.py
```

