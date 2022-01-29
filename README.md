# paddle_inference_ros
## **功能包介绍**
该功能包是全网首个能够完全基于python3将原生的`Paddle Inference`嵌入ROS中的工具，使`Paddle Inference`能够作为一个节点实现对ROS中图像信息进行实时检测功能，可以帮助开发者在Ubuntu18.04 ROS Melodic环境中中使用`Paddle inference`部署基于飞桨的CV模型。

## **功能包架构**
- paddle_inference_ros(ros package)
    - scripts
        - `camera.py`(camera_node)
        - `pp_infer.py`(ppinfer_node)
        - `download_model.sh`

## **Paddle Inference介绍**
`Paddle Inference`是飞桨的原生推理库，提供高性能的推理能力。
由于能力直接基于飞桨的训练算子，因此`Paddle Inference`可以通用支持飞桨训练出的所有模型。
`Paddle Inference`功能特性丰富，性能优异，针对不同平台不同的应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。
### **Paddle Inference的高性能实现**
- 内置高性能的**CPU/GPU Kernel**
- **子图集成TensorRT**加快GPU推理速度
### **主流软硬件环境兼容适配**
- 支持服务器端`X86 CPU`、`NVIDIA GPU`芯片 **(包括Jetson系列的GPU)**，兼容`Linux/Mac/Windows`系统。支持所有飞桨训练产出的模型，完全做到即训即用。
### **多语言环境丰富接口可灵活调用**
- 支持`C++`、`Python`、`C`，接口简单灵活，20行代码即可完成部署。对于其他语言，提供了ABI稳定的C API, 用户可以很方便地扩展。

## **1、环境准备**
- ubuntu 18.04
- ROS Melodic
- python3.6.9（系统默认）
- paddlepaddle-gpu 2.1.1+ （下载地址：[paddle-inference prebulided whl](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/hardware_info_cn.html#paddle-inference)或者参照博客：[Jetson Nano——基于python API部署Paddle Inference GPU预测库（2.1.1）](https://blog.csdn.net/qq_45779334/article/details/118611953)进行安装）

## **2、编译python3的cv_bridge**
在ROS中想使用原生`python3`的`Paddle Inference`，最重要的就是需要重新编译基于`python3`的`cv_bridge`，只有我们在编译完成后，才能在ROS中运行`python3`的`Paddle Inference`目标检测、分类、分割等相关节点时，自动调用基于`python3`的`cv_bridge`。所以**编译基于`python3`的`cv_bridge`便是最基础和最重要的一步，按以下步骤进行操作：**
```
$ mkdir -p paddle_ros_ws/src && cd paddle_ros_ws/src
$ catkin_init_workspace
$ git clone https://gitee.com/irvingao/vision_opencv.git
$ cd ../
$ catkin_make install -DPYTHON_EXECUTABLE=/usr/bin/python3
```
```
$ vim ~/.bashrc
```
**在最后添加：**
```
source ~/paddle_ros_ws/devel/setup.bash
source ~/paddle_ros_ws/install/setup.bash --extend
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

## **2、编译paddle_inference_ros功能包**
```
cd src
$ git clone https://gitee.com/irvingao/paddle_inference_ros.git
$ cd paddle_inference_ros/scripts
$ chmod +x *
$ cd ../../..
$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

## **3、运行节点**
**下载yolo_v3目标检测模型**
```
$ cd src/paddle_inference_ros/scripts $$ ./download_model.sh
```
**分别使用三个终端运行：**
```
$ roscore
$ rosrun paddle_inference_ros camera.py
$ rosrun paddle_inference_ros pp_infer.py
```

**可以成功在ROS中运行paddle inference，并实现GPU和TensorRT的加速！**

![在这里插入图片描述](https://img-blog.csdnimg.cn/dc3c6eac8ab64018adcfe2adaf7baf5b.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1Nzc5MzM0,size_16,color_FFFFFF,t_70)

