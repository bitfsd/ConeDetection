#!/usr/bin/env python3
# coding:utf-8
#
# Formula Student Driverless Project (FSD-Project).
# Copyright (c) 2022:
#  - FengYunJi <yunjifeng@bitfsd.cn>
# FSD-Project is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# FSD-Project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with FSD-Project.  If not, see <https://www.gnu.org/licenses/>.

from ast import Sub
from inspect import Parameter
import cv2
import numpy as np
from rospy.core import is_shutdown
from std_msgs.msg import Header
from fsd_common_msgs.msg import BoundingBox
from fsd_common_msgs.msg import BoundingBoxes
from fsd_common_msgs.msg import ObjectCount

import message_filters
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import os

from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor
import yaml
import time

pub_image = Image()
pub_boundingbox = BoundingBoxes()
pub_object = ObjectCount()

# ————————————————图像预处理函数———————————————— #

def resize(img, target_size):
    sdfklsjdflk
    """resize to target size"""
    if not isinstance(img, np.ndarray):
        raise TypeError('image type is not numpy.')
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale_x = float(target_size) / float(im_shape[1])
    im_scale_y = float(target_size) / float(im_shape[0])
    img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y)
    return img

def normalize(img, mean, std):
    img = img / 255.0
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std
    return img

def preprocess(img, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = resize(img, img_size)
    resize_img = img
    img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
    img = normalize(img, mean, std)
    img = img.transpose((2, 0, 1))  # hwc -> chw
    return img[np.newaxis, :], resize_img

# ——————————————————————模型配置、预测相关函数—————————————————————————— #
def predict_config(model_file, params_file):
    '''
    函数功能：初始化预测模型predictor
    函数输入：模型结构文件，模型参数文件
    函数输出：预测器predictor
    '''
    # 根据预测部署的实际情况，设置Config
    config = Config()
    # 读取模型文件
    config.set_prog_file(model_file)
    config.set_params_file(params_file)
    # Config默认是使用CPU预测，若要使用GPU预测，需要手动开启，设置运行的GPU卡号和分配的初始显存。
    config.enable_use_gpu(400, 0)
    # 可以设置开启IR优化、开启内存优化。
    config.switch_ir_optim()
    config.enable_memory_optim()
    config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=PrecisionType.Half,max_batch_size=1, min_subgraph_size=5, use_static=True, use_calib_mode=False)
    predictor = create_predictor(config)
    return predictor

def predict(predictor, img):
    
    '''
    函数功能：初始化预测模型predictor
    函数输入：模型结构文件，模型参数文件
    函数输出：预测器predictor
    '''
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())
    # 执行Predictor
    predictor.run()
    # 获取输出
    results = []
    # 获取输出
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results

# ——————————————————————后处理函数—————————————————————————— #

def draw_bbox(frame, result, label_list, shape, threshold=0.5):
    height, weight, _ = shape
    for res in result:
        cat_id, score, bbox = res[0], res[1], res[2:]
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = bbox / 224 * im_size
        xmin = xmin / im_size * weight
        xmax = xmax / im_size * weight
        ymin = ymin / im_size * height
        ymax = ymax / im_size * height
        cv2.rectangle(frame, (int(xmin ), int(ymin)), (int(xmax), int(ymax)), (255,0,255), 2)
        label_id = label_list[int(cat_id)]
        print('label is {}, bbox is {}'.format(label_id, [int(xmin), int(xmax), int(ymin), int(ymax)]))
        try:
            # #cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
            cv2.putText(frame, label_id, (int(xmin+10), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 / im_size * height, (255,255,255), 2)
            cv2.putText(frame, str(round(score,2)), (int(xmin-35), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 / im_size * height, (0,255,0), 2)
            # cv2.putText(frame, str(label_id), (int(xmin-35), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 / im_size * height, (0,255,0), 2)
        except KeyError:
            pass


def predict_img(msg):
    img = bridge.imgmsg_to_cv2(msg, "bgr8")

    ros_frame_ = Image()
    header = Header(stamp = rospy.Time.now())
    header.frame_id = "Camera"
    ros_frame_.header=header
    ros_frame_.width = img.shape[1]
    ros_frame_.height = img.shape[0]
    ros_frame_.encoding = "bgr8"
    ros_frame_.step = 1920
    ros_frame_.data = np.array(img).tostring() #图片格式转换

    global  predictor, im_size, im_shape, scale_factor, label_list
    cv_img = bridge.imgmsg_to_cv2(ros_frame_, "bgr8")
    img_data, cv_img_ = preprocess(cv_img, im_size)      
    # print(data.header.stamp)  
    # 预测
    t1 = time.time()
    result = predict(predictor, [im_shape, img_data, scale_factor])
    t2 = time.time()
    os.system("clear")
    print("detect speed: ",int(1/(t2-t1)), "fps")


    ros_frame = Image()

    im_scale_x = float(cv_img.shape[1]) / float(cv_img_.shape[1])
    im_scale_y = float(cv_img.shape[0]) / float(cv_img_.shape[0])
    img = cv2.resize(cv_img_, None, None, fx=im_scale_x, fy=im_scale_y)
    threshold = 0.5
    draw_bbox(img, result[0], label_list, cv_img.shape, threshold=threshold)
    header = Header(stamp = rospy.Time.now())
    header.frame_id = "Camera"
    ros_frame.header=header
    ros_frame.width = img.shape[1]
    ros_frame.height = img.shape[0]
    ros_frame.encoding = "bgr8"
    ros_frame.step = 1920
    ros_frame.data = np.array(img).tostring() #图片格式转换
    # image_pub.publish(ros_frame) #发布消息
    global pub_image
    pub_image = ros_frame
    
    # Publish bounding boxes
    boundingBoxedResults_ = BoundingBoxes()
    boundingBoxedResults_.header.frame_id = "detection"

    height, weight, _ = cv_img.shape
    box = []
    object_c = 0
    for res in result[0]:
        cat_id, score, bbox = res[0], res[1], res[2:]
        if score < threshold:
            continue
        boundingBox = BoundingBox()
        xmin, ymin, xmax, ymax = bbox
        xmin = xmin / 224 * weight
        xmax = xmax / 224 * weight
        ymin = ymin / 224 * height
        ymax = ymax / 224 * height
        object_c += 1
        boundingBox.xmin = int(xmin)
        boundingBox.xmax = int(xmax)
        boundingBox.ymin = int(ymin)
        boundingBox.ymax = int(ymax)
        boundingBox.probability = score
        boundingBox.id = int(cat_id)
        boundingBox.Class = label_list[int(cat_id)]
        box.append(boundingBox)
        boundingBoxedResults_.bounding_boxes = box
        boundingBoxedResults_.image_header = msg.header
        boundingBoxedResults_.header.stamp = boundingBoxedResults_.image_header.stamp
    # boundingbox_pub.publish(boundingBoxedResults_)
    global pub_boundingbox
    pub_boundingbox = boundingBoxedResults_

    # Publish found_object
    object_count = ObjectCount()
    object_count.header = msg.header
    object_count.count = object_c
    # object_pub.publish(object_count)
    global pub_object
    pub_object = object_count

def sendMsg():
    image_pub.publish(pub_image)
    object_pub.publish(pub_object)
    boundingbox_pub.publish(pub_boundingbox)

class PPyoloException(Exception):
    pass

def init():
    if not rospy.has_param('Subscriber'):
        raise PPyoloException('No Subscriber configuration found')
    Subscriber = rospy.get_param('Subscriber')
    if not rospy.has_param('Publisher'):
        raise PPyoloException('No Publisher configuration found')
    Publisher = rospy.get_param('Publisher')
    if not rospy.has_param('Publisher'):
        raise PPyoloException('No Parameter configuration found')
    Parameters = rospy.get_param('Parameter')
    
    return Subscriber, Publisher, Parameters


if __name__ == '__main__':
    import sys 
    print(sys.version) # 查看python版本
    Subscriber, Publisher, Parameters = init()
    # 初始化节点
    rospy.init_node('rospy_pp_infer', anonymous=True)
    bridge = CvBridge()

    model_dir = Parameters['model_dir']
    types = Parameters['type']
    # 从infer_cfg.yml中读出label
    if types == 'paddlex':
        infer_cfg = open(model_dir + 'model.yml')
        data = infer_cfg.read()
        yaml_reader = yaml.load(data)
        label_list = yaml_reader['_Attributes']['labels']
    else:
        infer_cfg = open(model_dir + 'infer_cfg.yml')
        data = infer_cfg.read()
        yaml_reader = yaml.load(data)
        label_list = yaml_reader['label_list']
    print(label_list)

    # 配置模型参数
    model_file = model_dir + "model.pdmodel"
    params_file = model_dir + "model.pdiparams"
    
    # 图像尺寸相关参数初始化
    try:
        img = bridge.imgmsg_to_cv2(data, "bgr8")
    except AttributeError:
        img = np.zeros((224,224,3), np.uint8)
    im_size = Parameters['img_size']
    scale_factor = np.array([im_size * 1. / img.shape[0], im_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)

    # 初始化预测模型
    predictor = predict_config(model_file, params_file)

    rospy.Subscriber(Subscriber['camera_topic_name'], Image, predict_img)

    image_pub=rospy.Publisher(Publisher['detected_image_topic_name'], Image, queue_size=1) #定义话题
    boundingbox_pub = rospy.Publisher(Publisher['boundingbox_topic_name'], BoundingBoxes, queue_size=1)
    object_pub = rospy.Publisher(Publisher['object_topic_name'], ObjectCount, queue_size=1)
    rate = rospy.Rate(Parameters['node_rate'])
    while(not rospy.is_shutdown()):
        sendMsg()
        rate.sleep()
    rospy.spin()

