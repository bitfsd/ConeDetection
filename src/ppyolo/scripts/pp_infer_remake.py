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


class PPyoloException(Exception):
    pass

class PPinfer:

    def __init__(self):
        rospy.init_node('rospy_pp_infer', anonymous=True)
        self.load_params()

        self.pub_image = Image()
        self.pub_boundingbox = BoundingBoxes()
        self.pub_object = ObjectCount()
        self.bridge = CvBridge()

        model_dir = self.Parameters['model_dir']
        types = self.Parameters['type']
        if types == 'paddlex':
            infer_cfg = open(model_dir + 'model.yml')
            data = infer_cfg.read()
            yaml_reader = yaml.load(data)
            self.label_list = yaml_reader['_Attributes']['labels']
        else:
            infer_cfg = open(model_dir + 'infer_cfg.yml')
            data = infer_cfg.read()
            yaml_reader = yaml.load(data)
            self.label_list = yaml_reader['label_list']

        print(self.label_list)

        model_file = model_dir + "model.pdmodel"
        params_file = model_dir + "model.pdiparams"
        
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except AttributeError:
            img = np.zeros((224,224,3), np.uint8)
        self.im_size = self.Parameters['img_size']
        self.scale_factor = np.array([self.im_size * 1. / img.shape[0], self.im_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
        self.im_shape = np.array([self.im_size, self.im_size]).reshape((1, 2)).astype(np.float32)
        self.predictor = self.predict_config(model_file, params_file)

        self.subscribe()
        self.publish()

        rate = rospy.Rate(self.Parameters['node_rate'])
        
        while(not rospy.is_shutdown()):
            self.sendMsg()
            rate.sleep()
        rospy.spin()
    
    def load_params(self):
        print('loading parameters...')
        if not rospy.has_param('Subscriber'):
            raise PPyoloException('No Subscriber configuration found')
        self.Subscriber = rospy.get_param('Subscriber')
        if not rospy.has_param('Publisher'):
            raise PPyoloException('No Publisher configuration found')
        self.Publisher = rospy.get_param('Publisher')
        if not rospy.has_param('Publisher'):
            raise PPyoloException('No Parameter configuration found')
        self.Parameters = rospy.get_param('Parameter')
    
    def subscribe(self):
        rospy.Subscriber(self.Subscriber['camera_topic_name'], Image, self.predict_img)
    
    def publish(self):
        self.image_pub=rospy.Publisher(self.Publisher['detected_image_topic_name'], Image, queue_size=1) #定义话题
        self.boundingbox_pub = rospy.Publisher(self.Publisher['boundingbox_topic_name'], BoundingBoxes, queue_size=1)
        self.object_pub = rospy.Publisher(self.Publisher['object_topic_name'], ObjectCount, queue_size=1)

    def resize(self, img, target_size):
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

    def normalize(self, img, mean, std):
        img = img / 255.0
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        img -= mean
        img /= std
        return img

    def preprocess(self, img, img_size):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = self.resize(img, img_size)
        resize_img = img
        img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
        img = self.normalize(img, mean, std)
        img = img.transpose((2, 0, 1))  # hwc -> chw
        return img[np.newaxis, :], resize_img

    def predict_config(self, model_file, params_file):
        config = Config()
        config.set_prog_file(model_file)
        config.set_params_file(params_file)
        if self.Parameters['use_gpu']:
            config.enable_use_gpu(400, 0)
        config.switch_ir_optim()
        config.enable_memory_optim()
        if self.Parameters['use_tensorrt']:
            if self.Parameters['tensorrt_type'] == 'FP16':
                config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=PrecisionType.Half,max_batch_size=1, min_subgraph_size=5, use_static=True, use_calib_mode=False)
            if self.Parameters['tensorrt_type'] == 'FP32':
                config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=PrecisionType.Float32,max_batch_size=1, min_subgraph_size=5, use_static=True, use_calib_mode=False)
            if self.Parameters['tensorrt_type'] == 'INT8':
                config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=PrecisionType.Int8,max_batch_size=1, min_subgraph_size=5, use_static=True, use_calib_mode=True)
        predictor = create_predictor(config)
        return predictor

    def predict(self, predictor, img):
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)
            input_tensor.copy_from_cpu(img[i].copy())
        # 执行Predictor
        predictor.run()
        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results

    def draw_bbox(self, frame, result, label_list, shape, threshold=0.5):
        height, weight, _ = shape
        for res in result:
            cat_id, score, bbox = res[0], res[1], res[2:]
            if score < threshold:
                continue
            xmin, ymin, xmax, ymax = bbox / 224 * self.im_size
            xmin = xmin / self.im_size * weight
            xmax = xmax / self.im_size * weight
            ymin = ymin / self.im_size * height
            ymax = ymax / self.im_size * height
            cv2.rectangle(frame, (int(xmin ), int(ymin)), (int(xmax), int(ymax)), (255,0,255), 2)
            label_id = label_list[int(cat_id)]
            print('label is {}, bbox is {}'.format(label_id, [int(xmin), int(xmax), int(ymin), int(ymax)]))
            try:
                cv2.putText(frame, label_id, (int(xmin+10), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 / self.im_size * height, (255,255,255), 2)
                cv2.putText(frame, str(round(score,2)), (int(xmin-35), int(ymin-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 / self.im_size * height, (0,255,0), 2)
            except KeyError:
                pass


    def predict_img(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        ros_frame_ = Image()
        header = Header(stamp = rospy.Time.now())
        header.frame_id = "Camera"
        ros_frame_.header=header
        ros_frame_.width = img.shape[1]
        ros_frame_.height = img.shape[0]
        ros_frame_.encoding = "bgr8"
        ros_frame_.step = 1920
        ros_frame_.data = np.array(img).tostring()
        cv_img = self.bridge.imgmsg_to_cv2(ros_frame_, "bgr8")
        img_data, cv_img_ = self.preprocess(cv_img, self.im_size)      
        t1 = time.time()
        result = self.predict(self.predictor, [self.im_shape, img_data, self.scale_factor])
        t2 = time.time()
        os.system("clear")
        print("detect speed: ",int(1/(t2-t1)), "fps")
        ros_frame = Image()
        im_scale_x = float(cv_img.shape[1]) / float(cv_img_.shape[1])
        im_scale_y = float(cv_img.shape[0]) / float(cv_img_.shape[0])
        img = cv2.resize(cv_img_, None, None, fx=im_scale_x, fy=im_scale_y)
        threshold = 0.5
        self.draw_bbox(img, result[0], self.label_list, cv_img.shape, threshold=threshold)
        header = Header(stamp = rospy.Time.now())
        header.frame_id = "Camera"
        ros_frame.header=header
        ros_frame.width = img.shape[1]
        ros_frame.height = img.shape[0]
        ros_frame.encoding = "bgr8"
        ros_frame.step = 1920
        ros_frame.data = np.array(img).tostring() #图片格式转换
        # image_pub.publish(ros_frame) #发布消息
        # global pub_image
        self.pub_image = ros_frame
        
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
            boundingBox.Class = self.label_list[int(cat_id)]
            box.append(boundingBox)
            boundingBoxedResults_.bounding_boxes = box
            boundingBoxedResults_.image_header = msg.header
            boundingBoxedResults_.header.stamp = boundingBoxedResults_.image_header.stamp
        self.pub_boundingbox = boundingBoxedResults_
        object_count = ObjectCount()
        object_count.header = msg.header
        object_count.count = object_c
        self.pub_object = object_count

    def sendMsg(self):
        self.image_pub.publish(self.pub_image)
        self.object_pub.publish(self.pub_object)
        self.boundingbox_pub.publish(self.pub_boundingbox)



if __name__ == '__main__':
    print("inference begining...")
    infer = PPinfer()