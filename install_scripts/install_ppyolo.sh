#!/bin/bash

sudo mv /opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge /opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge_old

sudo cp -r cv_bridge/ /opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge

echo "source ~/ConeDetection/devel/setup.bash" >> ~/.bashrc
echo "source ~/ConeDetection/install/setup.bash --extend" >> ~/.bashrc
source ~/.bashrc
