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


cmake_minimum_required(VERSION 3.0.2)
project(ppyolo)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosmsg
  rospy
)

catkin_python_setup()

catkin_package(
)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
