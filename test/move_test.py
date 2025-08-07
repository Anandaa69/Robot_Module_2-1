# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import robomaster
from robomaster import robot


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_gimbal = ep_robot.gimbal
    #ep_gimbal.recenter().wait_for_completed()

    # 控制云台向左旋转100度
    ep_gimbal.move(pitch=0, yaw=0,yaw_speed = 250).wait_for_completed()
    ep_blaster = ep_robot.blaster

    # 发射1颗水弹
    ep_blaster.fire(times=5)
    time.sleep(0.5)
    
    # ep_gimbal.move(pitch=0, yaw=31,yaw_speed = 250).wait_for_completed()
    # ep_blaster.fire(times=1)
    # time.sleep(0.5)
    # ep_gimbal.move(pitch=0, yaw=31,yaw_speed = 250).wait_for_completed()
    # ep_blaster.fire(times=1)
    # time.sleep(0.5)
    # 控制云台回中
    ep_gimbal.recenter().wait_for_completed()

    # 控制云台向右旋转100度
    #ep_gimbal.move(pitch=0, yaw=180).wait_for_completed()

    # 控制云台航向轴速度100度每秒，俯仰轴速度100度每秒回中
    #ep_gimbal.recenter(pitch_speed=100, yaw_speed=300).wait_for_completed()

    ep_robot.close()
