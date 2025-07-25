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

def sub_data_handler(sub_info):
    distance = sub_info
    print("tof:{0}".format(distance[0]))

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor

    ep_sensor.sub_distance(freq=5, callback=sub_data_handler)

    ep_gimbal.moveto(pitch=0, yaw=90).wait_for_completed()
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
    ep_gimbal.moveto(pitch=0, yaw=-90).wait_for_completed()
    ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()

    ep_sensor.unsub_distance()
    ep_robot.close()