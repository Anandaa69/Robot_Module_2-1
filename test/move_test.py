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

import cv2
import time
import robomaster
from robomaster import robot
from robomaster import blaster
from robomaster import camera
from robomaster import vision


markers = []
window_size = 720
height_window = window_size
weight_window = 1280


class MarkerInfo:

    def __init__(self, x, y, w, h, info):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._info = info

    @property
    def pt1(self):
        return int((self._x - self._w / 2) * weight_window), int((self._y - self._h / 2) * height_window)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * weight_window), int((self._y + self._h / 2) * height_window)

    @property
    def center(self):
        return int(self._x * weight_window), int(self._y * height_window)

    @property
    def text(self):
        return self._info



def on_detect_marker(marker_info):
    number = len(marker_info)
    markers.clear()
    for i in range(0, number):
        x, y, w, h, info = marker_info[i]
        markers.append(MarkerInfo(x, y, w, h, info))
        print("marker:{0} x:{1}, y:{2}, w:{3}, h:{4}".format(info, x * height_window, y * weight_window, w * height_window, h * weight_window))





if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_blaster = ep_robot.blaster
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    ep_gimbal.moveto(pitch=0, yaw=0 , yaw_speed = 250, pitch_speed = 250).wait_for_completed()
    ep_gimbal.move(pitch=0, yaw=31,yaw_speed = 250).wait_for_completed()
    
    
    ep_camera.start_video_stream(display=True, resolution=camera.STREAM_720P)
    result = ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)


    ep_gimbal.move(pitch=0, yaw=31,yaw_speed = 250).wait_for_completed()
    ep_blaster.fire(fire_type=blaster.INFRARED_FIRE, times=3)
    time.sleep(2)
    
    ep_gimbal.recenter().wait_for_completed()
    ep_robot.close()
