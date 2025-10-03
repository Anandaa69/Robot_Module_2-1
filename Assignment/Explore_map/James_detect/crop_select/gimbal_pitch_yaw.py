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
import cv2
import robomaster
from robomaster import robot

# ฟังก์ชันสำหรับจัดการข้อมูลมุมที่ได้รับมา
def sub_data_handler(angle_info):
    pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = angle_info
    print("Gimbal Angle: pitch_angle:{0}, yaw_angle:{1}".format(pitch_angle, yaw_angle))

if __name__ == '__main__':
    # เริ่มต้นการเชื่อมต่อหุ่นยนต์
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # เข้าถึงโมดูล gimbal และ camera
    ep_gimbal = ep_robot.gimbal
    ep_camera = ep_robot.camera

    # เริ่มการสตรีมวิดีโอ แต่ไม่แสดงผลโดยตรง
    ep_camera.start_video_stream(display=False)

    # สมัครรับข้อมูลมุมของ gimbal ที่ความถี่ 20Hz
    ep_gimbal.sub_angle(freq=20, callback=sub_data_handler)

    print("คุณสามารถใช้มือปรับมุมของ Gimbal ได้เลย โปรแกรมจะแสดงค่ามุมและภาพจากกล้องเป็นเวลา 30 วินาที")

    # วนลูปเพื่อแสดงภาพจากกล้องเป็นเวลาประมาณ 30 วินาที (20 เฟรมต่อวินาที * 30 วินาที = 600 เฟรม)
    for i in range(0, 10000000):
        try:
            # อ่านภาพจากกล้อง
            img = ep_camera.read_cv2_image()
            # แสดงภาพในหน้าต่างชื่อ "Camera Feed"
            cv2.imshow("Camera Feed", img)
            # รอการกดปุ่ม (จำเป็นสำหรับการแสดงผลของ cv2.imshow)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Error reading frame: {e}")
            break

    # ยกเลิกการสมัครรับข้อมูลมุม
    ep_gimbal.unsub_angle()

    # หยุดการสตรีมวิดีโอ
    ep_camera.stop_video_stream()

    # ปิดหน้าต่างทั้งหมดของ OpenCV
    cv2.destroyAllWindows()

    # ปิดการเชื่อมต่อหุ่นยนต์
    ep_robot.close()

    print("จบการทำงาน")