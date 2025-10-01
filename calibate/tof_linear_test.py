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


import robomaster
from robomaster import robot
import time

# --- กำหนดค่า Slope (m) และ Y-intercept (c) ที่ได้จากการสอบเทียบ ---
# สมการของคุณคือ: ระยะทางจริง = (0.0894 * ค่าเซ็นเซอร์) + (3.8409)
# ค่าเซ็นเซอร์ (raw_tof) ที่ได้รับจาก RoboMaster EP ToF มักจะอยู่ในหน่วยมิลลิเมตร (mm)
# ดังนั้น สมมติว่า Slope และ Y-intercept นี้ คำนวณเพื่อให้ได้ผลลัพธ์เป็นหน่วยเซนติเมตร (cm)
# ถ้าคุณสอบเทียบด้วยหน่วย mm และต้องการ mm ผลลัพธ์ก็จะอยู่ในหน่วย mm
# ตรวจสอบหน่วยที่คุณใช้ในการเก็บข้อมูลและสมการให้ดี
CALIBRATION_SLOPE = 0.0894
CALIBRATION_Y_INTERCEPT = 3.8409


# ระยะทางจริง = (0.0960 * ค่าเซ็นเซอร์) + (1.2260)
# ดังนั้น ค่า Slope (m): 0.0960
# และ ค่า Y-intercept (c): 1.2260

# ฟังก์ชันสำหรับแปลงค่าดิบจาก ToF ให้เป็นระยะทางที่สอบเทียบแล้ว
def calibrate_tof_value(raw_tof_value):
    """
    แปลงค่า ToF ดิบให้เป็นระยะทางที่สอบเทียบแล้วโดยใช้สมการ Linear Regression.
    Args:
        raw_tof_value (float): ค่าระยะทางดิบที่อ่านได้จากเซ็นเซอร์ ToF (มักจะอยู่ในหน่วย mm).
    Returns:
        float: ค่าระยะทางที่สอบเทียบแล้ว (หน่วยตามที่คุณสอบเทียบ, เช่น cm).
    """
    # ตัวเซ็นเซอร์ ToF ของ RoboMaster มักจะให้ค่าเป็นมิลลิเมตร (mm)
    # ถ้าค่า Slope และ Y-intercept ที่คุณได้มาจากการป้อนข้อมูล 'ค่าเซ็นเซอร์' ที่เป็น mm
    # และ 'ระยะทางจริง' ที่เป็น cm
    # ผลลัพธ์ที่ได้จากฟังก์ชันนี้ก็จะเป็น cm
    
    calibrated_value = (CALIBRATION_SLOPE * raw_tof_value) + CALIBRATION_Y_INTERCEPT
    return calibrated_value

def sub_data_handler(sub_info):
    # sub_info เป็น list ที่มีค่า ToF จากเซ็นเซอร์ทั้ง 4 ตัว
    # distance[0] คือ tof1, distance[1] คือ tof2, และอื่นๆ
    # ค่าเหล่านี้เป็นค่าดิบจากเซ็นเซอร์ (มักจะเป็นมิลลิเมตร)
    raw_tof1 = sub_info[0]
    raw_tof2 = sub_info[1]
    raw_tof3 = sub_info[2]
    raw_tof4 = sub_info[3]

    # --- นำค่าดิบแต่ละตัวมาทำการสอบเทียบ ---
    calibrated_tof1 = calibrate_tof_value(raw_tof1)
    calibrated_tof2 = calibrate_tof_value(raw_tof2)
    calibrated_tof3 = calibrate_tof_value(raw_tof3)
    calibrated_tof4 = calibrate_tof_value(raw_tof4)
    # ------------------------------------

    print(f"Raw TOF: {raw_tof1:.2f}, {raw_tof2:.2f}, {raw_tof3:.2f}, {raw_tof4:.2f}")
    print(f"Calibrated TOF: {calibrated_tof1:.2f}cm, {calibrated_tof2:.2f}cm, {calibrated_tof3:.2f}cm, {calibrated_tof4:.2f}cm")
    # คุณสามารถเปลี่ยน 'cm' เป็นหน่วยที่คุณสอบเทียบได้ เช่น 'mm' หากคุณวัดระยะจริงเป็น mm


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap") # หรือ "sta" ตามที่คุณเชื่อมต่อ

    ep_sensor = ep_robot.sensor
    ep_gimbal = ep_robot.gimbal
    
    print("✅ Recalibrating gimbal...")
    ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
    print("✅ Gimbal recalibrated.")

    try:
        print("✅ Subscribing to ToF distance data...")
        # freq=20 หมายถึงรับข้อมูล 20 ครั้งต่อวินาที
        ep_sensor.sub_distance(freq=10, callback=sub_data_handler)
        
        print("⏱️ Collecting data for 10 seconds...")
        time.sleep(1000) # รอ 10 วินาทีเพื่อเก็บข้อมูล

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    finally:
        print("⏹ Unsubscribing from ToF data and closing robot connection...")
        ep_sensor.unsub_distance() # ยกเลิกการรับข้อมูล
        ep_robot.close() # ปิดการเชื่อมต่อหุ่นยนต์
        print("✅ Robot connection closed.")