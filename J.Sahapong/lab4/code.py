
import time
import robomaster
from robomaster import robot
import csv
from datetime import datetime
import threading

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_sensor = ep_robot.sensor

    # สร้างชื่อไฟล์ CSV พร้อม timestamp
    filename = datetime.now().strftime("test2.csv")

    # ตัวแปรสำหรับเก็บข้อมูล ToF ล่าสุดและสถานะการทำงาน
    tof_data_buffer = [] # ใช้ buffer เพื่อเก็บข้อมูล ToF ก่อนเขียนลงไฟล์
    recording_active = threading.Event() # ใช้ Event เพื่อควบคุมการบันทึกข้อมูล
    recording_active.set() # เริ่มต้นให้สามารถบันทึกได้

    # ฟังก์ชันสำหรับ subscribe ข้อมูล ToF
    def tof_data_handler(sub_info):
        # ตรวจสอบว่ายังอยู่ในช่วงเวลาบันทึกหรือไม่
        if recording_active.is_set():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            tof_values = list(sub_info) # แปลง tuple เป็น list
            tof_data_buffer.append([timestamp] + tof_values)
            # แสดงผลทางคอนโซลเพื่อการตรวจสอบ
            print(f"Recorded ToF: {timestamp}, tof1={tof_values[0]}")

    print("กำลังเตรียมการเชื่อมต่อและเซ็นเซอร์...")

    try:
        # เริ่มการ subscribe ข้อมูล ToF ด้วยความถี่ 20 Hz
        ep_sensor.sub_distance(freq=40, callback=tof_data_handler)
        print("เริ่มการบันทึกข้อมูล ToF แล้ว...")
        print("จะบันทึกข้อมูลเป็นเวลา 30 วินาที...")

        # รอ 30 วินาทีสำหรับการบันทึกข้อมูล
        time.sleep(30)

        # หยุดการบันทึกข้อมูล
        recording_active.clear()
        print("ครบ 30 วินาทีแล้ว หยุดการบันทึกข้อมูล ToF")

        # ยกเลิกการ subscribe ข้อมูล
        ep_sensor.unsub_distance()
        print("ยกเลิกการ subscribe ToF sensor แล้ว")

        # เขียนข้อมูลที่เก็บใน buffer ลงไฟล์ CSV
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # เขียนหัวข้อคอลัมน์ในไฟล์ CSV
            csv_writer.writerow(['timestamp', 'tof1', 'tof2', 'tof3', 'tof4'])
            csv_writer.writerows(tof_data_buffer)
        print(f"บันทึกข้อมูลลงในไฟล์ '{filename}' เรียบร้อยแล้ว")

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
    finally:
        # ปิดการเชื่อมต่อหุ่นยนต์เสมอ ไม่ว่าจะเกิดข้อผิดพลาดหรือไม่
        ep_robot.close()
        print("ปิดการเชื่อมต่อ Robomaster แล้ว")