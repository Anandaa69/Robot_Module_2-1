# -*-coding:utf-8-*-

import robomaster
from robomaster import robot
import time

# =======================================================================
# --- ฟังก์ชันสำหรับแปลงค่า ADC เป็นระยะทาง (cm) ตามสูตรของคุณ ---
# y = 30263 * x^(-1.352)
# โดยที่ y คือ ระยะทาง (cm) และ x คือค่า ADC
# =======================================================================
def convert_adc_to_cm(adc_value):
    """
    แปลงค่า ADC ที่อ่านได้จากเซ็นเซอร์เป็นระยะทางในหน่วยเซนติเมตร
    โดยใช้สูตรที่ได้มาจากการทำ Regression
    """
    # ป้องกันการคำนวณผิดพลาดถ้าค่า ADC เป็น 0 หรือน้อยกว่า
    if adc_value <= 0:
        return float('inf') # คืนค่าเป็นอนันต์ (ไกลมาก)

    # --- ใช้ค่าคงที่จากสูตรของคุณ ---
    A = 30263
    B = -1.352
    
    distance_cm = A * (adc_value ** B)
    return distance_cm

def main():
    # --- 1. เริ่มการเชื่อมต่อกับหุ่นยนต์ ---
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # --- 2. เข้าถึงโมดูล Sensor Adaptor ---
    sensor_adaptor = ep_robot.sensor_adaptor
    
    # --- กำหนด ID และ Port ของเซ็นเซอร์ทั้ง 2 ตัว ---
    IR_SENSOR_1_ID = 3
    IR_SENSOR_1_PORT = 1
    
    IR_SENSOR_2_ID = 1
    IR_SENSOR_2_PORT = 1

    print("=============================================")
    print("=== IR Sensor Comparison Program (2 Sensors) ===")
    print(f"Sensor 1: Adaptor ID: {IR_SENSOR_1_ID}, Port: {IR_SENSOR_1_PORT}")
    print(f"Sensor 2: Adaptor ID: {IR_SENSOR_2_ID}, Port: {IR_SENSOR_2_PORT}")
    print("Formula: distance = 30263 * (ADC ^ -1.352)")
    print("=============================================")
    print("\nPress Ctrl+C to stop.")
    time.sleep(2)

    try:
        # --- 3. เริ่มลูปการทำงานเพื่ออ่านค่าและแปลงผล ---
        while True:
            # --- อ่านค่าจากเซ็นเซอร์ตัวที่ 1 ---
            adc1_value = sensor_adaptor.get_adc(id=IR_SENSOR_1_ID, port=IR_SENSOR_1_PORT) 
            distance1_cm = convert_adc_to_cm(adc1_value)
            
            # --- อ่านค่าจากเซ็นเซอร์ตัวที่ 2 ---
            adc2_value = sensor_adaptor.get_adc(id=IR_SENSOR_2_ID, port=IR_SENSOR_2_PORT) 
            distance2_cm = convert_adc_to_cm(adc2_value)
            
            # --- พิมพ์ค่าที่อ่านได้และค่าที่แปลงแล้วของทั้ง 2 ตัวออกมา ---
            # ใช้ \r เพื่อให้อัปเดตค่าในบรรทัดเดียว
            print(f"Sensor 1 (ID {IR_SENSOR_1_ID}): ADC {adc1_value:4d}, Dist: {distance1_cm:6.2f} cm   |   Sensor 2 (ID {IR_SENSOR_2_ID}): ADC {adc2_value:4d}, Dist: {distance2_cm:6.2f} cm      ", end='\r')
            
            # หน่วงเวลาเล็กน้อย
            time.sleep(0.1)

    except KeyboardInterrupt:
        # ทำงานเมื่อผู้ใช้กด Ctrl+C
        print("\n\nProgram stopped by user.")

    finally:
        # ปิดการเชื่อมต่อกับหุ่นยนต์
        print("Closing robot connection.")
        ep_robot.close()
        print("Connection closed.")

if __name__ == '__main__':
    main()