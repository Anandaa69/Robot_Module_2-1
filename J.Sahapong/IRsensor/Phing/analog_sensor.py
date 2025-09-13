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
    
    # กำหนด ID และ Port ของเซ็นเซอร์
    IR_SENSOR_ID = 3
    IR_SENSOR_PORT = 1

    print("=============================================")
    print("=== IR Sensor Calibration Program ===")
    print(f"Reading from Sensor Adaptor ID: {IR_SENSOR_ID}, Port: {IR_SENSOR_PORT}")
    print("Formula: distance = 30263 * (ADC ^ -1.352)")
    print("=============================================")
    print("\nHow to use:")
    print("1. Place an object at a KNOWN distance (e.g., 10cm, 15cm, 20cm).")
    print("2. Compare the 'Calculated Distance' with the real distance.")
    print("3. Note down the differences to fine-tune the formula if needed.")
    print("\nPress Ctrl+C to stop.")

    try:
        # --- 3. เริ่มลูปการทำงานเพื่ออ่านค่าและแปลงผล ---
        while True:
            # อ่านค่า Analog (ADC) จาก Sensor Adaptor
            adc_value = sensor_adaptor.get_adc(id=IR_SENSOR_ID, port=IR_SENSOR_PORT) 
            
            # แปลงค่า ADC เป็นระยะทาง (cm)
            distance_cm = convert_adc_to_cm(adc_value)
            
            # พิมพ์ค่าที่อ่านได้และค่าที่แปลงแล้วออกมา
            # ใช้ \r เพื่อให้อัปเดตค่าในบรรทัดเดียว
            print(f"Raw ADC Value: {adc_value:4d}  |  Calculated Distance: {distance_cm:6.2f} cm      ", end='\r')
            
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