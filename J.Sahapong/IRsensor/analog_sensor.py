# --- START OF FILE testcode_read_only.py ---

import robomaster
from robomaster import robot
import time

def main():
    # เริ่มการเชื่อมต่อกับหุ่นยนต์
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # เข้าถึงโมดูล Sensor Adaptor
    sensor_adaptor = ep_robot.sensor_adaptor

    print("Starting program: Continuously reading Sharp sensor value.")
    print("Press Ctrl+C to stop.")

    try:
        # เริ่มลูปการทำงานแบบไม่รู้จบเพื่ออ่านค่า
        while True:
            # อ่านค่า Analog (ADC) จาก Sensor Adaptor ID 1, Port 1
            # ค่าที่ได้จะอยู่ในช่วง 0-1023
            sensor_value = sensor_adaptor.get_adc(id=1, port=1) 
            
            # พิมพ์ค่าที่อ่านได้ออกมาทาง Terminal
            # \r (carriage return) จะทำให้เคอร์เซอร์กลับไปที่ต้นบรรทัดเดิม ทำให้ค่าที่แสดงผลอัปเดตในบรรทัดเดียว
            print(f"Sharp Sensor ADC Value: {sensor_value}    ", end='\r')
            
            # หน่วงเวลาเล็กน้อยเพื่อไม่ให้ส่งคำสั่งถี่เกินไป
            time.sleep(0.1)

    except KeyboardInterrupt:
        # ทำงานเมื่อผู้ใช้กด Ctrl+C เพื่อออกจากโปรแกรม
        print("\nProgram stopped by user.")

    finally:
        # ปิดการเชื่อมต่อกับหุ่นยนต์เสมอเมื่อจบโปรแกรม
        print("Closing connection.")
        ep_robot.close()

if __name__ == '__main__':
    main()