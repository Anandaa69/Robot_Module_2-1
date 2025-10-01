from robomaster import robot
def convert_adc_to_cm(adc_value):
    if adc_value <= 0:
        return float('inf')
    A = 30263
    B = -1.352
    distance_cm = A * (adc_value ** B)
    return distance_cm

def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap") # เชื่อมต่อผ่าน WiFi
# เข้าถึง sensor adaptor
    sensor_adaptor = ep_robot.sensor_adaptor
# อ่านค่าจากช่องสัญญาณ (เช่น ch1)
    while True:
        left_sharp = sensor_adaptor.get_adc(id=1, port=1) 
        left_distance_cm = convert_adc_to_cm(left_sharp)

        right_sharp = sensor_adaptor.get_adc(id=2, port=1) 
        right_distance_cm = convert_adc_to_cm(right_sharp)

        right_IR = sensor_adaptor.get_io(id=2, port=2) # id=1 หมายถึงช่อง 1 ของ adaptor
        left_IR = sensor_adaptor.get_io(id=1, port=2)
        print(f"IR Sensor Value: Left = {left_IR} Right = {right_IR} | Sharp Distance: Left = {left_distance_cm:.2f} cm, Right = {right_distance_cm:.2f} cm")
if __name__ == '__main__':
    main()

# --- START OF FILE testcode1.py ---

# import robomaster
# from robomaster import robot
# import time

# def main():
#     # เริ่มการเชื่อมต่อกับหุ่นยนต์
#     ep_robot = robot.Robot()
#     ep_robot.initialize(conn_type="ap")

#     # เข้าถึงโมดูลแชสซี (ระบบขับเคลื่อน)
#     ep_chassis = ep_robot.chassis
    
#     # เข้าถึงโมดูล Sensor Adaptor
#     sensor_adaptor = ep_robot.sensor_adaptor

#     # --- [ปรับปรุง] 1. ลดความเร็วในการสไลด์ลงเพื่อให้ควบคุมการหยุดได้ดีขึ้น ---
#     slide_speed = 0.15  # ลองปรับค่านี้ได้ตามความเหมาะสม (เช่น 0.15, 0.2, 0.25)

#     print("Starting program: Slide right until sensor is 0.")
#     print(f"Slide speed set to: {slide_speed}")

#     try:
#         # เริ่มลูปการทำงาน
#         while True:
#             # อ่านค่าจากช่องสัญญาณที่ 1 ของ Sensor Adaptor
#             sensor_value = sensor_adaptor.get_io(id=1) 
            
#             # ตรวจสอบค่าจากเซ็นเซอร์
#             if sensor_value == 0:
#                 # --- [ปรับปรุง] 2. ใช้คำสั่งเบรกเพื่อ "หยุดทันที" ---
#                 ep_chassis.wheel_brake(robomaster.chassis.BRAKE_ALL)
#                 print("\nSensor is 0, stopping immediately!")
#                 break # ออกจากลูป
#             else:
#                 # ถ้าค่าไม่เป็น 0 ให้สไลด์ไปทางขวาด้วยความเร็วที่ลดลง
#                 ep_chassis.drive_speed(x=0, y=slide_speed, z=0, timeout=1)
            
#             # --- [ปรับปรุง] 3. ลดการหน่วงเวลาเพื่อให้ตอบสนองเร็วขึ้น ---
#             # time.sleep(0.02) # ลดจาก 0.1 เหลือ 0.02 วินาที

#     except KeyboardInterrupt:
#         # กรณีที่ต้องการหยุดโปรแกรมด้วยตนเอง (Ctrl+C)
#         print("\nProgram stopped by user.")
#         ep_chassis.wheel_brake(robomaster.chassis.BRAKE_ALL)

#     finally:
#         # ปิดการเชื่อมต่อกับหุ่นยนต์เสมอเมื่อจบโปรแกรม
#         print("Closing connection.")
#         ep_robot.close()

# if __name__ == '__main__':
#     main()