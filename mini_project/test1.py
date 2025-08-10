# โค้ดสำหรับหุ่นยนต์ Robomaster เพื่อหาทางออกจากเขาวงกต
# โดยใช้หลักการ "เดินชิดกำแพงขวา" (Right-Hand Rule)

from robomaster import robot
import time
import random

# =========================================================================
# คลาสจำลองเซ็นเซอร์วัดระยะทาง
# ในสถานการณ์จริง คุณจะต้องเชื่อมต่อกับเซ็นเซอร์จริง (เช่น Lidar หรือ IR sensor)
# =========================================================================
class MockSensors:
    def __init__(self):
        # ค่าจำลองระยะห่างจากกำแพง (หน่วย: เมตร)
        # กำหนดค่าเริ่มต้นเพื่อเริ่มการทำงาน
        self.distance_front = 1.0
        self.distance_right = 0.3
        self.distance_left = 1.0
        self.maze_solved = False

    def update_sensors(self, current_pos):
        """
        จำลองการอัปเดตค่าจากเซ็นเซอร์
        ในโค้ดจริงจะอ่านค่าจากฮาร์ดแวร์จริง ๆ
        """
        # อัปเดตสถานะของเขาวงกตเมื่อหุ่นยนต์ถึงเป้าหมาย
        if current_pos > 5.0:  # สมมติว่าระยะทาง 5 เมตรคือทางออก
            self.maze_solved = True
            print("🎉 หุ่นยนต์เจอทางออกแล้ว!")
            return

        # จำลองการเปลี่ยนแปลงระยะห่างเมื่อหุ่นยนต์เคลื่อนที่
        # ค่าเหล่านี้จะเปลี่ยนไปตามตรรกะการเคลื่อนที่ของหุ่นยนต์
        # และตำแหน่งของกำแพงในเขาวงกต
        
        # ตัวอย่างการจำลองง่ายๆ:
        # ถ้าหุ่นยนต์เดินหน้าไป จะมีโอกาสที่กำแพงด้านหน้าจะอยู่ใกล้ขึ้น
        self.distance_front = random.uniform(0.1, 2.0)
        # ถ้าเดินชิดกำแพงขวา ระยะด้านขวาจะคงที่
        self.distance_right = random.uniform(0.2, 0.4)
        self.distance_left = random.uniform(0.5, 2.0)


# =========================================================================
# ฟังก์ชันหลักสำหรับการนำทางในเขาวงกต
# =========================================================================
def navigate_maze(ep_chassis):
    """
    ควบคุมหุ่นยนต์ให้เคลื่อนที่ในเขาวงกตด้วยหลักการเดินชิดกำแพง
    """
    sensors = MockSensors()
    maze_solved = False
    current_distance = 0.0
    
    # ตั้งค่าความเร็วคงที่สำหรับการเคลื่อนที่ไปข้างหน้า
    forward_speed = 0.5  # m/s
    rotate_speed = 30    # deg/s

    print("=== เริ่มต้นการนำทางในเขาวงกต ===")
    print("หุ่นยนต์จะใช้หลักการเดินชิดกำแพงขวา")
    
    try:
        while not sensors.maze_solved:
            # อัปเดตค่าเซ็นเซอร์จำลอง
            sensors.update_sensors(current_distance)
            
            # เงื่อนไขการตัดสินใจตามหลักการเดินชิดกำแพงขวา
            
            # 1. ถ้าทางขวามีทางเปิด (ระยะห่าง > 0.4 เมตร) ให้เลี้ยวขวา
            if sensors.distance_right > 0.4:
                print("เจอทางเปิดด้านขวา -> เลี้ยวขวา")
                ep_chassis.drive_speed(x=0, y=0, z=-rotate_speed, timeout=1.5) # หมุนตามเข็มนาฬิกา
                time.sleep(1.5)
            
            # 2. ถ้าข้างหน้ามีทางเปิด (ระยะห่าง > 0.5 เมตร) ให้เดินตรงไป
            elif sensors.distance_front > 0.5:
                print("ข้างหน้ามีทางว่าง -> เดินตรงไป")
                ep_chassis.drive_speed(x=forward_speed, y=0, z=0, timeout=1.0)
                time.sleep(1.0)
                current_distance += forward_speed * 1.0
            
            # 3. ถ้าทางขวาและข้างหน้าเป็นทางตัน ให้เลี้ยวซ้าย
            elif sensors.distance_left > 0.4:
                print("ทางตัน -> เลี้ยวซ้าย")
                ep_chassis.drive_speed(x=0, y=0, z=rotate_speed, timeout=1.5) # หมุนทวนเข็มนาฬิกา
                time.sleep(1.5)
            
            # 4. ถ้าทุกทิศทางตัน ให้หมุนกลับ
            else:
                print("ทางตันทุกทิศทาง -> หมุนกลับหลัง")
                ep_chassis.drive_speed(x=0, y=0, z=rotate_speed * 2, timeout=3.0) # หมุน 180 องศา
                time.sleep(3.0)
                
            # หยุดหุ่นยนต์ชั่วครู่
            ep_chassis.drive_speed(x=0, y=0, z=0)
            time.sleep(0.5)

        print("=== การนำทางเสร็จสิ้น ===")

    except KeyboardInterrupt:
        print("โปรแกรมถูกขัดจังหวะโดยผู้ใช้")
        ep_chassis.drive_speed(x=0, y=0, z=0)
    
# =========================================================================
# บล็อก Main ของโปรแกรม
# =========================================================================
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis
    
    try:
        # เริ่มต้นการนำทาง
        navigate_maze(ep_chassis)
        
    finally:
        # ปิดการเชื่อมต่อเมื่อจบการทำงาน
        ep_robot.close()
        print("ปิดการเชื่อมต่อกับหุ่นยนต์เรียบร้อย")