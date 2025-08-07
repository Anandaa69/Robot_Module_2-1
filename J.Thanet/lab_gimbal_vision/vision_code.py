import robomaster
from robomaster import robot
from robomaster import vision
from robomaster import camera
import time
import cv2

def main():
    # เชื่อมต่อกับ RoboMaster S1
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    # เข้าถึง gimbal และ camera
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    
    try:
        print("เริ่มต้นการควบคุม Gimbal...")
        
        # Step 1: รีเซ็ตตำแหน่ง gimbal ให้อยู่กึ่งกลาง
        print("รีเซ็ตตำแหน่ง Gimbal...")
        ep_gimbal.recenter().wait_for_completed()
        time.sleep(2)
        
        # Step 2: หมุน gimbal ไป -90 องศา (หันซ้าย)
        print("หมุน Gimbal ไป -90 องศา...")
        ep_gimbal.moveto(pitch=0, yaw=-90, pitch_speed=0, yaw_speed=50).wait_for_completed()
        time.sleep(2)
        
        # Step 3: เปิดกล้องและเริ่ม vision detection
        print("เปิดกล้องและเริ่มตรวจจับ marker...")
        ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
        
        # เปิดใช้งานการตรวจจับ marker
        ep_vision.sub_detect_info(name="marker", callback=marker_detection_callback)
        
        # Step 4: ค่อยๆ หมุนจาก -90 องศา ไป 90 องศา
        print("ค่อยๆ หมุนจาก -90 องศา ไป 90 องศา พร้อมตรวจจับ marker...")
        
        # หมุนทีละ 10 องศา เพื่อให้มีเวลาตรวจจับ marker
        current_yaw = -90
        target_yaw = 90
        step = 30  # เพิ่มทีละ 10 องศา
        
        while current_yaw < target_yaw:
            current_yaw += step
            if current_yaw > target_yaw:
                current_yaw = target_yaw
                
            print(f"หมุนไป {current_yaw} องศา...")
            ep_gimbal.moveto(pitch=0, yaw=current_yaw, pitch_speed=0, yaw_speed=30).wait_for_completed()
            
            # รอสักครู่เพื่อให้มีเวลาตรวจจับ
            time.sleep(2)
        
        print("เสร็จสิ้นการหมุน Gimbal!")
        
        # รอสักครู่ก่อนปิด
        time.sleep(5)
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
    
    finally:
        # ปิดการทำงาน
        print("ปิดกล้องและยกเลิกการตรวจจับ...")
        ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        
        # รีเซ็ต gimbal กลับไปตำแหน่งเริ่มต้น
        print("รีเซ็ต Gimbal กลับตำแหน่งเดิม...")
        ep_gimbal.recenter().wait_for_completed()
        
        # ปิดการเชื่อมต่อ
        ep_robot.close()
        print("ปิดการเชื่อมต่อเรียบร้อย")

def marker_detection_callback(sub_info):
    """
    Callback function สำหรับการตรวจจับ marker
    """
    distance, angle, info = sub_info
    
    # ตรวจสอบว่าเจอ marker รูปหัวใจหรือไม่
    for marker in info:
        marker_id = marker[0]  # ID ของ marker
        x = marker[1]         # ตำแหน่ง x
        y = marker[2]         # ตำแหน่ง y
        w = marker[3]         # ความกว้าง
        h = marker[4]         # ความสูง
        
        # RoboMaster S1 ใช้ marker ID สำหรับรูปร่างต่างๆ
        # Heart marker มี ID = 2
        if marker_id == 2:  # Heart marker
            print(f"🧡 ตรวจพบ Heart Marker!")
            print(f"   ตำแหน่ง: ({x}, {y})")
            print(f"   ขนาด: {w} x {h}")
            print(f"   ระยะทาง: {distance:.2f}")
            print(f"   มุม: {angle:.2f}")
            print("-" * 40)

def advanced_heart_detection():
    """
    ฟังก์ชันสำหรับการตรวจจับ marker แบบละเอียด
    """
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_gimbal = ep_robot.gimbal
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    
    try:
        # รีเซ็ต gimbal
        ep_gimbal.recenter().wait_for_completed()
        time.sleep(1)
        
        # เปิดกล้อง
        ep_camera.start_video_stream(display=True)
        
        # ตั้งค่า vision detection สำหรับ marker
        ep_vision.sub_detect_info(name="marker")
        
        print("กำลังค้นหา Heart Marker...")
        
        # หมุน gimbal และค้นหา
        for angle in range(0, 360, 15):  # หมุนทีละ 15 องศา
            print(f"ตรวจสอบที่มุม {angle} องศา...")
            ep_gimbal.moveto(pitch=0, yaw=angle, pitch_speed=50, yaw_speed=50).wait_for_completed()
            time.sleep(1)
            
            # ตรวจสอบ marker ที่พบ
            marker_info = ep_vision.get_marker_detection_info()
            if marker_info:
                for marker in marker_info:
                    if marker[0] == 2:  # Heart marker
                        print(f"🧡 พบ Heart Marker ที่มุม {angle} องศา!")
                        return True
        
        print("ไม่พบ Heart Marker")
        return False
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        return False
    
    finally:
        ep_vision.unsub_detect_info(name="marker")
        ep_camera.stop_video_stream()
        ep_gimbal.recenter().wait_for_completed()
        ep_robot.close()

if __name__ == "__main__":
    # เรียกใช้ฟังก์ชันหลัก
    main()
    
    # หรือใช้ฟังก์ชันการค้นหาแบบละเอียด
    advanced_heart_detection()