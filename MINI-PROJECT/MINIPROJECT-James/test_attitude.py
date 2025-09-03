import robomaster
from robomaster import robot
import time
import threading

class AttitudeHandler:
    def __init__(self):
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.target_yaw = 0.0
        self.yaw_tolerance = 3.0  # เพิ่ม tolerance เพราะมีการ drift
        self.is_monitoring = False
        
    def attitude_handler(self, attitude_info):
        if not self.is_monitoring:
            return
            
        yaw, pitch, roll = attitude_info
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.current_roll = roll
        print(f"\r🧭 Current chassis attitude: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°", end="", flush=True)
        
    def start_monitoring(self, chassis):
        """เริ่มการติดตาม attitude"""
        self.is_monitoring = True
        chassis.sub_attitude(freq=20, callback=self.attitude_handler)
        
    def stop_monitoring(self, chassis):
        """หยุดการติดตาม attitude"""
        self.is_monitoring = False
        try:
            chassis.unsub_attitude()
        except:
            pass
            
    def normalize_angle(self, angle):
        """ปรับมุมให้อยู่ในช่วง -180 ถึง 180"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
        
    def is_at_target_yaw(self, target_yaw=0.0):
        """ตรวจสอบว่า yaw อยู่ที่เป้าหมายหรือไม่"""
        # สำหรับ 180° ตรวจสอบทั้ง 180° และ -180°
        if abs(target_yaw) == 180:
            diff_180 = abs(self.normalize_angle(self.current_yaw - 180))
            diff_neg180 = abs(self.normalize_angle(self.current_yaw - (-180)))
            diff = min(diff_180, diff_neg180)
            target_display = f"±180"
        else:
            diff = abs(self.normalize_angle(self.current_yaw - target_yaw))
            target_display = f"{target_yaw}"
            
        is_correct = diff <= self.yaw_tolerance
        print(f"\n🎯 Yaw check: current={self.current_yaw:.1f}°, target={target_display}°, diff={diff:.1f}°, correct={is_correct}")
        return is_correct
        
    def correct_yaw_to_target(self, chassis, target_yaw=0.0):
        """แก้ไข yaw ให้อยู่ที่มุมเป้าหมายที่กำหนด"""
        
        if self.is_at_target_yaw(target_yaw):
            print(f"✅ Chassis already at correct yaw: {self.current_yaw:.1f}° (target: {target_yaw}°)")
            return True
            
        # FIXED: Gimbal coordinate คงที่ตาม physical orientation ของ gimbal เอง
        # ต้องหมุน robot ในทิศทางตรงข้ามกับที่ต้องการปรับ gimbal angle
        gimbal_to_target = target_yaw - self.current_yaw
        gimbal_diff = self.normalize_angle(gimbal_to_target)
        
        # หมุน robot ในทิศทางตรงข้าม
        robot_rotation = -gimbal_diff
        
        print(f"🔧 Correcting chassis yaw: from {self.current_yaw:.1f}° to {target_yaw}°")
        print(f"📐 Gimbal needs to change: {gimbal_diff:.1f}°")
        print(f"📐 Robot will rotate: {robot_rotation:.1f}°")
        
        # หมุน chassis
        try:
            if abs(robot_rotation) > self.yaw_tolerance:
                correction_speed = 30
                
                print(f"🔄 Rotating robot {robot_rotation:.1f}°")
                chassis.move(x=0, y=0, z=robot_rotation, z_speed=correction_speed).wait_for_completed()
                time.sleep(1.0)  # รอให้การหมุนเสร็จสมบูรณ์
            
            # ตรวจสอบผลลัพธ์
            final_check = self.is_at_target_yaw(target_yaw)
            
            if final_check:
                print(f"✅ Successfully corrected chassis yaw to {self.current_yaw:.1f}°")
                return True
            else:
                print(f"⚠️ Chassis yaw correction incomplete: {self.current_yaw:.1f}° (target: {target_yaw}°)")
                
                # คำนวณความต่างที่เหลือ
                remaining_gimbal = target_yaw - self.current_yaw
                remaining_diff = self.normalize_angle(remaining_gimbal)
                remaining_robot = -remaining_diff
                print(f"📐 Remaining gimbal difference: {remaining_diff:.1f}°")
                print(f"📐 Additional robot rotation needed: {remaining_robot:.1f}°")
                
                # ลองอีกครั้งด้วยการปรับเล็กน้อย
                if abs(remaining_robot) > self.yaw_tolerance and abs(remaining_robot) < 45:
                    print(f"🔧 Fine-tuning robot with additional {remaining_robot:.1f}°")
                    chassis.move(x=0, y=0, z=remaining_robot, z_speed=20).wait_for_completed()
                    time.sleep(0.5)
                    return self.is_at_target_yaw(target_yaw)
                else:
                    print(f"⚠️ Remaining rotation too large ({remaining_robot:.1f}°), may need multiple corrections")
                return False
                
        except Exception as e:
            print(f"❌ Failed to correct chassis yaw: {e}")
            return False

def get_user_input():
    """รับ input จากผู้ใช้"""
    try:
        return input().strip().lower()
    except KeyboardInterrupt:
        return 'q'
    except:
        return None

def input_handler(attitude_handler, chassis):
    """จัดการ input จากผู้ใช้"""
    print("\n" + "="*60)
    print("🎮 ROBOT ATTITUDE CONTROLLER")
    print("="*60)
    print("Commands:")
    print("  0   - Correct yaw to 0°")
    print("  90  - Correct yaw to 90°")
    print("  -90 - Correct yaw to -90°")
    print("  180 - Correct yaw to 180° (or -180°)")
    print("  q   - Quit program")
    print("  Any other number - Set custom target angle")
    print("="*60)
    
    while True:
        try:
            print("\nEnter target yaw angle (0, 90, -90, 180) or 'q' to quit: ", end="", flush=True)
            user_input = get_user_input()
            
            if user_input == 'q':
                print("\n👋 Exiting program...")
                return
            elif user_input is None:
                continue
            else:
                try:
                    # พยายามแปลงเป็นตัวเลข
                    target_angle = float(user_input)
                    
                    # จำกัดค่าให้อยู่ในช่วง -180 ถึง 180
                    target_angle = attitude_handler.normalize_angle(target_angle)
                    
                    print(f"\n🎯 Starting yaw correction to {target_angle}°...")
                    success = attitude_handler.correct_yaw_to_target(chassis, target_angle)
                    if success:
                        print(f"✅ Yaw correction to {target_angle}° completed!")
                    else:
                        print(f"❌ Yaw correction to {target_angle}° failed!")
                        
                except ValueError:
                    print(f"\n❌ Invalid input: '{user_input}'. Please enter a number or 'q' to quit.")
                    
        except KeyboardInterrupt:
            print("\n👋 Program interrupted by user")
            return
        except Exception as e:
            print(f"\n❌ Input handler error: {e}")
            time.sleep(0.5)

if __name__ == '__main__':
    # สร้าง robot instance
    ep_robot = robot.Robot()
    
    try:
        print("🔄 Initializing robot connection...")
        ep_robot.initialize(conn_type="ap")
        print("✅ Robot connected successfully!")
        
        ep_chassis = ep_robot.chassis
        
        # สร้าง attitude handler
        attitude_handler = AttitudeHandler()
        
        # เริ่มการติดตาม attitude
        print("📡 Starting attitude monitoring...")
        attitude_handler.start_monitoring(ep_chassis)
        
        # เริ่ม input handler ใน thread แยก
        input_thread = threading.Thread(target=input_handler, args=(attitude_handler, ep_chassis))
        input_thread.daemon = True
        input_thread.start()
        
        # รอจนกว่า input thread จะจบ
        input_thread.join()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # หยุดการติดตาม attitude และปิดการเชื่อมต่อ
        print("🔄 Cleaning up...")
        try:
            attitude_handler.stop_monitoring(ep_chassis)
        except:
            pass
        try:
            ep_robot.close()
            print("✅ Robot connection closed")
        except:
            pass