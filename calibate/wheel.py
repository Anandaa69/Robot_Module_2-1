from robomaster import robot
import time

# --- การตั้งค่าหุ่นยนต์ ---
class RobotCalibrator:
    def __init__(self):
        self.s1 = None
        self.chassis = None
        self.calibration_factor = 1.0
    
    def initialize_robot(self, conn_type="ap"):
        """เริ่มต้นการเชื่อมต่อหุ่นยนต์"""
        try:
            self.s1 = robot.Robot()
            self.s1.initialize(conn_type=conn_type)
            self.chassis = self.s1.chassis
            print("เชื่อมต่อหุ่นยนต์สำเร็จ")
            return True
        except Exception as e:
            print(f"ไม่สามารถเชื่อมต่อหุ่นยนต์ได้: {e}")
            return False
    
    def test_movement(self, distance_in_meters, speed=3):
        """
        ฟังก์ชันสำหรับทดสอบการเคลื่อนที่ไปข้างหน้าของหุ่นยนต์
        ใช้ API ของ DJI RoboMaster SDK เวอร์ชันปัจจุบัน

        Args:
            distance_in_meters (float): ระยะทางที่ต้องการให้หุ่นยนต์เคลื่อนที่ (หน่วยเป็นเมตร)
            speed (float): ความเร็วในการเคลื่อนที่ (m/s)
        
        Returns:
            bool: True ถ้าการเคลื่อนที่สำเร็จ, False ถ้าเกิดข้อผิดพลาด
        """
        if not self.chassis:
            print("ยังไม่ได้เชื่อมต่อหุ่นยนต์")
            return False
            
        print(f"กำลังทดสอบการเคลื่อนที่: เคลื่อนไปข้างหน้า {distance_in_meters:.2f} เมตร...")
        
        try:
            # ใช้ API ปัจจุบันของ DJI RoboMaster SDK
            # chassis.move() ใช้พารามิเตอร์ x, y, z สำหรับระยะทาง และ xy_speed สำหรับความเร็ว
            action = self.chassis.move(x=distance_in_meters, y=0, z=0, xy_speed=speed)
            action.wait_for_completed()
            print("การเคลื่อนที่เสร็จสิ้น")
            return True
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการเคลื่อนที่: {e}")
            print("กำลังลองใช้ API อื่น...")
            
            # ลองใช้วิธีอื่นถ้า move() ไม่ทำงาน
            try:
                # วิธีที่ 2: ใช้ speed control แล้วหยุดตามเวลา
                estimated_time = distance_in_meters / speed
                self.chassis.drive_speed(x=speed, y=0, z=0)
                time.sleep(estimated_time)
                self.chassis.drive_speed(x=0, y=0, z=0)  # หยุด
                print("การเคลื่อนที่เสร็จสิ้น (ใช้ speed control)")
                return True
            except Exception as e2:
                print(f"ไม่สามารถเคลื่อนที่ได้: {e2}")
                return False

    def calibrate_movement_factor(self, actual_distance, expected_distance):
        """
        คำนวณตัวคูณการปรับแก้ (calibration factor)
        เพื่อชดเชยความคลาดเคลื่อนในการเคลื่อนที่

        Args:
            actual_distance (float): ระยะทางจริงที่หุ่นยนต์เคลื่อนที่ได้ (หน่วยเป็นเมตร)
            expected_distance (float): ระยะทางที่ตั้งใจจะให้หุ่นยนต์เคลื่อนที่ (หน่วยเป็นเมตร)

        Returns:
            float: ตัวคูณการปรับแก้
        """
        if expected_distance == 0:
            print("เกิดข้อผิดพลาด: ระยะทางที่คาดหวังไม่สามารถเป็น 0 ได้")
            return 1.0
        
        if actual_distance <= 0:
            print("เกิดข้อผิดพลาด: ระยะทางจริงต้องมากกว่า 0")
            return 1.0
            
        factor = expected_distance / actual_distance
        self.calibration_factor = factor
        return factor

    def get_user_measurement(self, expected_distance):
        """รับค่าการวัดจากผู้ใช้พร้อมการตรวจสอบความถูกต้อง"""
        while True:
            try:
                user_input = input(f"โปรดวัดระยะทางจริงที่หุ่นยนต์เคลื่อนที่ได้ (คาดหวัง {expected_distance} เมตร): ")
                actual_dist = float(user_input)
                
                if actual_dist <= 0:
                    print("กรุณาใส่ค่าที่มากกว่า 0")
                    continue
                    
                if actual_dist > expected_distance * 2:
                    confirm = input(f"ค่าที่คุณใส่ ({actual_dist}m) สูงกว่าที่คาดหวังมาก ต้องการใช้ค่านี้หรือไม่? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                        
                return actual_dist
                
            except ValueError:
                print("กรุณาใส่ตัวเลขที่ถูกต้อง")
            except KeyboardInterrupt:
                print("\nการทำงานถูกยกเลิก")
                return None

    def run_calibration(self, test_distance=1.0, speed=5):
        """
        ลำดับการทำงานหลักของการ calibration
        
        Args:
            test_distance (float): ระยะทางที่ใช้ในการทดสอบ (เมตร)
            speed (float): ความเร็วในการเคลื่อนที่ (m/s)
        """
        
        print("=== โปรแกรม Calibration การเคลื่อนที่ของ RoboMaster S1/EP ===\n")
        
        # เชื่อมต่อหุ่นยนต์
        if not self.initialize_robot():
            return False
            
        try:
            # ขั้นตอนที่ 1: การทดสอบเพื่อหาความคลาดเคลื่อน
            print(f"ขั้นตอนที่ 1: ทดสอบการเคลื่อนที่ {test_distance} เมตร")
            print("โปรดเตรียมที่ว่างให้หุ่นยนต์เคลื่อนที่ และเตรียมอุปกรณ์วัดระยะทาง")
            input("กด Enter เพื่อเริ่มการทดสอบ...")
            
            if not self.test_movement(test_distance, speed):
                return False
            
            # รับค่าการวัดจากผู้ใช้
            actual_dist = self.get_user_measurement(test_distance)
            if actual_dist is None:
                return False
            
            # ขั้นตอนที่ 2: คำนวณค่าตัวคูณการปรับแก้
            calibration_factor = self.calibrate_movement_factor(actual_dist, test_distance)
            
            error_percentage = abs((actual_dist - test_distance) / test_distance) * 100
            print(f"\nผลลัพธ์การ Calibration:")
            print(f"  - ระยะทางที่คาดหวัง: {test_distance:.2f} เมตร")
            print(f"  - ระยะทางจริง: {actual_dist:.2f} เมตร")
            print(f"  - ความคลาดเคลื่อน: {error_percentage:.1f}%")
            print(f"  - ตัวคูณการปรับแก้: {calibration_factor:.4f}")

            # ขั้นตอนที่ 3: ทดสอบการเคลื่อนที่อีกครั้งด้วยค่าที่ปรับแก้แล้ว
            if error_percentage > 5:  # ถ้าความคลาดเคลื่อนมากกว่า 5%
                print(f"\nขั้นตอนที่ 2: ทดสอบด้วยค่าที่ปรับแก้แล้ว")
                confirm = input("ต้องการทดสอบด้วยค่าที่ปรับแก้แล้วหรือไม่? (y/n): ")
                
                if confirm.lower() == 'y':
                    adjusted_dist = test_distance * calibration_factor
                    print(f"สั่งให้หุ่นยนต์เคลื่อนที่: {adjusted_dist:.4f} เมตร เพื่อให้ได้ระยะทางจริง {test_distance} เมตร")
                    
                    if self.test_movement(adjusted_dist, speed):
                        print("กรุณาวัดระยะทางอีกครั้งเพื่อตรวจสอบความแม่นยำ")
                        final_dist = self.get_user_measurement(test_distance)
                        if final_dist:
                            final_error = abs((final_dist - test_distance) / test_distance) * 100
                            print(f"ความคลาดเคลื่อนหลังการปรับแก้: {final_error:.1f}%")
            else:
                print("ความคลาดเคลื่อนอยู่ในเกณฑ์ที่ยอมรับได้ (< 5%)")

            print(f"\nการ Calibration เสร็จสิ้น!")
            print(f"บันทึกค่า Calibration Factor: {calibration_factor:.4f}")
            print("คุณสามารถใช้ค่านี้ในโปรแกรมอื่นๆ โดยคูณกับระยะทางที่ต้องการ")
            print("ตัวอย่างการใช้งาน:")
            print(f"  desired_distance = 2.0  # เมตร")
            print(f"  actual_command = desired_distance * {calibration_factor:.4f}")
            print(f"  chassis.move(x=actual_command, y=0, z=0, xy_speed=0.5)")
            
            return True
            
        except KeyboardInterrupt:
            print("\nการทำงานถูกยกเลิกโดยผู้ใช้")
            return False
        except Exception as e:
            print(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
            return False
    
    def close(self):
        """ปิดการเชื่อมต่อหุ่นยนต์"""
        if self.s1:
            try:
                self.s1.close()
                print("ปิดการเชื่อมต่อหุ่นยนต์แล้ว")
            except:
                pass

def main():
    """ฟังก์ชันหลัก"""
    calibrator = RobotCalibrator()
    
    try:
        # สามารถปรับแก้พารามิเตอร์ได้ตามต้องการ
        success = calibrator.run_calibration(test_distance=1.0, speed=0.5)
        
        if success:
            input("\nกด Enter เพื่อสิ้นสุดโปรแกรม...")
        
    finally:
        calibrator.close()

# ฟังก์ชันง่ายๆ สำหรับการใช้งานด่วน (สำหรับโค้ดเดิม)
def test_movement(distance_in_meters, speed=5):
    """
    ฟังก์ชันเดิมสำหรับความเข้ากันได้ย้อนหลัง
    """
    s1 = robot.Robot()
    s1.initialize(conn_type="ap")
    chassis = s1.chassis
    
    print(f"กำลังทดสอบการเคลื่อนที่: เคลื่อนไปข้างหน้า {distance_in_meters} เมตร...")
    
    try:
        # ใช้ API ปัจจุบัน
        action = chassis.move(x=distance_in_meters, y=0, z=0, xy_speed=speed)
        action.wait_for_completed()
        print("การเคลื่อนที่เสร็จสิ้น")
    except:
        # Fallback method
        estimated_time = distance_in_meters / speed
        chassis.drive_speed(x=speed, y=0, z=0)
        time.sleep(estimated_time)
        chassis.drive_speed(x=0, y=0, z=0)
        print("การเคลื่อนที่เสร็จสิ้น (ใช้ speed control)")
    
    s1.close()

if __name__ == '__main__':
    main()