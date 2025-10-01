# -*-coding:utf-8-*-

import time
import robomaster
from robomaster import robot
import threading
import statistics

# =============================================================================
# ===== CONFIGURATION =========================================================
# =============================================================================
BLOCK_DISTANCE_M = 0.6
LEFT_SENSOR_ADAPTOR_ID = 1
LEFT_SENSOR_PORT = 1
LEFT_TARGET_CM = 18.0
RIGHT_SENSOR_ADAPTOR_ID = 2
RIGHT_SENSOR_PORT = 1
RIGHT_TARGET_CM = 14.0

# =============================================================================
# ===== HELPER FUNCTIONS ======================================================
# =============================================================================
def convert_adc_to_cm(adc_value):
    if adc_value <= 0: return float('inf')
    return 30263 * (adc_value ** -1.352)

def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle <= -180: angle += 360
    return angle

# =============================================================================
# ===== ROBOT MASTER CONTROLLER CLASS =========================================
# =============================================================================
class RobotMasterController:
    def __init__(self, ep_robot):
        self.robot = ep_robot
        self.chassis = ep_robot.chassis
        self.sensor_adaptor = ep_robot.sensor_adaptor
        
        # --- State Variables ---
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.master_target_yaw = 0.0 # <-- ตัวแปรใหม่สำหรับทิศทางหลัก

        print("Initializing Controller...")
        self.chassis.sub_attitude(freq=20, callback=self._attitude_callback)
        self.chassis.sub_position(freq=20, callback=self._position_callback)
        time.sleep(0.2)
        self.master_target_yaw = self.current_yaw
        print(f"Controller initialized. Master Yaw set to: {self.master_target_yaw:.2f}°")

    def _attitude_callback(self, attitude_info): self.current_yaw = attitude_info[0]
    def _position_callback(self, position_info): self.current_x = position_info[0]

    def set_master_heading(self):
        """กำหนดทิศทางหลักสำหรับการทำงานในบล็อกถัดไป"""
        self.master_target_yaw = self.current_yaw
        print(f"\n--- New Master Heading Locked: {self.master_target_yaw:.2f}° ---")

    def _calculate_yaw_correction_speed(self):
        """คำนวณความเร็วการหมุนเพื่อ 'ประคอง' ทิศทาง (สำหรับ drive_speed)"""
        KP_YAW, MAX_YAW_SPEED, DEADBAND = 1.8, 25, 0.5
        yaw_error = normalize_angle(self.master_target_yaw - self.current_yaw)
        if abs(yaw_error) < DEADBAND: return 0.0
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def hold_still(self, duration):
        """
        ฟังก์ชันใหม่: 'รอแบบประคองตำแหน่ง' เพื่อแก้ปัญหาหุ่นไถล
        แทนที่ time.sleep() ทั้งหมด
        """
        print(f"Active Hold for {duration}s...")
        start_time = time.time()
        while time.time() - start_time < duration:
            correction_speed = self._calculate_yaw_correction_speed()
            # ส่งคำสั่งให้หยุดนิ่งและแก้ทิศทางตลอดเวลา
            self.chassis.drive_speed(x=0, y=0, z=correction_speed, timeout=0.1)
            time.sleep(0.05)
        # หยุดสนิทเมื่อหมดเวลา
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

    def align_to_master_heading(self, yaw_tolerance=1.5):
        """
        ฟังก์ชันใหม่: ปรับมุม Yaw ความแม่นยำสูง (เลียนแบบ yor_code)
        มีการปรับหยาบและปรับละเอียด (Fine-tuning)
        """
        print(f"\n--- Aligning Robot to Master Heading: {self.master_target_yaw:.2f}° ---")
        
        # --- First Attempt (Coarse Adjustment) ---
        angle_to_correct = -normalize_angle(self.master_target_yaw - self.current_yaw)
        print(f"Current: {self.current_yaw:.2f}°, Correction needed: {angle_to_correct:.2f}°")
        
        if abs(angle_to_correct) > yaw_tolerance:
            print("🔧 Performing coarse adjustment...")
            self.chassis.move(x=0, y=0, z=angle_to_correct, z_speed=60).wait_for_completed(timeout=3)
            self.hold_still(0.3)
        
        # --- Second Attempt (Fine-Tuning) ---
        final_error = normalize_angle(self.master_target_yaw - self.current_yaw)
        if abs(final_error) <= yaw_tolerance:
            print(f"✅ Alignment Success! Final Yaw: {self.current_yaw:.2f}°")
            return True
            
        print(f"⚠️ Fine-tuning required. Remaining error: {final_error:.2f}°")
        angle_to_correct_fine = -final_error
        if abs(angle_to_correct_fine) > 0.5:
             print(f"   🔧 Fine-tuning by: {angle_to_correct_fine:.2f}°")
             self.chassis.move(x=0, y=0, z=angle_to_correct_fine, z_speed=40).wait_for_completed(timeout=2)
             self.hold_still(0.3)

        # Final check
        final_error_after_tune = normalize_angle(self.master_target_yaw - self.current_yaw)
        if abs(final_error_after_tune) <= yaw_tolerance:
            print(f"✅ Fine-Tuning Success! Final Yaw: {self.current_yaw:.2f}°")
        else:
            print(f"🔥🔥 ALIGNMENT FAILED. Final Yaw: {self.current_yaw:.2f}° (Error: {final_error_after_tune:.2f}°)")
        return True

    def adjust_position(self, sensor_id, sensor_port, target_distance_cm, side_name, direction_multiplier):
        print(f"\n--- Adjusting {side_name} Side (Yaw locked at {self.master_target_yaw:.2f}°) ---")
        
        TOLERANCE_CM, MAX_EXEC_TIME, KP_SLIDE, MAX_SLIDE_SPEED = 0.5, 10, 0.035, 0.15
        start_time = time.time()
        
        while time.time() - start_time < MAX_EXEC_TIME:
            current_dist = convert_adc_to_cm(self.sensor_adaptor.get_adc(id=sensor_id, port=sensor_port))
            dist_error = target_distance_cm - current_dist
            
            if abs(dist_error) <= TOLERANCE_CM:
                print(f"\n[{side_name}] Target distance reached!")
                break

            slide_speed = max(min(direction_multiplier * KP_SLIDE * dist_error, MAX_SLIDE_SPEED), -MAX_SLIDE_SPEED)
            yaw_correction = self._calculate_yaw_correction_speed()
            self.chassis.drive_speed(x=0, y=slide_speed, z=yaw_correction)
            
            print(f"Adjusting {side_name}... DistErr: {dist_error:5.2f}cm | YawErr: {normalize_angle(self.master_target_yaw - self.current_yaw):5.2f}°", end='\r')
            time.sleep(0.02)
        else:
             print(f"\n[{side_name}] Movement timed out!")
        
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0) # หยุดสนิท
        return True

    def move_forward_with_pid(self, target_distance):
        print(f"\n--- Moving Forward ({target_distance}m) (Yaw locked at {self.master_target_yaw:.2f}°) ---")
        
        PID_KP, PID_KI, PID_KD = 1.9, 0.25, 10
        RAMP_UP_TIME, MOVE_TIMEOUT = 0.8, 7.0
        
        prev_error, integral = 0, 0
        start_time, last_time = time.time(), time.time()
        start_position = self.current_x
        
        target_reached = False
        while time.time() - start_time < MOVE_TIMEOUT:
            relative_pos = abs(self.current_x - start_position)
            if relative_pos >= target_distance - 0.02:
                target_reached = True; break

            dt = time.time() - last_time; last_time = time.time()
            error = target_distance - relative_pos
            integral += error * dt
            derivative = (error - prev_error) / dt if dt > 0 else 0
            output = PID_KP * error + PID_KI * integral + PID_KD * derivative
            prev_error = error

            ramp = min(1.0, (time.time() - start_time) / RAMP_UP_TIME)
            speed = max(-1, min(1, output)) * ramp
            yaw_correction = self._calculate_yaw_correction_speed()
            
            self.chassis.drive_speed(x=speed, y=0, z=yaw_correction, timeout=0.1)
            print(f"Moving... Dist: {relative_pos:.3f}/{target_distance:.2f} m | YawErr: {normalize_angle(self.master_target_yaw - self.current_yaw):5.2f}°", end='\r')
        
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        print(f"\nMoved a total distance of {abs(self.current_x - start_position):.3f}m")
        print(f"✅ Target reached!" if target_reached else f"⚠️ Movement Timed Out.")

    def cleanup(self):
        print("Closing controller...")
        self.chassis.unsub_attitude()
        self.chassis.unsub_position()

# =============================================================================
# ===== MAIN EXECUTION ========================================================
# =============================================================================
def main():
    ep_robot, controller = None, None
    try:
        ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_robot.gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        controller = RobotMasterController(ep_robot)
        controller.hold_still(1) # เริ่มต้นด้วยการหยุดนิ่ง
        
        while True:
            try:
                num_blocks_str = input("🤖 อยากให้หุ่นเดินไปกี่บล็อค (0.6m/บล็อค) | พิมพ์ 'exit' เพื่อออก: ")
                if num_blocks_str.lower().strip() == 'exit': return
                num_blocks_to_move = int(num_blocks_str)
                if num_blocks_to_move > 0: break
            except ValueError: print("🚨 กรุณาใส่ตัวเลข")

        print(f"\n✅ OK! Starting sequence for {num_blocks_to_move} blocks.\n")
        controller.hold_still(2)

        for i in range(num_blocks_to_move):
            print("\n" + "="*60)
            print(f"===== PROCESSING BLOCK {i + 1} / {num_blocks_to_move} =====")
            print("="*60)

            # 1. กำหนดทิศทางหลักสำหรับบล็อกนี้
            controller.set_master_heading()
            controller.hold_still(0.2)

            # 2. ปรับด้านซ้าย
            controller.adjust_position(LEFT_SENSOR_ADAPTOR_ID, LEFT_SENSOR_PORT, 
                                       LEFT_TARGET_CM, "Left", direction_multiplier=1)
            controller.hold_still(0.2)

            # 3. ปรับด้านขวา
            controller.adjust_position(RIGHT_SENSOR_ADAPTOR_ID, RIGHT_SENSOR_PORT, 
                                       RIGHT_TARGET_CM, "Right", direction_multiplier=-1)
            controller.hold_still(0.2)

            # 4. ปรับมุมให้ตรงเป๊ะ! (หัวใจสำคัญ)
            controller.align_to_master_heading()
            controller.hold_still(0.5) # รอให้มั่นใจว่านิ่ง

            # 5. เดินหน้า
            controller.move_forward_with_pid(BLOCK_DISTANCE_M)
            print(f"\n--- ✅ Block {i + 1} complete. ---")
            controller.hold_still(0.5) # หยุดพักรอเริ่มบล็อกถัดไป

        print("\n🎉🎉🎉 SEQUENCE FINISHED! 🎉🎉🎉")

    except KeyboardInterrupt: print("\n\n⚠️ Program stopped by user.")
    except Exception as e: print(f"\n❌ Error: {e}"); import traceback; traceback.print_exc()
    finally:
        print("\n🔌 Cleaning up...")
        if controller: controller.cleanup()
        if ep_robot: ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=1); ep_robot.close()
        print("🔌 Connection closed.")

if __name__ == '__main__':
    main()