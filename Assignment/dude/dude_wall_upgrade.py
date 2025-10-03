# -*-coding:utf-8-*-

import time
import robomaster
from robomaster import robot
import statistics
import traceback

# =============================================================================
# ===== CONFIGURATION =========================================================
# =============================================================================
BLOCK_DISTANCE_M = 0.6
LEFT_SENSOR_ADAPTOR_ID = 1
LEFT_SENSOR_PORT = 1
LEFT_TARGET_CM = 13.5
RIGHT_SENSOR_ADAPTOR_ID = 2
RIGHT_SENSOR_PORT = 1
RIGHT_TARGET_CM = 13.0

# ⚙️ ความเร็วในการขยับเข้า/ถอยออกเพื่อจัดตำแหน่งกลางโหนด
TOF_ADJUST_SPEED = 0.1  # ✅ ปรับค่าได้เอง

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

# ---------------- ToF calibration ----------------
TOF_CALIBRATION_SLOPE = 0.0894
TOF_CALIBRATION_Y_INTERCEPT = 3.8409

def calibrate_tof_value(raw_tof_value):
    """แปลงค่า raw ToF (mm) ให้เป็น cm"""
    try:
        if raw_tof_value is None:
            return float('inf')
        return (TOF_CALIBRATION_SLOPE * raw_tof_value) + TOF_CALIBRATION_Y_INTERCEPT
    except Exception:
        return float('inf')
# -------------------------------------------------

# =============================================================================
# ===== ROBOT MASTER CONTROLLER CLASS =========================================
# =============================================================================
class RobotMasterController:
    def __init__(self, ep_robot):
        self.robot = ep_robot
        self.chassis = ep_robot.chassis
        self.sensor_adaptor = ep_robot.sensor_adaptor
        
        self.current_yaw = 0.0
        self.current_x = 0.0
        self.master_target_yaw = 0.0

        print("Initializing Controller...")
        self.chassis.sub_attitude(freq=20, callback=self._attitude_callback)
        self.chassis.sub_position(freq=20, callback=self._position_callback)
        time.sleep(0.2)
        self.master_target_yaw = self.current_yaw
        print(f"Controller initialized. Master Yaw set to: {self.master_target_yaw:.2f}°")

        # ✅ Subscribe ToF sensor
        self.sensor = getattr(ep_robot, 'sensor', None)
        self.tof_latest = None
        if self.sensor:
            try:
                self.sensor.sub_distance(freq=10, callback=self._tof_callback)
                time.sleep(0.05)
                print("✅ Subscribed to ToF sensor.")
            except Exception as e:
                print(f"⚠️ ToF subscribe failed: {e}")
        else:
            print("⚠️ ep_robot.sensor not available; ToF will be disabled.")

    def _attitude_callback(self, attitude_info): self.current_yaw = attitude_info[0]
    def _position_callback(self, position_info): self.current_x = position_info[0]

    def _tof_callback(self, sub_info):
        try:
            raw_tof1 = sub_info[0]
            self.tof_latest = calibrate_tof_value(raw_tof1)
        except Exception:
            pass

    def set_master_heading(self):
        self.master_target_yaw = self.current_yaw
        print(f"\n--- New Master Heading Locked: {self.master_target_yaw:.2f}° ---")

    def _calculate_yaw_correction_speed(self):
        KP_YAW, MAX_YAW_SPEED, DEADBAND = 1.8, 25, 0.5
        yaw_error = normalize_angle(self.master_target_yaw - self.current_yaw)
        if abs(yaw_error) < DEADBAND: return 0.0
        speed = KP_YAW * yaw_error
        return max(min(speed, MAX_YAW_SPEED), -MAX_YAW_SPEED)

    def hold_still(self, duration):
        print(f"Active Hold for {duration}s...")
        start_time = time.time()
        while time.time() - start_time < duration:
            correction_speed = self._calculate_yaw_correction_speed()
            self.chassis.drive_speed(x=0, y=0, z=correction_speed, timeout=0.1)
            time.sleep(0.05)
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

    def check_for_wall(self, sensor_id, port, side_name):
        print(f"\n[{side_name}] Performing wall detection check (with active Yaw lock)...")
        CHECK_DURATION_S = 1
        MAX_STD_DEV_THRESHOLD = 0.8
        MAX_AVG_DISTANCE_THRESHOLD = 50
        readings = []
        start_time = time.time()

        while time.time() - start_time < CHECK_DURATION_S:
            yaw_correction_speed = self._calculate_yaw_correction_speed()
            self.chassis.drive_speed(x=0, y=0, z=yaw_correction_speed, timeout=0.1)

            adc = self.sensor_adaptor.get_adc(id=sensor_id, port=port)
            readings.append(convert_adc_to_cm(adc))
            time.sleep(0.05)
        
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

        if len(readings) < 5:
            print(f"[{side_name}] Wall Check Error: Not enough sensor data collected.")
            return False

        avg_distance = statistics.mean(readings)
        std_dev = statistics.stdev(readings)
        print(f"[{side_name}] Wall Check Stats -> Avg Dist: {avg_distance:.2f} cm, Std Dev: {std_dev:.2f}")

        if std_dev > MAX_STD_DEV_THRESHOLD:
            print(f"[{side_name}] Wall NOT Detected: Unstable readings.")
            return False
        if avg_distance > MAX_AVG_DISTANCE_THRESHOLD:
            print(f"[{side_name}] Wall NOT Detected: Too far.")
            return False

        print(f"[{side_name}] Wall detected. Ready for adjustment.")
        return True

    def align_to_master_heading(self, yaw_tolerance=1.5):
        print(f"\n--- Aligning Robot to Master Heading: {self.master_target_yaw:.2f}° ---")
        angle_to_correct = -normalize_angle(self.master_target_yaw - self.current_yaw)
        if abs(angle_to_correct) > yaw_tolerance:
            print(f"🔧 Coarse adjustment by {angle_to_correct:.2f}°...")
            self.chassis.move(x=0, y=0, z=angle_to_correct, z_speed=60).wait_for_completed(timeout=3)
            self.hold_still(0.3)
        
        final_error = normalize_angle(self.master_target_yaw - self.current_yaw)
        if abs(final_error) > yaw_tolerance:
            print(f"⚠️ Fine-tuning by {-final_error:.2f}°...")
            self.chassis.move(x=0, y=0, z=-final_error, z_speed=40).wait_for_completed(timeout=2)
            self.hold_still(0.3)

        final_error_after_tune = normalize_angle(self.master_target_yaw - self.current_yaw)
        if abs(final_error_after_tune) <= yaw_tolerance:
            print(f"✅ Alignment Success! Final Yaw: {self.current_yaw:.2f}°")
        else:
            print(f"🔥 ALIGNMENT FAILED. Final Yaw: {self.current_yaw:.2f}° (Error: {final_error_after_tune:.2f}°)")
        return True

    # ✅ ฟังก์ชันใหม่: เคลื่อนที่ไปข้างหน้าพร้อมรักษาระยะจากกำแพง real-time
    def move_forward_with_wall_follow(self, target_distance,
                                    left_id, left_port, left_target,
                                    right_id, right_port, right_target,
                                    tol_cm=2.0):
        print(f"\n--- Moving Forward with Wall Following ({target_distance}m) ---")
        MOVE_TIMEOUT = 10.0
        BASE_SPEED = 0.25   # ความเร็วเดินหน้าคงที่
        MAX_SLIDE = 0.1     # ความเร็วแกน y สูงสุด
        KP_WALL = 0.02      # ค่าคุมการแก้ด้านข้าง

        start_time = time.time()
        start_position = self.current_x
        last_valid_target = (left_target + right_target) / 2

        while time.time() - start_time < MOVE_TIMEOUT:
            relative_pos = abs(self.current_x - start_position)
            if relative_pos >= target_distance - 0.02:
                break

            # ========= อ่านค่าจากเซนเซอร์ =========
            left_val = convert_adc_to_cm(self.sensor_adaptor.get_adc(id=left_id, port=left_port))
            right_val = convert_adc_to_cm(self.sensor_adaptor.get_adc(id=right_id, port=right_port))

            active_sides = []
            if left_val < 50:
                active_sides.append(left_val)
            if right_val < 50:
                active_sides.append(right_val)

            if active_sides:
                current_target = sum(active_sides) / len(active_sides)
                last_valid_target = current_target
            else:
                current_target = last_valid_target

            # ========= คำนวณ error =========
            dist_error = 0
            if left_val < 50 and right_val < 50:
                dist_error = ((left_val + right_val) / 2) - current_target
            elif left_val < 50:
                dist_error = left_val - current_target
            elif right_val < 50:
                dist_error = right_val - current_target

            # ========= คำนวณการแก้ด้านข้าง =========
            slide_speed = 0.0
            if abs(dist_error) > tol_cm:
                slide_speed = max(min(-KP_WALL * dist_error, MAX_SLIDE), -MAX_SLIDE)

            # ========= เดินไปข้างหน้า + ปรับนิดหน่อย =========
            yaw_correction = self._calculate_yaw_correction_speed()
            self.chassis.drive_speed(x=BASE_SPEED, y=slide_speed, z=yaw_correction, timeout=0.1)

            print(f"Moving... {relative_pos:.2f}/{target_distance:.2f} m | "
                f"L:{left_val:.1f} R:{right_val:.1f} CorrY:{slide_speed:.3f}", end="\r")

        # หยุดเมื่อถึงเป้าหมาย
        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        print(f"\nMoved {abs(self.current_x - start_position):.3f}m (with wall following).")

    def cleanup(self):
        print("Closing controller...")
        self.chassis.unsub_attitude()
        self.chassis.unsub_position()
        try:
            if getattr(self, 'sensor', None):
                self.sensor.unsub_distance()
        except:
            pass

# =============================================================================
# ===== MAIN EXECUTION ========================================================
# =============================================================================
def main():
    ep_robot, controller = None, None
    try:
        ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_robot.gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        controller = RobotMasterController(ep_robot)
        controller.hold_still(1)
        
        while True:
            try:
                num_blocks_str = input("🤖 เดินไปกี่บล็อก (0.6m/บล็อก) | พิมพ์ 'exit' เพื่อออก: ")
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

            controller.set_master_heading()
            controller.hold_still(0.15)
            
            print("\n--- Stage 2: Wall Detection & Side Alignment ---")
            left_wall_present = controller.check_for_wall(LEFT_SENSOR_ADAPTOR_ID, LEFT_SENSOR_PORT, "Left")
            right_wall_present = controller.check_for_wall(RIGHT_SENSOR_ADAPTOR_ID, RIGHT_SENSOR_PORT, "Right")

            if not left_wall_present and not right_wall_present:
                print("\n⚠️  WARNING: No walls detected. จะใช้ wall-following ด้วยค่าล่าสุดแทน")
                controller.hold_still(0.15)

            controller.align_to_master_heading()
            controller.hold_still(0.15)

            # ✅ เคลื่อนที่พร้อมรักษาระยะจากกำแพงแบบ real-time
            controller.move_forward_with_wall_follow(
                BLOCK_DISTANCE_M,
                LEFT_SENSOR_ADAPTOR_ID, LEFT_SENSOR_PORT, LEFT_TARGET_CM,
                RIGHT_SENSOR_ADAPTOR_ID, RIGHT_SENSOR_PORT, RIGHT_TARGET_CM
            )
            controller.hold_still(0.15)
            

            print(f"\n--- ✅ Block {i + 1} complete. ---")
            controller.hold_still(0.15)

        print("\n🎉🎉🎉 SEQUENCE FINISHED! 🎉🎉🎉")

    except KeyboardInterrupt: print("\n\n⚠️ Program stopped by user.")
    except Exception as e: print(f"\n❌ Error: {e}"); traceback.print_exc()
    finally:
        print("\n🔌 Cleaning up...")
        if controller: controller.cleanup()
        if ep_robot: ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=1); ep_robot.close()
        print("🔌 Connection closed.")

if __name__ == '__main__':
    main()
