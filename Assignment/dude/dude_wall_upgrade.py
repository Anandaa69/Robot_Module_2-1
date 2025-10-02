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
RIGHT_SENSOR_ADAPTOR_ID = 2
RIGHT_SENSOR_PORT = 1

TARGET_WALL_DISTANCE_CM = 15.0   # ✅ ระยะห่างจากกำแพงที่ต้องการ (ซ้าย/ขวา)

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

    # ✅ ฟังก์ชันใหม่: เดินไปข้างหน้า + รักษาระยะจากผนัง
    def move_forward_with_wall_following(self, target_distance_m=0.6, target_cm=15.0):
        print(f"\n--- Moving Forward with Wall Following ({target_distance_m} m, {target_cm} cm from wall) ---")
        KP_SIDE = 0.035       # ค่าขยายการแก้ด้านข้าง
        KP_FORWARD = 0.3      # ความเร็วเดินหน้า
        MAX_SPEED_X = 0.4     # จำกัดความเร็วแกน x
        MAX_SPEED_Y = 0.2     # จำกัดความเร็วแกน y

        start_position = self.current_x
        start_time = time.time()
        MOVE_TIMEOUT = 10.0

        while time.time() - start_time < MOVE_TIMEOUT:
            relative_pos = abs(self.current_x - start_position)
            if relative_pos >= target_distance_m:
                print(f"✅ Reached {target_distance_m} m")
                break

            # อ่านค่าเซนเซอร์ซ้าย/ขวา
            left_dist = convert_adc_to_cm(self.sensor_adaptor.get_adc(id=LEFT_SENSOR_ADAPTOR_ID, port=LEFT_SENSOR_PORT))
            right_dist = convert_adc_to_cm(self.sensor_adaptor.get_adc(id=RIGHT_SENSOR_ADAPTOR_ID, port=RIGHT_SENSOR_PORT))

            has_left = left_dist < 50
            has_right = right_dist < 50

            side_error = 0.0
            if has_left and has_right:
                side_error = (left_dist - right_dist) / 2.0
            elif has_left:
                side_error = target_cm - left_dist
            elif has_right:
                side_error = right_dist - target_cm
            else:
                side_error = 0.0

            # คำนวณความเร็ว
            x_speed = min(MAX_SPEED_X, KP_FORWARD)
            y_speed = max(min(KP_SIDE * side_error, MAX_SPEED_Y), -MAX_SPEED_Y)
            yaw_correction = self._calculate_yaw_correction_speed()

            self.chassis.drive_speed(x=x_speed, y=y_speed, z=yaw_correction, timeout=0.1)

            print(f"Dist: {relative_pos:.2f}/{target_distance_m:.2f} m | "
                f"L:{left_dist:.1f} R:{right_dist:.1f} cm | "
                f"Error:{side_error:.2f}", end="\r")

            time.sleep(0.05)

        self.chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        print("\n--- Finished Wall Following Move ---")

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

            controller.master_target_yaw = controller.current_yaw
            controller.hold_still(0.15)

            # ✅ เคลื่อนที่พร้อมรักษาระยะ 15 cm
            controller.move_forward_with_wall_following(BLOCK_DISTANCE_M, target_cm=TARGET_WALL_DISTANCE_CM)
            controller.hold_still(0.2)

            print(f"\n--- ✅ Block {i + 1} complete. ---")

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
