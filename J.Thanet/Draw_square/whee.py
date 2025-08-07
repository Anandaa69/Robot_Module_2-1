from robomaster import robot
import time
import sys
import math

# --- PID Controller Class ---
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_max = 1.0

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# --- Callback Function and Global Variables ---
current_x = 0.0
current_z_angle = 0.0
position_updated = False

def sub_position_handler(position_info):
    global current_x, current_z_angle, position_updated
    current_x = position_info[0]
    current_z_angle = position_info[2]
    position_updated = True

# --- Main Program ---
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    # กำหนดค่าคงที่สำหรับหุ่นยนต์ (Robomaster EP)
    WHEEL_RADIUS = 0.0625  # รัศมีล้อ (เมตร)
    BASE_WIDTH = 0.35      # ความกว้างระหว่างล้อซ้าย-ขวา (เมตร)
    BASE_LENGTH = 0.35     # ความยาวระหว่างล้อหน้า-หลัง (เมตร)
    
    # คำนวณค่าคงที่สำหรับสูตร
    k1 = 1 / WHEEL_RADIUS
    k2 = BASE_WIDTH / (2 * WHEEL_RADIUS)
    k3 = BASE_LENGTH / (2 * WHEEL_RADIUS)

    print("--- 1. Initializing Robot and Resetting Position ---")
    ep_chassis.move(x=0, y=0, z=0, xy_speed=0.1).wait_for_completed()
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    
    timeout_start = time.time()
    while not position_updated:
        if time.time() - timeout_start > 5:
            print("ERROR: Position data not received. Please check robot connection.")
            ep_robot.close()
            sys.exit()
        time.sleep(0.1)
    
    print(f"Initial position received: X={current_x:.2f}, Z-Angle={current_z_angle:.2f}")

    target_x = 1.2
    target_z_angle = 0

    pid_x = PID(Kp=2.0, Ki=0.05, Kd=0.1, setpoint=target_x)
    pid_z = PID(Kp=2.0, Ki=0.01, Kd=0.2, setpoint=target_z_angle)
    
    last_time = time.time()
    target_reached = False
    
    try:
        print("\n--- 2. Starting Control Loop ---")
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now

            if dt == 0:
                continue

            speed_x = pid_x.compute(current_x, dt)
            speed_z = pid_z.compute(current_z_angle, dt)

            max_speed = 100 # ความเร็วสูงสุดของมอเตอร์ (dps)
            max_speed_x = 0.8 # ความเร็วสูงสุดแกน X (m/s)
            
            # แปลงความเร็วเชิงเส้น (m/s) และความเร็วเชิงมุม (deg/s) เป็น dps ของล้อ
            # ใช้สูตรการแปลงสำหรับ Mecanum Wheel
            # (สูตรนี้ใช้สำหรับการเคลื่อนที่ในแกน X และ Z เท่านั้น)
            v_front_right = k1 * speed_x + k2 * speed_z
            v_front_left = k1 * speed_x - k2 * speed_z
            v_rear_right = k1 * speed_x - k2 * speed_z
            v_rear_left = k1 * speed_x + k2 * speed_z

            # Cap the speeds and convert to integer dps
            wheel_speed_rf = int(max(min(v_front_right, max_speed), -max_speed))
            wheel_speed_lf = int(max(min(v_front_left, max_speed), -max_speed))
            wheel_speed_rb = int(max(min(v_rear_right, max_speed), -max_speed))
            wheel_speed_lb = int(max(min(v_rear_left, max_speed), -max_speed))

            # ตรวจสอบและส่งคำสั่ง
            if abs(speed_x) < 0.02 and abs(speed_z) < 0.01:
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=1)
            else:
                ep_chassis.drive_wheels(w1=wheel_speed_rf, w2=wheel_speed_lf, w3=wheel_speed_rb, w4=wheel_speed_lb, timeout=1)

            print(f"X: {current_x:.2f} | Z-Angle: {current_z_angle:.2f} | Speed_X: {speed_x:.2f} | Speed_Z: {speed_z:.2f}")

            if abs(current_x - target_x) < 0.02:
                print("\n--- 3. Target Reached ---")
                print(f"Final position: X={current_x:.2f}, Z-Angle={current_z_angle:.2f}")
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=1)
                target_reached = True

    except KeyboardInterrupt:
        print("\n--- Program interrupted by user ---")
    finally:
        print("\n--- 4. Stopping robot and cleaning up ---")
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0, timeout=1)
        ep_chassis.unsub_position()
        ep_robot.close()