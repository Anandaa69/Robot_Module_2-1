from robomaster import robot
import time
import math

# -------------------------
# พารามิเตอร์ PID แยกกัน
# -------------------------
# PID สำหรับเดินหน้า-ถอยหลัง (แกน x)
KP_move = 2.1
KI_move = 0.3
KD_move = 10

# PID สำหรับหมุนซ้าย-ขวา (แกน z)
KP_rotate = 4.0   # ตั้งให้สูงกว่าเพื่อควบคุมการหมุน
KI_rotate = 0.5
KD_rotate = 15

RAMP_UP_TIME = 0.5

# -------------------------
# PID Controller Function
# -------------------------
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.integral_max = 1.0  # limit integral term

    def compute(self, current, dt, is_angle=False):
        error = self.setpoint - current

        # ถ้าเป็น angle ให้จัดการ wrap-around
        if is_angle:
            if error > 180:
                error -= 360
            elif error < -180:
                error += 360

        self.integral += error * dt
        # Anti-windup
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max

        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# -------------------------
# Callback Variables
# -------------------------
current_x = 0.0
current_y = 0.0
current_yaw = 0.0  # in degrees 0-360

# -------------------------
# Position Callback
# -------------------------
def sub_position_handler(position_info):
    global current_x, current_y, current_yaw
    current_x = position_info[0]
    current_y = position_info[1]
    yaw_rad = position_info[2]
    current_yaw = math.degrees(yaw_rad) % 360
    print(f"Position x:{current_x:.2f}, y:{current_y:.2f}, yaw:{current_yaw:.1f}°")

# -------------------------
# Movement Function with PID
# -------------------------
def move_forward_with_pid(ep_chassis, target_distance, axis, direction=1):
    pid_move = PID(Kp=KP_move, Ki=KI_move, Kd=KD_move, setpoint=target_distance)

    start_time = time.time()
    last_time = start_time
    target_reached = False

    ramp_up_time = RAMP_UP_TIME
    min_speed = 0.1
    max_speed = 1.5

    if axis == 'x':
        start_pos = current_x
    else:
        start_pos = current_y

    print(f"Move forward {target_distance}m on {axis}-axis, direction {direction}")

    while not target_reached:
        now = time.time()
        dt = now - last_time
        last_time = now

        elapsed = now - start_time

        pos = current_x if axis == 'x' else current_y
        relative_pos = abs(pos - start_pos)

        output = pid_move.compute(relative_pos, dt)

        # ramp up
        if elapsed < ramp_up_time:
            ramp_mul = min_speed + (elapsed / ramp_up_time) * (1.0 - min_speed)
        else:
            ramp_mul = 1.0

        speed = max(min(output * ramp_mul, max_speed), -max_speed)

        ep_chassis.drive_speed(x=speed * direction, y=0, z=0, timeout=1)

        if abs(relative_pos - target_distance) < 0.02:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            print(f"Reached target at pos {pos:.2f}")
            target_reached = True

        time.sleep(0.01)

# -------------------------
# Rotation Function with separate PID
# -------------------------
def rotate_with_pid(ep_chassis, target_angle_deg, clockwise=True):
    pid_rotate = PID(Kp=KP_rotate, Ki=KI_rotate, Kd=KD_rotate, setpoint=target_angle_deg)

    last_time = time.time()
    target_reached = False
    direction = 1 if clockwise else -1
    max_angular_speed = 45

    print(f"Rotate to {target_angle_deg} degrees {'clockwise' if clockwise else 'counter-clockwise'}")

    while not target_reached:
        now = time.time()
        dt = now - last_time
        last_time = now

        current_angle = current_yaw

        output = pid_rotate.compute(current_angle, dt, is_angle=True)

        speed_z = max(min(output, max_angular_speed), -max_angular_speed)

        ep_chassis.drive_speed(x=0, y=0, z=speed_z * direction, timeout=1)

        error = abs((target_angle_deg - current_angle + 180) % 360 - 180)

        if error < 1.0:
            ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
            print(f"Rotation to {target_angle_deg}° complete")
            target_reached = True

        time.sleep(0.01)

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    ep_chassis.move(x=0, y=0, z=0, xy_speed=0.5).wait_for_completed()
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(0.25)

    try:
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=1)
        time.sleep(0.5)
        move_forward_with_pid(ep_chassis, 0.6, 'x', direction=-1)

        rotate_with_pid(ep_chassis, 90, clockwise=True)
        time.sleep(0.5)
        rotate_with_pid(ep_chassis, 0, clockwise=False)

        print("=== All movements completed! ===")

    except KeyboardInterrupt:
        print("Program interrupted!")

    finally:
        try:
            ep_chassis.unsub_position()
        except:
            pass
        ep_robot.close()
