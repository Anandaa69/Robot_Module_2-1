from robomaster import robot
import time

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
        self.integral_max = 1.0 # Limit integral term

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        
        # Anti-windup: limit the integral term
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < -self.integral_max:
            self.integral = -self.integral_max
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# -------------------------
# Callback and Main Logic
# -------------------------
current_x = 0.0

def sub_position_handler(position_info):
    global current_x
    current_x = position_info[0]
    x, y, z = position_info
    print("chassis position: x:{0}, y:{1}, z:{2}".format(x, y, z))

if __name__ == '__main__':
    # Connect to robot
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    # Reset position to 0 before starting
    ep_chassis.move(x=0, y=0, z=0, xy_speed=0.5).wait_for_completed()
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)
    time.sleep(1)

    # Target distance (meters)
    target_x = 0.6

    # Setup PID
    pid = PID(Kp=1.8, Ki=0.1, Kd=4, setpoint=target_x)

    # Control loop for forward movement
    last_time = time.time()
    target_reached = False
    
    try:
        while not target_reached:
            now = time.time()
            dt = now - last_time
            last_time = now
            output = pid.compute(current_x, dt)

            max_speed = 1.5
            speed = max(min(output, max_speed), -max_speed)

            ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=1)

            if abs(current_x - target_x) < 0.02:
                print("Target reached. Final position: {:.2f}".format(current_x))
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
                target_reached = True
                
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    # ----------------------------------
    # ส่วนคำสั่งสำหรับการหมุน
    # ----------------------------------
    # เพิ่มเวลาหน่วงเล็กน้อยเพื่อให้คำสั่ง drive_speed หยุดทำงานสมบูรณ์
    time.sleep(0.5) 
    
    print("Now turning 90 degrees...")
    
    # ปรับลดความเร็ว z_speed เพื่อเพิ่มความแม่นยำ และเพิ่ม timeout ให้มีระยะเวลาพอสมควร
    ep_chassis.move(x=0, y=0, z=-93, z_speed=45).wait_for_completed(timeout=5)

    # ย้ายการปิดระบบมาไว้ตรงนี้
    ep_robot.close()