from robomaster import robot
import time
import matplotlib.pyplot as plt
from collections import deque
import threading

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

    def compute(self, current, dt):
        error = self.setpoint - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# -------------------------
# Plotting Class
# -------------------------
class RealTimePlotter:
    def __init__(self, max_points=200):
        self.max_points = max_points
        self.time_data = deque(maxlen=max_points)
        self.position_data = deque(maxlen=max_points)
        self.target_data = deque(maxlen=max_points)
        self.error_data = deque(maxlen=max_points)
        self.speed_data = deque(maxlen=max_points)
        
        # Setup plot
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('RoboMaster PID Control Analysis', fontsize=16)
        
        # Position plot
        self.ax1.set_title('Position vs Target')
        self.ax1.set_ylabel('Position (m)')
        self.ax1.grid(True, alpha=0.3)
        
        # Error plot
        self.ax2.set_title('Position Error')
        self.ax2.set_ylabel('Error (m)')
        self.ax2.grid(True, alpha=0.3)
        
        # Speed plot
        self.ax3.set_title('Speed Command')
        self.ax3.set_ylabel('Speed (m/s)')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def add_data(self, timestamp, position, target, error, speed):
        self.time_data.append(timestamp)
        self.position_data.append(position)
        self.target_data.append(target)
        self.error_data.append(error)
        self.speed_data.append(speed)
        
    def update_plot(self):
        if len(self.time_data) < 2:
            return
            
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        time_list = list(self.time_data)
        
        # Position plot
        self.ax1.plot(time_list, list(self.position_data), 'b-', label='Current Position', linewidth=2)
        self.ax1.plot(time_list, list(self.target_data), 'r--', label='Target Position', linewidth=2)
        self.ax1.set_title('Position vs Target')
        self.ax1.set_ylabel('Position (m)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Error plot
        self.ax2.plot(time_list, list(self.error_data), 'g-', label='Position Error', linewidth=2)
        self.ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax2.set_title('Position Error')
        self.ax2.set_ylabel('Error (m)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Speed plot
        self.ax3.plot(time_list, list(self.speed_data), 'm-', label='Speed Command', linewidth=2)
        self.ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax3.set_title('Speed Command')
        self.ax3.set_ylabel('Speed (m/s)')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
        
    def save_plot(self, filename='pid_control_analysis.png'):
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

# -------------------------
# Callback and Main Logic
# -------------------------
current_x = 0.0

def sub_position_handler(position_info):
    global current_x
    current_x = position_info[0]
    print("Current X: {:.2f}".format(current_x))

if __name__ == '__main__':
    # Connect to robot
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis = ep_robot.chassis

    # Target distance (meters)
    target_x = 1.8

    # Setup PID
    pid = PID(Kp=0.5, Ki=0.2, Kd=0.2, setpoint=target_x)
    
    # Setup plotter
    plotter = RealTimePlotter()

    # Subscribe to position
    ep_chassis.sub_position(freq=20, callback=sub_position_handler)

    # Control loop
    start_time = time.time()
    last_time = start_time
    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            # คำนวณค่า PID ตามตำแหน่งปัจจุบัน
            output = pid.compute(current_x, dt)

            # จำกัดความเร็วไม่ให้เร็วเกิน
            max_speed = 0.8
            speed = max(min(output, max_speed), -max_speed)

            print("PID output: {:.2f}, speed command: {:.2f}".format(output, speed))
            
            # เพิ่มข้อมูลใน plotter
            error = target_x - current_x
            plotter.add_data(now - start_time, current_x, target_x, error, speed)
            
            # อัพเดทกราฟทุก 10 loops เพื่อไม่ให้ช้า
            if len(plotter.time_data) % 10 == 0:
                plotter.update_plot()

            # คำสั่งเดินในแกน x
            ep_chassis.move(x=speed * dt, y=0, z=0, xy_speed=abs(speed)).wait_for_completed()

            # เงื่อนไขหยุด: เข้าใกล้ target แล้ว
            if abs(current_x - target_x) < 0.05:
                print("Target reached.")
                ep_chassis.stop()
                ep_chassis.unsub_position()
                
                # อัพเดทกราฟครั้งสุดท้ายและบันทึก
                plotter.update_plot()
                plotter.save_plot()
                
                time.sleep(1)
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
        plotter.save_plot('pid_control_interrupted.png')
        
    finally:
        # หยุดการเคลื่อนที่ และยกเลิกการ subscribe
        ep_robot.close()
        plt.ioff()
        plt.show()  # แสดงกราหสุดท้าย