import numpy as np
import matplotlib.pyplot as plt # ใช้สำหรับแสดงกราฟ (ถ้าติดตั้ง)

def calibrate_sensor_linear_regression():
    """
    ฟังก์ชันสำหรับสอบเทียบเซ็นเซอร์โดยใช้ Linear Regression
    """
    print("--- เริ่มต้นการสอบเทียบเซ็นเซอร์ด้วย Linear Regression ---")
    print("ใช้ข้อมูลที่ป้อนไว้ล่วงหน้า และสามารถเพิ่มข้อมูลใหม่ได้")

    # --- ข้อมูลที่คุณป้อนไว้ล่วงหน้า ---
    sensor_readings = [160, 263, 410, 495, 598, 723, 826, 931, 1041, 1259, 1453, 1675, 1885, 2095, 2312, 2508, 2708, 2914, 3120, 3279, 3469]  # เก็บค่าที่เซ็นเซอร์อ่านได้ (ค่าดิบ)
    actual_distances = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340] # เก็บระยะทางจริงที่วัดได้
    # ---------------------------------

    print(f"\nข้อมูลเริ่มต้นที่มี: {len(sensor_readings)} คู่")
    for i in range(len(sensor_readings)):
        print(f"  เซ็นเซอร์={sensor_readings[i]}, จริง={actual_distances[i]}")

    # --- ส่วนนี้เปิดโอกาสให้ป้อนข้อมูลเพิ่มเติม (ถ้าต้องการ) ---
    print("\n--- ต้องการป้อนข้อมูลเพิ่มเติมหรือไม่? ---")
    print("คุณสามารถป้อน 'ค่าที่เซ็นเซอร์อ่านได้' และ 'ระยะทางจริง' เพิ่มเติมได้")
    print("เมื่อต้องการหยุด ให้พิมพ์ 'done' ที่ช่องค่าเซ็นเซอร์")
    
    while True:
        try:
            sensor_input = input("\nป้อนค่าที่เซ็นเซอร์อ่านได้เพิ่มเติม (หรือ 'done' เพื่อจบ): ")
            if sensor_input.lower() == 'done':
                break
            
            sensor_val = float(sensor_input)

            actual_input = input(f"ป้อนระยะทางจริงที่วัดได้สำหรับค่า {sensor_val} (หน่วยเดียวกันกับเซ็นเซอร์): ")
            actual_val = float(actual_input)

            sensor_readings.append(sensor_val)
            actual_distances.append(actual_val)
            print(f"เพิ่มข้อมูลแล้ว: เซ็นเซอร์={sensor_val}, จริง={actual_val}")

        except ValueError:
            print("ข้อผิดพลาด: โปรดป้อนค่าตัวเลขที่ถูกต้อง หรือ 'done' เพื่อจบ")
        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {e}")

    if len(sensor_readings) < 2:
        print("\n** ต้องมีข้อมูลอย่างน้อย 2 คู่เพื่อทำการถดถอยเชิงเส้นได้ **")
        print("ยกเลิกการสอบเทียบ")
        return

    # แปลงข้อมูลเป็น NumPy array
    x = np.array(sensor_readings) # ค่าเซ็นเซอร์ (ตัวแปรอิสระ)
    y = np.array(actual_distances) # ระยะทางจริง (ตัวแปรตาม)

    print("\n--- กำลังทำการวิเคราะห์ Linear Regression ---")
    # ใช้ np.polyfit(x, y, 1) เพื่อหา m (slope) และ c (y-intercept)
    # เลข 1 หมายถึงการทำ Polynomial Regression ดีกรี 1 ซึ่งก็คือ Linear Regression
    m, c = np.polyfit(x, y, 1)

    print("\n--- ผลลัพธ์การสอบเทียบ ---")
    print(f"สมการการสอบเทียบของคุณคือ: ")
    print(f"ระยะทางจริง = ({m:.4f} * ค่าเซ็นเซอร์) + ({c:.4f})")
    print(f"ดังนั้น ค่า Slope (m): {m:.4f}")
    print(f"และ ค่า Y-intercept (c): {c:.4f}")

    # --- การแสดงกราฟ (เป็นทางเลือก) ---
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='blue', label='ข้อมูลที่เก็บได้ (Sensor vs Actual)')
        plt.plot(x, m*x + c, color='red', label='เส้น Regression (ค่าที่สอบเทียบแล้ว)')
        plt.xlabel('ค่าที่เซ็นเซอร์อ่านได้')
        plt.ylabel('ระยะทางจริง')
        plt.title('กราฟแสดงการสอบเทียบเซ็นเซอร์ด้วย Linear Regression')
        plt.grid(True)
        plt.legend()
        plt.show()
    except ImportError:
        print("\n** คำแนะนำ: ติดตั้ง matplotlib (pip install matplotlib) เพื่อดูผลลัพธ์ในรูปแบบกราฟได้ **")

    # --- การทดสอบการใช้งานค่าที่ได้ ---
    print("\n--- ทดสอบการใช้งานค่าสอบเทียบ ---")
    while True:
        test_input = input("ป้อนค่าที่เซ็นเซอร์อ่านได้เพื่อทดสอบ (หรือ 'q' เพื่อออก): ")
        if test_input.lower() == 'q':
            break
        try:
            test_sensor_val = float(test_input)
            calibrated_value = (m * test_sensor_val) + c
            print(f"ถ้าเซ็นเซอร์อ่านได้ {test_sensor_val:.2f}, ค่าที่สอบเทียบแล้วคือ: {calibrated_value:.2f}")
        except ValueError:
            print("โปรดป้อนค่าตัวเลขที่ถูกต้อง")

# เรียกใช้งานฟังก์ชัน
if __name__ == "__main__":
    calibrate_sensor_linear_regression()

# (0.0894 * ค่าเซ็นเซอร์) + (3.8409)

#---- 7 block ----
# สมการการสอบเทียบของคุณคือ:
# ระยะทางจริง = (0.0960 * ค่าเซ็นเซอร์) + (1.2260)
# ดังนั้น ค่า Slope (m): 0.0960
# และ ค่า Y-intercept (c): 1.2260