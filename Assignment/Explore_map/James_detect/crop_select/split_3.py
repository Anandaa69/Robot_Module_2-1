import cv2
import robomaster
from robomaster import robot
import time

# --- ตัวแปรโกลบอลสำหรับเก็บสถานะ ---
dividers_x = []  # เก็บพิกัด X ของเส้นแบ่ง (ค่า x1, x2)
MODE = "DEFINING"  # สถานะของโปรแกรม: "DEFINING" หรือ "TESTING"
WINDOW_NAME = "Gimbal Zone Definer"

def mouse_callback(event, x, y, flags, param):
    """
    ฟังก์ชันจัดการอีเวนต์ของเมาส์
    """
    global dividers_x, MODE

    # เมื่อมีการคลิกซ้าย
    if event == cv2.EVENT_LBUTTONDOWN:
        if MODE == "DEFINING":
            dividers_x.append(x)
            print(f"กำหนดเส้นแบ่งที่ {len(dividers_x)} ที่พิกัด x = {x}")
            
            # เมื่อกำหนดครบ 2 เส้นแล้ว
            if len(dividers_x) == 2:
                # จัดเรียงเพื่อให้ค่าแรกน้อยกว่าเสมอ
                dividers_x.sort()
                MODE = "TESTING"
                print("-" * 30)
                print("✅ กำหนดโซนเรียบร้อยแล้ว!")
                print(f"   - เส้นแบ่งที่ 1: x = {dividers_x[0]}")
                print(f"   - เส้นแบ่งที่ 2: x = {dividers_x[1]}")
                print("👉 เข้าสู่โหมดทดสอบ: คลิกบนภาพเพื่อดูว่าอยู่โซนไหน")
                print("-" * 30)
        
        elif MODE == "TESTING":
            x1, x2 = dividers_x[0], dividers_x[1]
            zone = ""
            if x < x1:
                zone = "LEFT"
            elif x1 <= x < x2:
                zone = "CENTER"
            else: # x >= x2
                zone = "RIGHT"
            print(f"🖱️ คลิกที่พิกัด (x={x}) อยู่ในโซน: {zone}")

def draw_ui_on_frame(frame):
    """
    วาดเส้นแบ่ง, ชื่อโซน, และข้อความแนะนำลงบนเฟรมภาพ
    """
    display_frame = frame.copy()
    height, width, _ = display_frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    # วาดเส้นแบ่งที่มีอยู่
    for x_val in dividers_x:
        cv2.line(display_frame, (x_val, 0), (x_val, height), (0, 255, 255), 2)

    # แสดงข้อความแนะนำตามสถานะ (MODE)
    if MODE == "DEFINING":
        instruction = f"Click to set divider {len(dividers_x) + 1}/2"
        cv2.putText(display_frame, instruction, (20, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    elif MODE == "TESTING":
        x1, x2 = dividers_x[0], dividers_x[1]
        
        # คำนวณตำแหน่งกลางของแต่ละโซนเพื่อวางข้อความ
        left_mid = x1 // 2
        center_mid = (x1 + x2) // 2
        right_mid = (x2 + width) // 2

        # วาดชื่อโซน
        cv2.putText(display_frame, "LEFT", (left_mid - 50, 50), font, 1.2, (255, 255, 0), 3)
        cv2.putText(display_frame, "CENTER", (center_mid - 70, 50), font, 1.2, (255, 255, 0), 3)
        cv2.putText(display_frame, "RIGHT", (right_mid - 60, 50), font, 1.2, (255, 255, 0), 3)
        cv2.putText(display_frame, "Zones Defined. Click to test.", (20, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # แสดงข้อความแนะนำการควบคุมทั่วไป
    cv2.putText(display_frame, "r: Reset | s: Save Frame | q: Quit", (20, height - 20), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return display_frame

if __name__ == '__main__':
    ep_robot = robot.Robot()
    
    try:
        print("🤖 กำลังเชื่อมต่อกับ Robomaster...")
        ep_robot.initialize(conn_type="ap")
        print("✅ เชื่อมต่อสำเร็จ!")

        ep_robot.camera.start_video_stream(display=False, resolution="720p")
        print("📷 เปิดกล้องแล้ว รอสักครู่...")
        time.sleep(2) # รอให้กล้องนิ่ง

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
        
        print("\n--- 🚀 เริ่มโปรแกรมกำหนดโซน ---")
        print("คำแนะนำ:")
        print("  1. หน้าต่างวิดีโอจะปรากฏขึ้น")
        print("  2. คลิกซ้าย 2 ครั้งบนวิดีโอเพื่อกำหนดเส้นแบ่ง 2 เส้น")
        print("  3. เมื่อกำหนดครบแล้ว โปรแกรมจะเข้าสู่โหมดทดสอบ")
        print("  4. กด 'r' เพื่อรีเซ็ต, 's' เพื่อบันทึกภาพ, 'q' เพื่อปิด")

        while True:
            frame = ep_robot.camera.read_cv2_image(timeout=5)
            if frame is None:
                print("⚠️ ไม่ได้รับเฟรมภาพ...")
                time.sleep(0.5)
                continue

            # วาด UI ทั้งหมดลงบนเฟรม
            display_frame = draw_ui_on_frame(frame)
            
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            # กด 'q' หรือ ESC เพื่อออก
            if key == ord('q') or key == 27:
                break
            
            # กด 'r' เพื่อรีเซ็ต
            if key == ord('r'):
                print("\n🔄 รีเซ็ตเส้นแบ่ง เริ่มต้นใหม่...")
                dividers_x = []
                MODE = "DEFINING"
            
            # กด 's' เพื่อบันทึกภาพ
            if key == ord('s'):
                filename = f"capture_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"\n📸 บันทึกภาพหน้าจอเป็น '{filename}'")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

    finally:
        if len(dividers_x) == 2:
            print("\n--- สรุปผล ---")
            print(f"พิกัดเส้นแบ่งสุดท้ายคือ: x1 = {dividers_x[0]} และ x2 = {dividers_x[1]}")
            print("นำค่าเหล่านี้ไปใช้ในสคริปต์หลักของคุณได้เลย")

        print("\n🔌 กำลังปิดการเชื่อมต่อ...")
        cv2.destroyAllWindows()
        ep_robot.camera.stop_video_stream()
        ep_robot.close()
        print("✅ ปิดการเชื่อมต่อเรียบร้อย")