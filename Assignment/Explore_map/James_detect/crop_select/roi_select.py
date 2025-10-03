import cv2
import robomaster
from robomaster import robot
import time

# Dictionary สำหรับเก็บสถานะการเลือก ROI ด้วยเมาส์
roi_state = {
    'start_point': None,
    'end_point': None,
    'drawing': False,
    'roi_defined': False
}

def select_roi_callback(event, x, y, flags, param):
    """ฟังก์ชัน Callback สำหรับจัดการ Event ของเมาส์"""
    global roi_state

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_state['start_point'] = (x, y)
        roi_state['end_point'] = (x, y)
        roi_state['drawing'] = True
        roi_state['roi_defined'] = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_state['drawing']:
            roi_state['end_point'] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_state['end_point'] = (x, y)
        roi_state['drawing'] = False
        roi_state['roi_defined'] = True

def main():
    """ฟังก์ชันหลักของโปรแกรม"""
    
    # สร้าง instance ของหุ่นยนต์
    ep_robot = robot.Robot()
    final_roi = None

    try:
        # --- 1. เชื่อมต่อกับหุ่นยนต์ ---
        print("🤖 Attempting to connect to Robomaster...")
        # (หากคุณเชื่อมต่อแบบ Station Mode ให้เปลี่ยน conn_type="sta")
        ep_robot.initialize(conn_type="ap")
        print("✅ Connection successful!")

        # --- 2. เปิดสตรีมวิดีโอจากกล้อง ---
        # display=False คือเราจะจัดการการแสดงผลเองด้วย cv2.imshow
        ep_robot.camera.start_video_stream(display=False, resolution='720p')
        print("📷 Gimbal camera stream is ON.")
        # รอสักครู่เพื่อให้สตรีมวิดีโอนิ่ง
        time.sleep(1)

        window_name = "Gimbal ROI Selector"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_roi_callback)

        print("\nINSTRUCTIONS:")
        print("  - Drag the mouse on the video feed to draw a rectangle.")
        print("  - Press 'c' to confirm the selection and get coordinates.")
        print("  - Press 'r' to reset the selection.")
        print("  - Press 'q' to quit.")

        # --- 3. ลูปสำหรับเลือก ROI ---
        while True:
            # อ่านเฟรมล่าสุดจากกล้องของหุ่นยนต์
            frame = ep_robot.camera.read_cv2_image(timeout=2)
            
            # ถ้าไม่ได้รับเฟรม ให้ข้ามไปรอบถัดไป
            if frame is None:
                continue

            clone = frame.copy()

            # วาดสี่เหลี่ยม ROI บนหน้าจอ
            if roi_state['start_point'] and roi_state['end_point']:
                cv2.rectangle(clone, roi_state['start_point'], roi_state['end_point'], (0, 255, 0), 2)
            
            # แสดงผล
            cv2.imshow(window_name, clone)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting program.")
                break
            
            elif key == ord('r'):
                print("Selection reset. Please draw a new ROI.")
                roi_state['start_point'] = None
                roi_state['end_point'] = None
                roi_state['roi_defined'] = False

            elif key == ord('c'):
                if roi_state['roi_defined']:
                    x1, y1 = roi_state['start_point']
                    x2, y2 = roi_state['end_point']
                    
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x1 - x2)
                    h = abs(y1 - y2)

                    if w > 0 and h > 0:
                        final_roi = (x, y, w, h)
                        print("\n" + "="*40)
                        print(f"✅ ROI Confirmed!")
                        print(f"   - Pixel Coordinates (x, y, w, h): {final_roi}")
                        print("="*40)
                        break # ออกจากลูปเมื่อยืนยันสำเร็จ
                    else:
                        print("⚠️ Invalid ROI. Please draw a box with size.")
                        roi_state['roi_defined'] = False
                else:
                    print("⚠️ No ROI has been drawn yet. Please draw a box first.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

    finally:
        # --- 4. ปิดการเชื่อมต่อและคืนทรัพยากร ---
        # ส่วนนี้จะทำงานเสมอ ไม่ว่าโปรแกรมจะจบลงอย่างไร
        print("\n🔌 Shutting down and cleaning up resources...")
        try:
            # ตรวจสอบว่า ep_robot ถูก initialize แล้วหรือยัง
            if ep_robot._is_initialized:
                ep_robot.camera.stop_video_stream()
                ep_robot.close()
                print("✅ Robot connection closed.")
        except Exception as e:
            print(f"   (Note) Error during cleanup, but continuing: {e}")
        
        cv2.destroyAllWindows()
        print("✅ Program finished.")

# เริ่มการทำงานของโปรแกรม
if __name__ == '__main__':
    # ตรวจสอบว่าได้ติดตั้ง robomaster หรือยัง
    try:
        import robomaster
    except ImportError:
        print("\nError: robomaster library not found.")
        print("Please install it using: pip install robomaster")
        
    main()