import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera

# --------------------------------------------------------------------
# ส่วนที่ 1: ฟังก์ชันทั้งหมด (ไม่มีการเปลี่ยนแปลง)
# --------------------------------------------------------------------

def create_pink_mask(img_rgb):
    """สร้าง mask สำหรับสีชมพู โดยใช้ HSV color space"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # ช่วงค่าสีชมพู (สามารถปรับได้ตามสภาพแสง)
    lower_pink = np.array([120, 10, 100])
    upper_pink = np.array([170, 100, 200])
    
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # ลด Noise
    mask = cv2.medianBlur(mask, 7)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def match_template_masked(img_masked, tmpl_masked, threshold=0.7):
    """Template matching บน masked grayscale images และคืนค่าพร้อม confidence score"""
    if tmpl_masked.shape[0] > img_masked.shape[0] or tmpl_masked.shape[1] > img_masked.shape[1]:
        return []
        
    result = cv2.matchTemplate(img_masked, tmpl_masked, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    
    boxes = []
    h, w = tmpl_masked.shape
    
    for pt in zip(*locations[::-1]):
        confidence = result[pt[1], pt[0]]
        boxes.append(((pt[0], pt[1]), (pt[0] + w, pt[1] + h), confidence))
    
    return boxes

def calculate_iou(box1, box2):
    """คำนวณ Intersection over Union (IoU) ระหว่าง 2 กรอบ"""
    # box format: ((x1, y1), (x2, y2), ...) -> เราสนใจแค่ 2 ค่าแรก
    x1_1, y1_1 = box1[0]
    x2_1, y2_1 = box1[1]
    x1_2, y1_2 = box2[0]
    x2_2, y2_2 = box2[1]
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def non_maximum_suppression_combined(all_detections, iou_threshold=0.3):
    """
    Non-Maximum Suppression เพื่อกำจัดกรอบที่ทับซ้อนจากทุก Template รวมกัน
    ฟังก์ชันนี้จะรับลิสต์ของ detections ทั้งหมด (จาก T1, T2, T3)
    และคืนค่าเฉพาะกรอบที่ดีที่สุดสำหรับแต่ละ object
    """
    if not all_detections:
        return []
    
    # *** จุดสำคัญที่ 1: เรียงลำดับ detections ทั้งหมดจากทุก Template ตาม confidence score ***
    # ไม่ว่ากรอบจะมาจาก T1, T2 หรือ T3 ถ้า score สูงสุด ก็จะถูกพิจารณาก่อน
    # ข้อมูลในลิสต์มี format: (top_left, bottom_right, confidence, template_id, color)
    # เราจึง sort ด้วย index ที่ 2 (confidence)
    detections_sorted = sorted(all_detections, key=lambda x: x[2], reverse=True)
    
    selected_boxes = []
    while detections_sorted:
        # *** จุดสำคัญที่ 2: หยิบกรอบที่ confidence สูงสุดออกมาเสมอ ***
        current_best_box = detections_sorted.pop(0)
        selected_boxes.append(current_best_box)
        
        # *** จุดสำคัญที่ 3: คัดกรองกรอบที่เหลือทั้งหมดออก ***
        # โดยจะลบกรอบใดๆ (ไม่ว่าจะมาจาก Template ไหน) ที่มีค่า IoU กับกรอบที่ดีที่สุด
        # ที่เราเพิ่งหยิบออกมา สูงกว่า threshold ที่กำหนด
        detections_sorted = [
            box for box in detections_sorted
            if calculate_iou(current_best_box, box) < iou_threshold
        ]
        
    return selected_boxes

# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program)
# ---------------------------------------------------

def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับ"""
    
    TEMPLATE_FILES = [
        "image/template/template_night_pic1_x_557_y_266_w_107_h_275.jpg",
        "image/template/template_night_pic2_x_607_y_281_w_55_h_143.jpg",
        "image/template/template_night_pic3_x_614_y_284_w_43_h_95.jpg"
    ]
    MATCH_THRESHOLD = 0.55
    IOU_THRESHOLD = 0.3 # <--- ลองปรับค่านี้ดู อาจจะเพิ่มเป็น 0.4 หรือ 0.5
    
    print("Loading and processing templates...")
    templates_masked = []
    try:
        for f in TEMPLATE_FILES:
            tmpl = cv2.imread(f)
            if tmpl is None:
                raise FileNotFoundError(f"Template file not found: {f}")
            tmpl_rgb = cv2.cvtColor(tmpl, cv2.COLOR_BGR2RGB)
            tmpl_pink_mask = create_pink_mask(tmpl_rgb)
            tmpl_gray = cv2.cvtColor(tmpl_rgb, cv2.COLOR_RGB2GRAY)
            tmpl_masked = cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask)
            templates_masked.append(tmpl_masked)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    print("Robot connected. Starting real-time detection...")

    try:
        last_time = time.time()
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None: continue

            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            main_pink_mask = create_pink_mask(frame_rgb)
            main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)

            # *** หัวใจของการทำงาน: รวบรวมผลจากทุก Template ลงในลิสต์เดียว ***
            all_detections = []
            colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]
            
            for i, tmpl_masked in enumerate(templates_masked):
                boxes = match_template_masked(main_masked, tmpl_masked, threshold=MATCH_THRESHOLD)
                for top_left, bottom_right, confidence in boxes:
                    # เพิ่มข้อมูลทั้งหมดลงไปในลิสต์เดียวกัน
                    all_detections.append((top_left, bottom_right, confidence, i, colors[i]))

            # *** เรียกใช้ NMS กับลิสต์ที่รวบรวมผลจากทุก Template แล้ว ***
            # ณ จุดนี้ ฟังก์ชัน NMS จะไม่สนใจว่ากรอบไหนมาจาก T1, T2, หรือ T3
            # มันจะมองแค่ confidence score และค่า IoU เท่านั้น
            final_detections = non_maximum_suppression_combined(all_detections, IOU_THRESHOLD)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # วาดผลลัพธ์สุดท้ายที่ผ่านการคัดกรองแล้ว
            for top_left, bottom_right, confidence, template_id, color in final_detections:
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
                label = f"Best (T{template_id+1}): {confidence:.2f}" # ระบุว่ามาจาก Template ไหน
                cv2.putText(frame, label, (top_left[0], top_left[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Robomaster Pink Cup Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Stopping detection and releasing resources.")
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()