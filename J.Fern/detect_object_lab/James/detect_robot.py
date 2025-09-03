import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera

# --------------------------------------------------------------------
# ส่วนที่ 1: ฟังก์ชันทั้งหมดที่คัดลอกมาจาก pyramid_method.ipynb
# --------------------------------------------------------------------

def create_pink_mask(img_rgb):
    """สร้าง mask สำหรับสีชมพู โดยใช้ HSV color space"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # ช่วงค่าสีชมพู (สามารถปรับได้ตามสภาพแสง)
    lower_pink = np.array([120, 10, 120])
    upper_pink = np.array([170, 100, 200])
    
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # ลด Noise
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def match_template_masked(img_masked, tmpl_masked, threshold=0.7):
    """Template matching บน masked grayscale images และคืนค่าพร้อม confidence score"""
    if tmpl_masked.shape[0] > img_masked.shape[0] or tmpl_masked.shape[1] > img_masked.shape[1]:
        # ถ้า template ใหญ่กว่าภาพหลัก ให้ข้ามไป
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
    # box format: ((x1, y1), (x2, y2), confidence)
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

def non_maximum_suppression(boxes_with_scores, iou_threshold=0.3):
    """Non-Maximum Suppression เพื่อกำจัดกรอบที่ทับซ้อน"""
    if not boxes_with_scores:
        return []
    
    # เรียงตาม confidence score (จากมากไปน้อย)
    boxes_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    selected_boxes = []
    while boxes_with_scores:
        current_box = boxes_with_scores.pop(0)
        selected_boxes.append(current_box)
        
        # คัดกรองกรอบที่เหลือ โดยเก็บเฉพาะอันที่มี IoU กับกรอบปัจจุบันน้อยกว่า threshold
        boxes_with_scores = [
            box for box in boxes_with_scores
            if calculate_iou(current_box, box) < iou_threshold
        ]
    return selected_boxes

# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program)
# ---------------------------------------------------

def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับ"""
    
    # --- ค่าที่ปรับได้ ---
    TEMPLATE_FILES = [
        "image/template/template_pic1_x_573_y_276_w_115_h_312.jpg",
        "image/template/template_pic2_x_634_y_291_w_50_h_134.jpg",
        "image/template/template_pic3_x_629_y_291_w_35_h_92.jpg"
    ]
    MATCH_THRESHOLD = 0.65  # ลด threshold ลงเล็กน้อยเพื่อให้ตรวจจับได้ง่ายขึ้นในวิดีโอ
    IOU_THRESHOLD = 0.3
    
    # --- 1. โหลดและประมวลผล Templates ล่วงหน้า (ทำครั้งเดียว) ---
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
        return # ออกจากโปรแกรมถ้าหาไฟล์ไม่เจอ

    # --- 2. เชื่อมต่อกับ Robomaster EP ---
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    
    # !!! แก้ปัญหา Delay: ใช้ความละเอียด 360P แทน 720P !!!
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_540P)
    print("Robot connected. Starting real-time detection...")

    # --- 3. เริ่ม Loop การประมวลผลวิดีโอแบบเรียลไทม์ ---
    try:
        last_time = time.time()
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None:
                continue

            # --- คำนวณ FPS เพื่อวัดประสิทธิภาพ ---
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time

            # --- 4. ประมวลผลแต่ละเฟรมตามขั้นตอนของ pyramid_method ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # สร้าง Mask สำหรับภาพหลัก
            main_pink_mask = create_pink_mask(frame_rgb)
            main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)

            # รวบรวมผลการตรวจจับจากทุก Template
            all_detections = []
            colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255)] # เขียว, ส้ม, น้ำเงิน
            
            for i, tmpl_masked in enumerate(templates_masked):
                boxes = match_template_masked(main_masked, tmpl_masked, threshold=MATCH_THRESHOLD)
                for top_left, bottom_right, confidence in boxes:
                    # เพิ่มข้อมูล: (top_left, bottom_right, confidence, template_id, color)
                    all_detections.append((top_left, bottom_right, confidence, i, colors[i]))

            # --- 5. ใช้ Non-Maximum Suppression (NMS) เพื่อลดกรอบที่ซ้ำซ้อน ---
            final_detections = non_maximum_suppression(all_detections, IOU_THRESHOLD)

            # --- 6. วาดผลลัพธ์ลงบนเฟรม ---
            # แสดง FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for top_left, bottom_right, confidence, template_id, color in final_detections:
                # วาดกรอบ
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
                # สร้างข้อความแสดงผล
                label = f"T{template_id+1}: {confidence:.2f}"
                # วาดข้อความ
                cv2.putText(frame, label, (top_left[0], top_left[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- 7. แสดงผลลัพธ์ ---
            cv2.imshow("Robomaster Pink Cup Detection", frame)

            # กด 'q' เพื่อออก
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # --- 8. คืนทรัพยากรเมื่อจบการทำงาน ---
        print("Stopping detection and releasing resources.")
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()