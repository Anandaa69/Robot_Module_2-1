import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera

# --------------------------------------------------------------------
# ส่วนที่ 1: ฟังก์ชัน (มีการปรับปรุง)
# --------------------------------------------------------------------

def create_pink_mask(img_rgb):
    """สร้าง mask สำหรับสีชมพู (ปรับปรุง Kernel ให้เล็กลงเพื่อความเร็ว)"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([120, 10, 100])
    upper_pink = np.array([170, 100, 200])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # --- OPTIMIZATION: ลดขนาด Kernel เพื่อความเร็ว ---
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def match_template_masked(img_masked, tmpl_masked, threshold=0.7):
    # ไม่มีการเปลี่ยนแปลง
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
    # ไม่มีการเปลี่ยนแปลง
    x1_1, y1_1 = box1[0]; x2_1, y2_1 = box1[1]
    x1_2, y1_2 = box2[0]; x2_2, y2_2 = box2[1]
    x1_i = max(x1_1, x1_2); y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2); y2_i = min(y2_1, y2_2)
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

# --- ฟังก์ชัน NMS ตัวใหม่ที่ฉลาดขึ้น: ใช้ทั้ง IoU และ Center Point ---
def advanced_prioritized_nms(boxes_with_scores, iou_threshold=0.3):
    """
    NMS ที่ให้ความสำคัญกับ Template ใหญ่ และเพิ่มการตรวจสอบ "Center Point Containment"
    เพื่อแก้ปัญหากรอบซ้อนกันที่ระยะไกล
    """
    if not boxes_with_scores:
        return []

    detections = sorted(boxes_with_scores, key=lambda x: x[3]) # เรียงตาม Priority (T1, T2, T3) ก่อน
    
    final_boxes = []
    while detections:
        # หยิบตัวที่มี Priority สูงสุด (T1 มาก่อน)
        current_box = detections.pop(0)
        final_boxes.append(current_box)
        
        remaining_boxes = []
        for box_to_check in detections:
            iou = calculate_iou(current_box, box_to_check)
            
            # --- ตรรกะใหม่: ตรวจสอบว่าจุดศูนย์กลางของ box_to_check อยู่ใน current_box หรือไม่ ---
            center_x = box_to_check[0][0] + (box_to_check[1][0] - box_to_check[0][0]) / 2
            center_y = box_to_check[0][1] + (box_to_check[1][1] - box_to_check[0][1]) / 2
            
            is_center_inside = (current_box[0][0] < center_x < current_box[1][0] and
                                current_box[0][1] < center_y < current_box[1][1])
            
            # --- เงื่อนไขการลบ ---
            # ลบ box_to_check ทิ้ง ถ้า...
            # 1. IoU สูงเกินไป OR
            # 2. จุดศูนย์กลางของมัน อยู่ในกรอบที่มี Priority สูงกว่า
            if iou > iou_threshold or is_center_inside:
                continue # ไม่ต้องเก็บ box_to_check ไว้
            
            remaining_boxes.append(box_to_check)
            
        detections = remaining_boxes
        
    return final_boxes


# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program) - ปรับปรุงเพื่อลด Lag
# ---------------------------------------------------

def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับ"""
    
    # --- OPTIMIZATION: กำหนดขนาดภาพที่จะใช้ประมวลผล ---
    # เราจะย่อภาพจาก 1280x720 ให้มีความกว้างเหลือ 640px เพื่อลดภาระ
    PROCESSING_WIDTH = 640.0
    
    TEMPLATE_FILES = [
        "image/template/use/template_night_pic1_x_557_y_278_w_120_h_293.jpg", # T1
        "image/template/use/template_night_pic2_x_609_y_290_w_57_h_138.jpg",  # T2
        "image/template/use/template_night_pic3_x_622_y_293_w_40_h_93.jpg"   # T3
    ]
    MATCH_THRESHOLD = 0.55
    IOU_THRESHOLD = 0.1 # ลด IoU ลงเพื่อให้กำจัดกรอบที่แค่ "เฉียดกัน" ได้ดีขึ้น
    
    print("Loading and processing templates...")
    templates_original = []
    try:
        for f in TEMPLATE_FILES:
            tmpl = cv2.imread(f)
            if tmpl is None: raise FileNotFoundError(f"Template file not found: {f}")
            templates_original.append(tmpl)
    except FileNotFoundError as e:
        print(f"Error: {e}"); return

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    
    # --- OPTIMIZATION: คำนวณ Scale และย่อขนาด Templates ล่วงหน้า (ทำครั้งเดียว) ---
    # รอรับเฟรมแรกเพื่อหาขนาดและคำนวณ scale
    print("Waiting for first frame to calculate scale...")
    frame_for_scale = ep_camera.read_cv2_image(timeout=5)
    if frame_for_scale is None:
        print("Error: Could not get frame from camera.")
        ep_robot.close()
        return
        
    h_orig, w_orig, _ = frame_for_scale.shape
    processing_scale = PROCESSING_WIDTH / w_orig
    h_proc = int(h_orig * processing_scale)
    
    print(f"Original frame: {w_orig}x{h_orig}, Processing at: {int(PROCESSING_WIDTH)}x{h_proc}")
    
    templates_masked = []
    for tmpl in templates_original:
        # ย่อขนาด Template ตามสัดส่วนของภาพหลัก
        w_tmpl = int(tmpl.shape[1] * processing_scale)
        h_tmpl = int(tmpl.shape[0] * processing_scale)
        tmpl_resized = cv2.resize(tmpl, (w_tmpl, h_tmpl), interpolation=cv2.INTER_AREA)
        
        # ทำ Mask บน Template ที่ย่อขนาดแล้ว
        tmpl_rgb = cv2.cvtColor(tmpl_resized, cv2.COLOR_BGR2RGB)
        tmpl_pink_mask = create_pink_mask(tmpl_rgb)
        tmpl_gray = cv2.cvtColor(tmpl_rgb, cv2.COLOR_RGB2GRAY)
        tmpl_masked = cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask)
        templates_masked.append(tmpl_masked)
        
    print("Robot connected. Starting real-time detection...")

    try:
        last_time = time.time()
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None: continue

            # --- OPTIMIZATION: ย่อขนาดเฟรมปัจจุบันก่อนประมวลผล ---
            frame_proc = cv2.resize(frame, (int(PROCESSING_WIDTH), h_proc), interpolation=cv2.INTER_AREA)
            
            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            main_pink_mask = create_pink_mask(frame_rgb)
            main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)

            all_detections = []
            colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]
            
            for i, tmpl_masked in enumerate(templates_masked):
                # ประมวลผลบนภาพขนาดย่อ
                boxes = match_template_masked(main_masked, tmpl_masked, threshold=MATCH_THRESHOLD)
                for top_left, bottom_right, confidence in boxes:
                    all_detections.append((top_left, bottom_right, confidence, i, colors[i]))

            # --- *** เรียกใช้ฟังก์ชัน NMS ตัวใหม่ที่ฉลาดขึ้น *** ---
            final_detections = advanced_prioritized_nms(all_detections, IOU_THRESHOLD)
            
            # --- วาดผลลัพธ์ลงบนเฟรม "ขนาดเต็ม" เพื่อการแสดงผลที่ชัดเจน ---
            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for top_left, bottom_right, confidence, template_id, color in final_detections:
                # แปลงพิกัดกลับไปเป็นขนาดของเฟรมดั้งเดิม
                tl_orig = (int(top_left[0] / processing_scale), int(top_left[1] / processing_scale))
                br_orig = (int(bottom_right[0] / processing_scale), int(bottom_right[1] / processing_scale))
                
                cv2.rectangle(frame, tl_orig, br_orig, color, 3) # เพิ่มความหนาของเส้น
                label = f"T{template_id+1} (Priority Fit)"
                cv2.putText(frame, label, (tl_orig[0], tl_orig[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Robomaster Pink Cup Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print("Stopping detection and releasing resources.")
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()

if __name__ == '__main__':
    main()