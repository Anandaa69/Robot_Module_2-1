import cv2
import numpy as np
import time
from robomaster import robot
from robomaster import camera

# --------------------------------------------------------------------
# ส่วนที่ 1: ฟังก์ชัน (มีการเพิ่มฟังก์ชันใหม่เข้ามา)
# --------------------------------------------------------------------

def create_pink_mask(img_rgb):
    """สร้าง mask สำหรับสีชมพู โดยใช้ HSV color space"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([120, 10, 100])
    upper_pink = np.array([170, 100, 200])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
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
    x1_1, y1_1 = box1[0]; x2_1, y2_1 = box1[1]
    x1_2, y1_2 = box2[0]; x2_2, y2_2 = box2[1]
    x1_i = max(x1_1, x1_2); y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2); y2_i = min(y2_1, y2_2)
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

# --- ฟังก์ชัน NMS ตัวใหม่ที่ใช้ "ลำดับความสำคัญ" ---
def prioritized_non_maximum_suppression(boxes_with_scores, iou_threshold=0.3):
    """
    NMS ที่ให้ความสำคัญกับ Template ขนาดใหญ่ก่อน (T1 > T2 > T3)
    ถ้ากรอบซ้อนกัน จะเลือกกรอบที่มาจาก Template ที่มี index น้อยกว่าเสมอ
    โดยไม่สนใจค่า confidence score
    """
    if not boxes_with_scores:
        return []

    # เรียงตาม confidence เพื่อจัดการกับวัตถุที่อยู่คนละที่ได้ดีขึ้น
    # แต่การตัดสินใจว่าจะลบอันไหน จะใช้ priority ของ template เป็นหลัก
    detections = sorted(boxes_with_scores, key=lambda x: x[2], reverse=True)
    
    # สร้าง list เพื่อเก็บสถานะว่ากรอบไหนควรถูกลบ (True = ลบ)
    num_detections = len(detections)
    suppress = [False] * num_detections

    for i in range(num_detections):
        if suppress[i]:
            continue

        box_i = detections[i]
        template_i_priority = box_i[3]  # ID ของ template (0=T1, 1=T2, 2=T3)

        for j in range(i + 1, num_detections):
            if suppress[j]:
                continue

            box_j = detections[j]
            
            # --- จุดตัดสินใจหลัก ---
            # ถ้ากรอบทับซ้อนกันมากพอ
            if calculate_iou(box_i, box_j) > iou_threshold:
                template_j_priority = box_j[3]

                # เปรียบเทียบ priority: index น้อยกว่า = priority สูงกว่า
                if template_i_priority <= template_j_priority:
                    # box_i มี priority สูงกว่าหรือเท่ากัน -> ลบ box_j
                    suppress[j] = True
                else:
                    # box_j มี priority สูงกว่า -> ลบ box_i และหยุดเช็คสำหรับ i นี้
                    suppress[i] = True
                    break # ออกจาก vòng j loop

    # รวบรวมเฉพาะกรอบที่ไม่ถูกสั่งให้ลบ
    final_boxes = [detections[i] for i in range(num_detections) if not suppress[i]]
    return final_boxes


# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program)
# ---------------------------------------------------

def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับ"""
    
    # --- สำคัญ: เรียงลำดับไฟล์ Template จากใหญ่ไปเล็ก ---
    # Index 0 (T1) จะมี Priority สูงสุด
    # Index 1 (T2)
    # Index 2 (T3) ...
    TEMPLATE_FILES = [
        "image/template/template_night_pic1_x_557_y_266_w_107_h_275.jpg", # T1 - Priority 1
        "image/template/template_night_pic2_x_607_y_281_w_55_h_143.jpg",  # T2 - Priority 2
        "image/template/template_night_pic4_x_318_y_146_w_17_h_38.jpg"   # T3 - Priority 3
    ]
    MATCH_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.3
    
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
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    print("Robot connected. Starting real-time detection...")

    try:
        last_time = time.time()
        while True:
            frame = ep_camera.read_cv2_image()
            if frame is None:
                continue

            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            main_pink_mask = create_pink_mask(frame_rgb)
            main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)

            all_detections = []
            colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]
            
            for i, tmpl_masked in enumerate(templates_masked):
                boxes = match_template_masked(main_masked, tmpl_masked, threshold=MATCH_THRESHOLD)
                for top_left, bottom_right, confidence in boxes:
                    # `i` คือ ID ของ template ซึ่งจะใช้เป็นตัวกำหนด Priority
                    all_detections.append((top_left, bottom_right, confidence, i, colors[i]))

            # --- *** เรียกใช้ฟังก์ชัน NMS ตัวใหม่ที่ให้ความสำคัญกับ Template *** ---
            final_detections = prioritized_non_maximum_suppression(all_detections, IOU_THRESHOLD)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for top_left, bottom_right, confidence, template_id, color in final_detections:
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
                # ปรับข้อความให้ชัดเจนว่ากรอบนี้ถูกเลือกเพราะ Priority
                label = f"T{template_id+1} (Best Fit)"
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