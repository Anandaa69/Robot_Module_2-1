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
    x1_1, y1_1 = box1[0]
    x2_1, y2_1 = box1[1]
    x1_2, y1_2 = box2[0]
    x2_2, y2_2 = box2[1]
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

# --- ฟังก์ชันใหม่ที่เพิ่มเข้ามา ---
def non_maximum_suppression_with_containment(boxes_with_scores, iou_threshold=0.3, containment_threshold=0.9):
    """
    NMS ที่ปรับปรุงใหม่เพื่อจัดการกับกรอบที่ซ้อนกันอยู่ข้างใน (Nested Boxes)
    โดยจะตัดกรอบออกหาก:
    1. มีค่า IoU สูงกว่า threshold (เหมือนเดิม)
    2. กรอบที่เล็กกว่าถูกล้อมรอบโดยกรอบที่ใหญ่กว่า เกินกว่า containment_threshold
    """
    if not boxes_with_scores:
        return []
    
    boxes_with_scores.sort(key=lambda x: x[2], reverse=True)
    
    selected_boxes = []
    while boxes_with_scores:
        current_box = boxes_with_scores.pop(0)
        selected_boxes.append(current_box)
        
        remaining_boxes = []
        for box in boxes_with_scores:
            iou = calculate_iou(current_box, box)
            
            # --- ตรรกะใหม่: ตรวจสอบการถูกล้อมรอบ ---
            is_contained = False
            # คำนวณพื้นที่ของแต่ละกรอบ
            area_current = (current_box[1][0] - current_box[0][0]) * (current_box[1][1] - current_box[0][1])
            area_box = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
            
            # หาพื้นที่ที่ซ้อนกัน (Intersection)
            x1_i = max(current_box[0][0], box[0][0])
            y1_i = max(current_box[0][1], box[0][1])
            x2_i = min(current_box[1][0], box[1][0])
            y2_i = min(current_box[1][1], box[1][1])
            intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
            
            # ถ้ากรอบ 'box' เล็กกว่า 'current_box' และพื้นที่ซ้อนทับกันเกือบเท่าพื้นที่ของ 'box'
            if area_box < area_current and area_box > 0:
                if (intersection_area / area_box) > containment_threshold:
                    is_contained = True

            # ถ้ากรอบ 'current_box' เล็กกว่า 'box' และพื้นที่ซ้อนทับกันเกือบเท่าพื้นที่ของ 'current_box'
            elif area_current < area_box and area_current > 0:
                if (intersection_area / area_current) > containment_threshold:
                    is_contained = True

            # --- เก็บกรอบไว้ถ้าไม่ทับซ้อน (IoU ต่ำ) และ ไม่ถูกล้อมรอบ ---
            if iou < iou_threshold and not is_contained:
                remaining_boxes.append(box)

        boxes_with_scores = remaining_boxes
        
    return selected_boxes


# ---------------------------------------------------
# ส่วนที่ 2: การทำงานหลัก (Main Program) - มีการแก้ไขเล็กน้อย
# ---------------------------------------------------

def main():
    """ฟังก์ชันหลักสำหรับเชื่อมต่อหุ่นยนต์และเริ่มการตรวจจับ"""
    
    TEMPLATE_FILES = [
        "image/template/template_night_pic1_x_557_y_266_w_107_h_275.jpg",
        "image/template/template_night_pic2_x_607_y_281_w_55_h_143.jpg",
        "image/template/template_night_pic4_x_318_y_146_w_17_h_38.jpg"
    ]
    MATCH_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.3
    # --- ค่าใหม่สำหรับควบคุมการตรวจจับกรอบซ้อน ---
    # หมายความว่า ถ้ากรอบเล็กมีพื้นที่ 90% อยู่ในกรอบใหญ่ ให้ถือว่าเป็นกรอบเดียวกัน
    CONTAINMENT_THRESHOLD = 0.9 
    
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
                    all_detections.append((top_left, bottom_right, confidence, i, colors[i]))

            # --- *** แก้ไขจุดนี้: เรียกใช้ฟังก์ชัน NMS ตัวใหม่ *** ---
            final_detections = non_maximum_suppression_with_containment(
                all_detections, 
                iou_threshold=IOU_THRESHOLD,
                containment_threshold=CONTAINMENT_THRESHOLD
            )

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for top_left, bottom_right, confidence, template_id, color in final_detections:
                cv2.rectangle(frame, top_left, bottom_right, color, 2)
                label = f"T{template_id+1}: {confidence:.2f}"
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