# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import cv2
import numpy as np
from robomaster import robot
from robomaster import camera

class PinkMaskDetector:
    """Pink Mask Template Matching Detector สำหรับ RoboMaster"""
    
    def __init__(self, template_paths, threshold=0.7, iou_threshold=0.3):
        """
        Initialize detector with template images
        
        Args:
            template_paths: รายการ path ของ template images
            threshold: threshold สำหรับ template matching
            iou_threshold: threshold สำหรับ Non-Maximum Suppression
        """
        self.template_paths = template_paths
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.templates_masked = []
        self.templates_rgb = []
        self.colors = [(0, 255, 0), (255, 165, 0), (0, 0, 255)]  # เขียว, ส้ม, น้ำเงิน
        
        # โหลดและเตรียม templates
        self._load_templates()
    
    def create_pink_mask(self, img_rgb):
        """สร้าง mask สำหรับสีชมพู โดยใช้ HSV color space (เหมือนใน notebook)"""
        # แปลงเป็น HSV เพื่อจับสีชมพูได้ดีกว่า
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # ช่วงค่าต่ำสุด ของสีชมพู (ปรับจาก detect_object.ipynb)
        lower_pink = np.array([120, 10, 120])
        # ช่วงค่าสูงสุด ของสีชมพู
        upper_pink = np.array([170, 100, 200])
        
        # สร้าง mask
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # เพิ่มการ morphology เพื่อลด noise (จาก detect_object.ipynb)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((7, 7), np.uint8)
        # ปิดช่องว่างเล็กๆ
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _load_templates(self):
        """โหลดและเตรียม template images"""
        for template_path in self.template_paths:
            try:
                # โหลด template
                tmpl = cv2.imread(template_path)
                if tmpl is None:
                    print(f"Warning: Cannot load template {template_path}")
                    continue
                
                tmpl_rgb = cv2.cvtColor(tmpl, cv2.COLOR_BGR2RGB)
                self.templates_rgb.append(tmpl_rgb)
                
                # สร้าง pink mask สำหรับ template
                tmpl_pink_mask = self.create_pink_mask(tmpl_rgb)
                
                # แปลงเป็น grayscale และใช้ mask
                tmpl_gray = cv2.cvtColor(tmpl_rgb, cv2.COLOR_RGB2GRAY)
                tmpl_masked = cv2.bitwise_and(tmpl_gray, tmpl_gray, mask=tmpl_pink_mask)
                
                self.templates_masked.append(tmpl_masked)
                
                print(f"Loaded template: {template_path}")
                
            except Exception as e:
                print(f"Error loading template {template_path}: {e}")
    
    def match_template_masked(self, img_masked, tmpl_masked):
        """Template matching บน masked grayscale images (เหมือนใน notebook)"""
        result = cv2.matchTemplate(img_masked, tmpl_masked, cv2.TM_CCOEFF_NORMED)
        
        # หา location ที่ score >= threshold
        locations = np.where(result >= self.threshold)
        
        boxes = []
        h, w = tmpl_masked.shape
        
        for pt in zip(*locations[::-1]):  # แปลง (row, col) -> (x, y)
            boxes.append((pt, (pt[0]+w, pt[1]+h)))
        
        return boxes, result
    
    def calculate_iou(self, box1, box2):
        """คำนวณ Intersection over Union (IoU) ระหว่าง 2 กรอบ (เหมือนใน notebook)"""
        # box format: ((x1, y1), (x2, y2))
        x1_1, y1_1 = box1[0]
        x2_1, y2_1 = box1[1]
        x1_2, y1_2 = box2[0]
        x2_2, y2_2 = box2[1]
        
        # หาพื้นที่ทับซ้อน (intersection)
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # หาพื้นที่รวม (union)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def non_maximum_suppression(self, boxes_with_scores):
        """Non-Maximum Suppression เพื่อกำจัดกรอบที่ทับซ้อน (เหมือนใน notebook)"""
        if len(boxes_with_scores) == 0:
            return []
        
        # เรียงตาม confidence score (จากมากไปน้อย)
        boxes_with_scores.sort(key=lambda x: x[2], reverse=True)
        
        selected_boxes = []
        
        while len(boxes_with_scores) > 0:
            # เลือกกรอบที่มี confidence สูงสุด
            current_box = boxes_with_scores.pop(0)
            selected_boxes.append(current_box)
            
            # กำจัดกรอบที่มี IoU สูงเกินไปกับกรอบที่เลือก
            boxes_with_scores = [
                box for box in boxes_with_scores
                if self.calculate_iou(current_box[:2], box[:2]) < self.iou_threshold
            ]
        
        return selected_boxes
    
    def detect_objects(self, frame):
        """ตรวจจับวัตถุในเฟรม (หลักการเหมือนใน notebook)"""
        # แปลงจาก BGR เป็น RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # สร้าง pink mask สำหรับภาพหลัก
        main_pink_mask = self.create_pink_mask(frame_rgb)
        
        # แปลงภาพหลักเป็น grayscale และใช้ mask
        main_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        main_masked = cv2.bitwise_and(main_gray, main_gray, mask=main_pink_mask)
        
        # รวบรวมการตรวจจับทั้งหมด
        all_detections = []
        
        # ทำ matching สำหรับทุก template
        for i, tmpl_masked in enumerate(self.templates_masked):
            boxes, result = self.match_template_masked(main_masked, tmpl_masked)
            
            for box in boxes:
                top_left, bottom_right = box
                # คำนวณ confidence score
                confidence = result[top_left[1], top_left[0]] if top_left[1] < result.shape[0] and top_left[0] < result.shape[1] else 0
                
                # เก็บข้อมูล: (top_left, bottom_right, confidence, template_id, color)
                color = self.colors[i % len(self.colors)]
                all_detections.append((top_left, bottom_right, confidence, i+1, color))
        
        # ใช้ NMS กับการตรวจจับทั้งหมด
        final_detections = self.non_maximum_suppression(all_detections)
        
        return final_detections
    
    def draw_detections(self, frame, detections):
        """วาดกรอบการตรวจจับบนเฟรม"""
        result_frame = frame.copy()
        
        for detection in detections:
            top_left, bottom_right, confidence, template_id, color = detection
            
            # วาดกรอบ
            cv2.rectangle(result_frame, top_left, bottom_right, color, 3)
            
            # วาดข้อความ template ID
            cv2.putText(result_frame, f"T{template_id}", 
                       (top_left[0]+5, top_left[1]+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # วาด confidence score
            cv2.putText(result_frame, f"{confidence:.2f}", 
                    (top_left[0]+5, top_left[1]+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # วาดพื้นหลังสีดำให้ข้อความ
            cv2.rectangle(result_frame, (top_left[0]+2, top_left[1]+2), 
                        (top_left[0]+80, top_left[1]+45), (0, 0, 0), -1)
            
            # วาดข้อความใหม่ทับ
            cv2.putText(result_frame, f"T{template_id}", 
                    (top_left[0]+5, top_left[1]+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(result_frame, f"{confidence:.2f}", 
                    (top_left[0]+5, top_left[1]+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_frame

def main():
    """ฟังก์ชันหลักสำหรับรันการตรวจจับแบบ real-time"""
    
    # กำหนด path ของ template images (ปรับตามที่ตั้งไฟล์จริง)
    template_paths = [
        "image/template/template_pic1_x_573_y_276_w_115_h_312.jpg",
        "image/template/template_pic2_x_634_y_291_w_50_h_134.jpg", 
        "image/template/template_pic3_x_629_y_291_w_35_h_92.jpg"
    ]
    
    # สร้าง detector
    detector = PinkMaskDetector(
        template_paths=template_paths,
        threshold=0.7,           # threshold สำหรับ template matching
        iou_threshold=0.3        # threshold สำหรับ NMS
    )
    
    # เชื่อมต่อ RoboMaster
    print("Connecting to RoboMaster...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    # เริ่มการส่งวิดีโอจากกล้อง
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    
    print("Starting real-time detection...")
    print("Press 'q' to quit")
    
    try:
        while True:
            # รับเฟรมจากกล้อง
            frame = ep_camera.read_cv2_image(timeout=1.0)
            
            if frame is not None:
                # ตรวจจับวัตถุ
                detections = detector.detect_objects(frame)
                
                # วาดผลลัพธ์
                result_frame = detector.draw_detections(frame, detections)
                
                # แสดงจำนวนการตรวจจับ
                info_text = f"Detections: {len(detections)}"
                cv2.putText(result_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # แสดงผล
                cv2.imshow("RoboMaster Pink Detection", result_frame)
                
                # แสดงรายละเอียดการตรวจจับใน console
                if detections:
                    print(f"\nFrame detections: {len(detections)}")
                    for i, detection in enumerate(detections):
                        top_left, bottom_right, confidence, template_id, _ = detection
                        x, y = top_left
                        w = bottom_right[0] - top_left[0]
                        h = bottom_right[1] - top_left[1]
                        print(f"  Detection {i+1}: T{template_id} at ({x}, {y}) "
                            f"size {w}x{h} confidence {confidence:.3f}")
            
            # ตรวจสอบการกดปุ่ม 'q' เพื่อออก
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    
    finally:
        # ปิดการเชื่อมต่อ
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()
        ep_robot.close()
        print("Detection stopped and robot disconnected.")

# ฟังก์ชันสำหรับทดสอบ detector กับภาพเดียว
def test_detector_single_image(image_path, template_paths):
    """ทดสอบ detector กับภาพเดียว"""
    detector = PinkMaskDetector(template_paths)
    
    # โหลดภาพ
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return
    
    # ตรวจจับ
    detections = detector.detect_objects(image)
    
    # วาดผลลัพธ์
    result = detector.draw_detections(image, detections)
    
    # แสดงผล
    cv2.imshow("Test Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Found {len(detections)} detections")
    for i, detection in enumerate(detections):
        top_left, bottom_right, confidence, template_id, _ = detection
        print(f"Detection {i+1}: Template {template_id}, Confidence: {confidence:.3f}")

if __name__ == '__main__':
    # รันการตรวจจับแบบ real-time
    main()
    
    # หรือทดสอบกับภาพเดียว (uncommment บรรทัดข้างล่าง)
    # template_paths = ["template1.jpg", "template2.jpg", "template3.jpg"]
    # test_detector_single_image("test_image.jpg", template_paths)