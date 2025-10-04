# zone_realtimescan.py
import cv2
import numpy as np
import sys
import robomaster
from robomaster import robot
from robomaster import camera as r_camera # Import camera with an alias to avoid confusion
import time
import math
import matplotlib.pyplot as plt
import subprocess

# =====================================================
# Night-robust color pre-processing (From zone.py)
# =====================================================
def apply_awb(bgr):
    """Applies a learning-based or simple auto white balance."""
    if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createLearningBasedWB"):
        wb = cv2.xphoto.createLearningBasedWB()
        try:
            wb.setSaturationThreshold(0.99)
        except Exception: pass
        return wb.balanceWhite(bgr)
    elif hasattr(cv2, "xphoto"):
        wb = cv2.xphoto.createSimpleWB()
        try:
            wb.setP(0.5)
        except Exception: pass
        return wb.balanceWhite(bgr)
    else:
        return bgr

def retinex_msrcp(bgr, sigmas=(15, 80, 250), eps=1e-6):
    """Implements the Multi-Scale Retinex with Color Restoration."""
    img = bgr.astype(np.float32) + 1.0
    intensity = img.mean(axis=2)
    log_I = np.log(intensity + eps)
    msr = np.zeros_like(intensity, dtype=np.float32)
    for s in sigmas:
        blur = cv2.GaussianBlur(intensity, (0, 0), s)
        msr += (log_I - np.log(blur + eps))
    msr /= float(len(sigmas))
    msr -= msr.min()
    msr /= (msr.max() + eps)
    msr = (msr * 255.0).astype(np.float32)
    scale = (msr + 1.0) / (intensity + eps)
    out = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    return out

def night_enhance_pipeline(bgr):
    """Full pipeline for enhancing low-light images."""
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    awb = apply_awb(den)
    ret = retinex_msrcp(awb)
    return ret

# =====================================================
# ObjectTracker (From zone.py, no changes)
# =====================================================
class ObjectTracker:
    def __init__(self):
        print("🖼️  ObjectTracker initialized.")
        # Templates are not used in this detection logic but kept for context
        pass

    def _get_angle(self, pt1, pt2, pt0):
        """Calculates the angle between three points."""
        dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        magnitude2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if magnitude1 * magnitude2 == 0: return 0
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
        return math.degrees(angle_rad)

    def _get_raw_detections(self, frame):
        """Performs color and shape detection on a given frame."""
        enhanced = night_enhance_pipeline(frame)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        v_eq = clahe.apply(v)
        normalized_hsv = cv2.merge([h, s, v_eq])
        
        color_ranges = {
            'Red': [np.array([0, 80, 40]), np.array([10, 255, 255]), np.array([170, 80, 40]), np.array([180, 255, 255])],
            'Yellow': [np.array([20, 60, 40]), np.array([35, 255, 255])],
            'Green': [np.array([35, 40, 30]), np.array([85, 255, 255])],
            'Blue': [np.array([90, 40, 30]), np.array([130, 255, 255])]
        }

        masks = {}
        lower1, upper1, lower2, upper2 = color_ranges['Red']
        masks['Red'] = cv2.inRange(normalized_hsv, lower1, upper1) | cv2.inRange(normalized_hsv, lower2, upper2)
        for name in ['Yellow', 'Green', 'Blue']:
            lower, upper = color_ranges[name]
            masks[name] = cv2.inRange(normalized_hsv, lower, upper)
            
        combined_mask = masks['Red'] | masks['Yellow'] | masks['Green'] | masks['Blue']
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1), cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500: continue
            
            contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            max_mean, found_color = 0, "Unknown"
            for color_name, m in masks.items():
                mean_val = cv2.mean(m, mask=contour_mask)[0]
                if mean_val > max_mean:
                    max_mean, found_color = mean_val, color_name
            if max_mean <= 20: continue
            
            shape = "Uncertain"
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            
            if circularity > 0.84:
                shape = "Circle"
            elif len(approx) == 4:
                points = [tuple(p[0]) for p in approx]
                angles = [self._get_angle(points[(i - 1) % 4], points[(i + 1) % 4], p) for i, p in enumerate(points)]
                if all(75 <= ang <= 105 for ang in angles):
                    _, (w, h), _ = cv2.minAreaRect(cnt)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    shape = "Square" if 0.90 <= aspect_ratio <= 1.10 else "Rectangle_H" if w < h else "Rectangle_V"
                    
            raw_detections.append({'contour': cnt, 'shape': shape, 'color': found_color})
            
        return raw_detections

# =====================================================
# Interaction and Drawing (From zone.py)
# =====================================================
def get_target_choice():
    """Prompts the user to define the target shape and color."""
    VALID_SHAPES = ["Circle", "Square", "Rectangle_H", "Rectangle_V"]
    VALID_COLORS = ["Red", "Yellow", "Green", "Blue"]
    print("\n--- 🎯 Define Target Characteristics ---")
    while True:
        shape = input(f"Select Shape ({'/'.join(VALID_SHAPES)}): ").strip().title()
        if shape in VALID_SHAPES: break
        print("⚠️ Invalid shape, please try again.")
    while True:
        color = input(f"Select Color ({'/'.join(VALID_COLORS)}): ").strip().title()
        if color in VALID_COLORS: break
        print("⚠️ Invalid color, please try again.")
    print(f"✅ Target defined: {color} {shape}.")
    return shape, color

def scan_and_process_maze_wall(frame, tracker, target_shape, target_color):
    """Processes a single frame to find and classify objects."""
    if frame is None:
        print("❌ Cannot process a null frame.")
        return None, "", "", "", []

    # Apply the same Region of Interest (ROI) as in the original zone.py
    # NOTE: This ROI is for 720p (1280x720). It might need adjustment for 540p (960x540).
    # Original 720p ROI: frame[352:352+360, 14:14+1215]
    # Scaled for 540p (540/720 = 0.75):
    # y_start = int(352 * 0.75) = 264
    # height = int(360 * 0.75) = 270
    # x_start = int(14 * 0.75) = 10
    # width = int(1215 * 0.75) = 911
    # We will apply this scaled ROI.
    roi_frame = frame[264:264+270, 10:10+911]
    
    print("🧠 Processing captured frame...")
    output_image = roi_frame.copy()
    height, width, _ = roi_frame.shape

    # Define dividers based on the new ROI width (911px)
    DIVIDER_X1, DIVIDER_X2 = 300, 600
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(output_image, (DIVIDER_X1, 0), (DIVIDER_X1, height), (255, 255, 0), 2)
    cv2.line(output_image, (DIVIDER_X2, 0), (DIVIDER_X2, height), (255, 255, 0), 2)
    cv2.putText(output_image, "LEFT", (DIVIDER_X1 // 2 - 50, 40), font, 1, (255, 255, 0), 2)
    cv2.putText(output_image, "CENTER", ((DIVIDER_X1 + DIVIDER_X2) // 2 - 70, 40), font, 1, (255, 255, 0), 2)
    cv2.putText(output_image, "RIGHT", ((DIVIDER_X2 + width) // 2 - 60, 40), font, 1, (255, 255, 0), 2)

    detections = tracker._get_raw_detections(roi_frame)
    
    summary_lines = {'Left': [], 'Center': [], 'Right': []}
    detailed_results = []
    object_id_counter = 1

    for det in detections:
        shape, color, contour = det['shape'], det['color'], det['contour']
        x, y, w, h = cv2.boundingRect(contour)
        obj_start_x, obj_end_x = x, x + w

        zone_label = "Unknown"
        if obj_end_x < DIVIDER_X1: zone_label = "Left"
        elif obj_start_x >= DIVIDER_X2: zone_label = "Right"
        elif obj_start_x >= DIVIDER_X1 and obj_end_x < DIVIDER_X2: zone_label = "Center"
        elif obj_start_x < DIVIDER_X1 and obj_end_x >= DIVIDER_X1: zone_label = "Corner (L/C)"
        elif obj_start_x < DIVIDER_X2 and obj_end_x >= DIVIDER_X2: zone_label = "Corner (C/R)"

        is_target = (shape == target_shape and color == target_color)
        is_uncertain = (shape == "Uncertain")
        
        if is_target:
            box_color, thickness, summary_prefix = (0, 0, 255), 4, "TARGET FOUND"
        elif is_uncertain:
            box_color, thickness, summary_prefix = (0, 255, 255), 2, "Uncertain object"
        else:
            box_color, thickness, summary_prefix = (0, 255, 0), 2, "Found object"
        
        summary_text = f"  - {summary_prefix}: {color} {shape}"

        if "Left" in zone_label or "(L/C)" in zone_label:
            summary_lines['Left'].append(f"{summary_text} [ID: {object_id_counter}]")
        if "Center" in zone_label or "(L/C)" in zone_label or "(C/R)" in zone_label:
            summary_lines['Center'].append(f"{summary_text} [ID: {object_id_counter}]")
        if "Right" in zone_label or "(C/R)" in zone_label:
            summary_lines['Right'].append(f"{summary_text} [ID: {object_id_counter}]")
            
        detailed_results.append({
            "id": object_id_counter, "color": color, "shape": shape,
            "zone": zone_label, "is_target": is_target
        })

        label_on_box = str(object_id_counter)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), box_color, thickness)
        cv2.putText(output_image, label_on_box, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        
        object_id_counter += 1

    final_left_summary = f"--- Left Zone Detections ---\n" + ('\n'.join(summary_lines['Left']) if summary_lines['Left'] else "  - No objects found.")
    final_center_summary = f"--- Center Zone Detections ---\n" + ('\n'.join(summary_lines['Center']) if summary_lines['Center'] else "  - No objects found.")
    final_right_summary = f"--- Right Zone Detections ---\n" + ('\n'.join(summary_lines['Right']) if summary_lines['Right'] else "  - No objects found.")

    return output_image, final_left_summary, final_center_summary, final_right_summary, detailed_results

# =====================================================
# Main Execution Block
# =====================================================
if __name__ == '__main__':
    # --- Library installation check ---
    try:
        import scipy
    except ImportError:
        print("Installing required library: scipy")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        print("Installation successful!")
    try:
        _ = cv2.xphoto
    except Exception:
        print("🔎 Tip: For best AWB, install opencv-contrib-python (`pip install opencv-contrib-python`)")

    # --- Setup ---
    target_shape, target_color = get_target_choice()
    tracker = ObjectTracker()
    ep_robot = robot.Robot()
    
    try:
        # --- Robot Connection and Camera Start ---
        print("\n🤖 Connecting to Robomaster robot...")
        ep_robot.initialize(conn_type="ap")
        print("✅ Connection successful!")
        
        print("📷 Starting camera stream (540p)...")
        ep_robot.camera.start_video_stream(display=False, resolution=r_camera.STREAM_540P)
        print("✅ Camera stream started.")

        print("\n--- Live Camera Feed ---")
        print("Press 's' in the 'Live Feed' window to scan the current frame.")
        print("Press 'q' in the 'Live Feed' window to quit.")
        
        # --- Main Loop for Real-time Display and On-demand Scan ---
        while True:
            # Continuously read frames from the camera
            frame = ep_robot.camera.read_cv2_image(timeout=5)
            
            if frame is None:
                print("Waiting for frame...")
                time.sleep(0.1)
                continue

            # Display the live feed
            cv2.imshow("Live Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            # Quit condition
            if key == ord('q'):
                print("'q' pressed. Shutting down.")
                break

            # Scan condition
            if key == ord('s'):
                print("\n's' pressed. Initiating scan...")
                
                # --- Run detection on the current frame ---
                result_image, left_summary, center_summary, right_summary, detailed_results = scan_and_process_maze_wall(
                    frame, tracker, target_shape, target_color
                )
                
                print("\n--- SCAN COMPLETE ---")
                print(left_summary)
                print(center_summary)
                print(right_summary)
                
                print("\nDisplaying result image with Matplotlib...")
                if result_image is not None:
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    # --- Create and display the result plot with legend ---
                    fig = plt.figure(figsize=(12, 10))
                    plt.imshow(result_image_rgb)
                    plt.title("Maze Wall Scan Analysis Result (3-Zone)")
                    plt.axis('off')

                    if detailed_results:
                        legend_lines = ["--- Object Details Legend ---"]
                        for obj in detailed_results:
                            target_str = " (TARGET!)" if obj['is_target'] else ""
                            line = f"ID {obj['id']}: {obj['color']} {obj['shape']} in Zone [{obj['zone']}] {target_str}"
                            legend_lines.append(line)
                        legend_text = "\n".join(legend_lines)
                    else:
                        legend_text = "--- No objects detected ---"

                    plt.subplots_adjust(bottom=0.25)
                    plt.figtext(0.5, 0.01, legend_text, 
                                ha="center", va="bottom", 
                                fontsize=10, 
                                bbox={"facecolor":"white", "alpha":0.8, "pad":5})

                    plt.show() # This will block until the user closes the plot window
                    print("Result window closed. Returning to live feed.")
                else:
                    print("❌ Scan resulted in a null image, not displaying.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        # --- Cleanup ---
        print("\n🔌 Closing robot connection...")
        try:
            cv2.destroyAllWindows()
            ep_robot.camera.stop_video_stream()
            ep_robot.close()
            print("✅ Robot connection closed successfully.")
        except Exception as e:
            print(f"   (Note) Error during cleanup, but program is terminating: {e}")