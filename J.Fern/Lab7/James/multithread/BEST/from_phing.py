# File: best_multithread_singlefile_GIMBAL_FIXED.py
# Description: Fixed gimbal "drooping" issue by increasing PID gains and adding anti-windup.
#              Also fixed the ZeroDivisionError by checking if dt > 0.

import cv2
import numpy as np
import time
import threading
import os
from robomaster import robot
from robomaster import camera

# --- คลาสและฟังก์ชันต่างๆ (ไม่มีการเปลี่ยนแปลง) ---
class SimpleKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_value=0.0):
        self.Q, self.R, self.P, self.x = process_noise, measurement_noise, 1.0, initial_value
    def update(self, measurement):
        K = self.P / (self.P + self.R); self.x += K * (measurement - self.x); self.P = (1 - K) * self.P
    def predict(self): self.P += self.Q
    def get_state(self): return self.x

def create_pink_mask(img_rgb):
    hsv = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([128, 20, 100]), np.array([158, 130, 200]))
    mask = cv2.medianBlur(mask, 1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

def match_template_masked(img_masked, tmpl_masked, threshold=0.7):
    if tmpl_masked.shape[0]>img_masked.shape[0] or tmpl_masked.shape[1]>img_masked.shape[1]: return []
    result = cv2.matchTemplate(img_masked, tmpl_masked, cv2.TM_CCOEFF_NORMED)
    locs = np.where(result >= threshold)
    boxes, (h, w) = [], tmpl_masked.shape
    for pt in zip(*locs[::-1]): boxes.append(((pt[0], pt[1]), (pt[0]+w, pt[1]+h), result[pt[1], pt[0]]))
    return boxes

def calculate_iou(b1, b2):
    i_area=max(0,min(b1[1][0],b2[1][0])-max(b1[0][0],b2[0][0]))*max(0,min(b1[1][1],b2[1][1])-max(b1[0][1],b2[0][1]))
    u_area=(b1[1][0]-b1[0][0])*(b1[1][1]-b1[0][1])+(b2[1][0]-b2[0][0])*(b2[1][1]-b2[0][1])-i_area
    return i_area/u_area if u_area>0 else 0.0

def advanced_prioritized_nms(boxes, thresh=0.3):
    if not boxes: return []
    dets = sorted(boxes, key=lambda x: x[3]); final = []
    while dets:
        curr = dets.pop(0); final.append(curr)
        dets = [b for b in dets if calculate_iou(curr, b) <= thresh]
    return final

def prepare_templates(original_template_paths, processing_scale):
    output_dir = "image/template/processed"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    processed_paths = []
    for i, file_path in enumerate(original_template_paths):
        base_name = os.path.basename(file_path).split('.')[0]
        output_filename = os.path.join(output_dir, f"cached_{base_name}_{i+1}.png")
        processed_paths.append(output_filename)
        if os.path.exists(output_filename): continue
        print(f"Processing and caching new template: {file_path}...")
        tmpl = cv2.imread(file_path)
        if tmpl is None: raise FileNotFoundError(f"Original template not found: {file_path}")
        w, h = int(tmpl.shape[1] * processing_scale), int(tmpl.shape[0] * processing_scale)
        resized = cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=create_pink_mask(rgb))
        cv2.imwrite(output_filename, masked)
        print(f"  Saved cached template to {output_filename}")
    return processed_paths

# --- ส่วนของ Multithreading ---
latest_frame = None; processed_output = {"annotated_frame": None, "speed_x": 0.0, "speed_y": 0.0}
frame_lock, output_lock, stop_event = threading.Lock(), threading.Lock(), threading.Event()

# --- Threads (Capture, Processing) ---
def capture_thread_func(ep_camera):
    global latest_frame
    print("Capture thread started.")
    while not stop_event.is_set():
        try:
            frame = ep_camera.read_cv2_image(timeout=0.5)
            if frame is not None:
                with frame_lock: latest_frame = frame.copy()
        except Exception:
            time.sleep(0.1)
            continue
    print("Capture thread stopped.")

def processing_thread_func(templates_masked, params):
    global latest_frame, processed_output
    print("Processing thread started.")
    h_proc, center_x_orig, center_y_orig, p_scale = params["dims"]
    P_YAW, I_YAW, D_YAW = params["pid_yaw"]
    P_PITCH, I_PITCH, D_PITCH = params["pid_pitch"]
    K_X, K_Y = params["k_values"]; W_CM, H_CM = params["real_dims"]
    THRESH, IOU = params["detection"]; Y_ADJUSTS = params["y_adjustments"]
    
    p_err_x, p_err_y, acc_err_x, acc_err_y = 0,0,0,0
    p_time, tracking = time.time(), False
    kf_dist = SimpleKalmanFilter(process_noise=0.01, measurement_noise=25.0)
    
    INTEGRAL_LIMIT_Y = 200

    while not stop_event.is_set():
        with frame_lock:
            if latest_frame is None: time.sleep(0.01); continue
            frame = latest_frame.copy()
        
        found, err_x, err_y, raw_dist, speed_x, speed_y = False, 0, 0, 0.0, 0, 0
        
        proc = cv2.resize(frame, (int(params["PROCESSING_WIDTH"]), h_proc), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=create_pink_mask(rgb))
        dets = []; colors = [(0,255,0),(255,165,0),(0,0,255)]
        for i, tmpl in enumerate(templates_masked):
            for tl, br, conf in match_template_masked(masked, tmpl, THRESH):
                dets.append((tl, br, conf, i, colors[i]))
        final_dets = advanced_prioritized_nms(dets, IOU)

        if final_dets:
            tl_p, br_p, conf, tid, color = final_dets[0]
            y_adj = Y_ADJUSTS[tid]
            tl_adj = (int(tl_p[0]/p_scale), int(tl_p[1]/p_scale) + y_adj)
            br_adj = (int(br_p[0]/p_scale), int(br_p[1]/p_scale) + y_adj)
            cv2.rectangle(frame, tl_adj, br_adj, color, 3)
            label = f"T{tid+1} Conf:{conf:.2f}"
            cv2.putText(frame, label, (tl_adj[0], tl_adj[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            roi = frame[max(0,tl_adj[1]):min(frame.shape[0],br_adj[1]), max(0,tl_adj[0]):min(frame.shape[1],br_adj[0])]
            if roi.size > 0:
                contours, _ = cv2.findContours(create_pink_mask(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    real_tl = (tl_adj[0]+x, tl_adj[1]+y); real_br = (tl_adj[0]+x+w, tl_adj[1]+y+h)
                    cv2.rectangle(frame, real_tl, real_br, (255, 255, 0), 2)
                    center_x, center_y = tl_adj[0]+x+w/2, tl_adj[1]+y+h/2
                    err_x, err_y = center_x_orig - center_x, center_y_orig - center_y
                    found = True
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0,0,255),-1)
                    dist_w = (K_Y*H_CM)/w if w>0 else 0; dist_h = (K_X*W_CM)/h if h>0 else 0
                    raw_dist = (dist_w + dist_h) / 2 if dist_w > 0 and dist_h > 0 else max(dist_w, dist_h)
                    if raw_dist > 0: kf_dist.predict(); kf_dist.update(raw_dist)

        kalman_dist = kf_dist.get_state()
        if not found: kf_dist.predict()
        
        curr_time = time.time()
        dt = curr_time - p_time if p_time else 1/30.0

        if found:
            if not tracking: 
                D_x, D_y, acc_err_x, acc_err_y = 0,0,0,0
                tracking=True
            else:
                # --- [FIXED] Prevent ZeroDivisionError by checking if dt is positive ---
                if dt > 0:
                    D_x = D_YAW * ((err_x - p_err_x) / dt)
                    D_y = D_PITCH * ((err_y - p_err_y) / dt)
                else:
                    D_x, D_y = 0, 0 # If dt is zero, skip derivative calculation for this frame
                
                acc_err_x += err_x * dt
                acc_err_y += err_y * dt
            
            acc_err_y = np.clip(acc_err_y, -INTEGRAL_LIMIT_Y, INTEGRAL_LIMIT_Y)

            speed_x = (P_YAW*err_x) + D_x + (I_YAW*acc_err_x)
            speed_y = (P_PITCH*err_y) + D_y + (I_PITCH*acc_err_y)
            p_err_x, p_err_y = err_x, err_y
        else: 
            tracking,speed_x,speed_y = False,0,0
            p_err_x,p_err_y,acc_err_x,acc_err_y = 0,0,0,0
        
        p_time = curr_time

        cv2.putText(frame, f"e_x: {err_x:.2f}, e_y: {err_y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sp_x: {speed_x:.2f}, Sp_y: {speed_y:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Raw Dist: {raw_dist:.2f} cm", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Kalman Dist: {kalman_dist:.2f} cm", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.circle(frame, (int(center_x_orig), int(center_y_orig)), 5, (255,0,255), -1)

        with output_lock: 
            processed_output={"annotated_frame":frame, "speed_x":speed_x, "speed_y":speed_y}
            
    print("Processing thread stopped.")

# --- Main Thread: Controller & UI ---
def main():
    global processed_output
    params = {
        "PROCESSING_WIDTH": 640.0,
        "pid_yaw":   (-0.3, -0.01, -0.01),
        "pid_pitch": (-0.25, -0.15, -0.03),
        "k_values": (603.766, 393.264),
        "real_dims": (21.2, 13.2),
        "detection": (0.45, 0.1), 
        "y_adjustments": [0, 0, 0]
    }
    
    ORIGINAL_TEMPLATE_FILES = [
        "image/template/use/long_template_new1_pic3_x_327_y_344_w_157_h_345.jpg",
        "image/template/use/template_new2_pic2_x_580_y_291_w_115_h_235.jpg",
        "image/template/use/template_new1_pic3_x_607_y_286_w_70_h_142.jpg"
    ]

    print("Init robot..."); ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
    ep_camera, ep_gimbal = ep_robot.camera, ep_robot.gimbal
    print("Starting stream..."); ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)
    print("Recentering..."); ep_gimbal.recenter(pitch_speed=200, yaw_speed=200).wait_for_completed()
    time.sleep(0.5)
    print("Getting frame for scale..."); frame = ep_camera.read_cv2_image(timeout=5)
    if frame is None: print("Error: No frame."); ep_robot.close(); return
    h, w, _ = frame.shape
    scale = params["PROCESSING_WIDTH"] / w
    params["dims"] = (int(h*scale), w/2, h/2, scale)
    print("Preparing and loading templates...")
    try:
        processed_template_paths = prepare_templates(ORIGINAL_TEMPLATE_FILES, scale)
        tmpls = []
        for f in processed_template_paths:
            tmpl = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if tmpl is None: raise FileNotFoundError(f"Failed to load processed template: {f}")
            tmpls.append(tmpl)
    except Exception as e:
        print(f"Error during template preparation: {e}"); ep_robot.close(); return
    print("Templates are ready.")
    print("Starting threads..."); cap_t = threading.Thread(target=capture_thread_func, args=(ep_camera,))
    proc_t = threading.Thread(target=processing_thread_func, args=(tmpls, params))
    cap_t.start(); proc_t.start()
    print("System running. Press 'q' to exit.")
    try:
        while True:
            with output_lock: 
                s_x, s_y, ann_frame = processed_output["speed_x"], processed_output["speed_y"], processed_output["annotated_frame"]
            ep_gimbal.drive_speed(pitch_speed=-s_y, yaw_speed=s_x)
            if ann_frame is not None: 
                cv2.imshow("Robomaster Detection (Multithreaded)", ann_frame)
            if cv2.waitKey(1)&0xFF==ord('q') or not proc_t.is_alive(): 
                break
            time.sleep(0.01)
    finally:
        print("Stopping..."); stop_event.set(); cap_t.join(); proc_t.join()
        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0); cv2.destroyAllWindows()
        ep_camera.stop_video_stream(); ep_robot.close(); print("Terminated.")

if __name__ == '__main__':
    main()