import cv2
import numpy as np
import time
import sys

# 0 goes to webcam
# VIDEO_SOURCE = 0
VIDEO_SOURCE = "herd.mp4"

BG_HISTORY = 500
BG_THRESHOLD = 16    
BG_DETECT_SHADOWS = True
MIN_CONTOUR_AREA = 800   
MAX_CONTOUR_AREA = 50000  
MORPH_KERNEL_SIZE = (5, 5)
MORPH_DILATE_ITERS = 3    
OUTLIER_THRESHOLD_FRACTION = 0.25
Kp = 0.005          
CENTER_DEADBAND = 40     

# display
BOX_COLOR       = (0, 255, 100)      
CENTROID_COLOR  = (0, 255, 255)      
GROUP_COLOR     = (255, 80, 0)       
OUTLIER_COLOR   = (0, 60, 255)       
TEXT_COLOR      = (240, 240, 240)    
SHADOW_COLOR    = (0, 0, 0)      

# helper functions
def draw_text_with_shadow(img, text, pos, scale=0.6, color=TEXT_COLOR,
                          thickness=1, shadow=SHADOW_COLOR):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, shadow, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def get_frame_diagonal(h, w):
    return np.sqrt(h ** 2 + w ** 2)

def compute_centroid(points):
    pts = np.array(points, dtype=np.float32)
    return tuple(np.mean(pts, axis=0).astype(int))

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def classify_control(error_x, deadband=CENTER_DEADBAND):
    if abs(error_x) <= deadband:
        return "CENTERED / TRACK"
    elif error_x < 0:
        return "STEER LEFT"
    else:
        return "STEER RIGHT"


# main
def main():
    source = VIDEO_SOURCE
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Accept integer (webcam index) or file path
        source = int(arg) if arg.isdigit() else arg

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        sys.exit(1)

    frame_h, frame_w = first_frame.shape[:2]
    frame_cx = frame_w // 2   
    frame_cy = frame_h // 2   
    diag = get_frame_diagonal(frame_h, frame_w)
    outlier_thresh = OUTLIER_THRESHOLD_FRACTION * diag

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_THRESHOLD,
        detectShadows=BG_DETECT_SHADOWS
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

    prev_time = time.time()
    fps = 0.0

    print("[INFO] FieldVision running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break


        # apply background subtractor - foreground mask (255 = moving, 0 = bg)
        fg_mask = bg_subtractor.apply(frame)

        # threshold to remove shadow pixels (127) — keep only strong foreground
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)

        fg_mask = cv2.dilate(fg_mask, kernel, iterations=MORPH_DILATE_ITERS)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [
            c for c in contours
            if MIN_CONTOUR_AREA <= cv2.contourArea(c) <= MAX_CONTOUR_AREA
        ]

        object_centers = []   # (cx, cy) for each detected object
        bounding_boxes  = []  # (x, y, w, h) for each detected object

        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            object_centers.append((cx, cy))
            bounding_boxes.append((x, y, w, h))

        object_count = len(object_centers)

        group_centroid = None
        outlier_flags  = [False] * object_count  

        if object_count > 0:
            group_centroid = compute_centroid(object_centers)

            if object_count > 1:
                for i, center in enumerate(object_centers):
                    dist = euclidean_distance(center, group_centroid)
                    if dist > outlier_thresh:
                        outlier_flags[i] = True

        if group_centroid is not None:
            error_x = group_centroid[0] - frame_cx
            steering_signal = Kp * error_x          # P-controller output
            control_label = classify_control(error_x)
        else:
            error_x = 0
            steering_signal = 0.0
            control_label = "SEARCHING"

        cv2.line(frame, (frame_cx, 0), (frame_cx, frame_h),
                 (80, 80, 80), 1, cv2.LINE_AA)

        for i, ((x, y, w, h), center) in enumerate(zip(bounding_boxes, object_centers)):
            is_outlier = outlier_flags[i]
            box_color = OUTLIER_COLOR if is_outlier else BOX_COLOR

            # bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # object centroid dot
            cv2.circle(frame, center, 5, CENTROID_COLOR, -1)

            # object ID label
            label = f"OBJ {i+1}"
            if is_outlier:
                label += " [OUTLIER]"
            draw_text_with_shadow(frame, label, (x, y - 8), scale=0.45,
                                  color=box_color)

        # draw group centroid
        if group_centroid is not None:
            cv2.circle(frame, group_centroid, 10, GROUP_COLOR, -1)
            cv2.circle(frame, group_centroid, 12, (255, 255, 255), 2)
            # Line from frame center to group centroid
            cv2.arrowedLine(frame, (frame_cx, frame_cy), group_centroid,
                            GROUP_COLOR, 2, tipLength=0.2)
            draw_text_with_shadow(frame, "GROUP", 
                                  (group_centroid[0] + 14, group_centroid[1] + 5),
                                  scale=0.45, color=GROUP_COLOR)


        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time

        ctrl_colors = {
            "STEER LEFT":       (0, 180, 255),
            "STEER RIGHT":      (0, 180, 255),
            "CENTERED / TRACK": (0, 255, 100),
            "SEARCHING":        (100, 100, 100),
        }
        ctrl_color = ctrl_colors.get(control_label, TEXT_COLOR)

        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (310, 160), (15, 15, 15), -1)
        cv2.addWeighted(hud_overlay, 0.55, frame, 0.45, 0, frame)

        draw_text_with_shadow(frame, f"FieldVision  |  Mobile Robotics",
                              (10, 22), scale=0.55, color=(200, 200, 200))
        cv2.line(frame, (10, 30), (300, 30), (60, 60, 60), 1)

        draw_text_with_shadow(frame, f"FPS:      {fps:.1f}",
                              (10, 52), scale=0.52)
        draw_text_with_shadow(frame, f"Objects:  {object_count}",
                              (10, 74), scale=0.52)
        draw_text_with_shadow(frame, f"Error X:  {error_x:+.0f} px",
                              (10, 96), scale=0.52)
        draw_text_with_shadow(frame, f"Steering: {steering_signal:+.3f}",
                              (10, 118), scale=0.52)
        draw_text_with_shadow(frame, f"Cmd: {control_label}",
                              (10, 142), scale=0.6, color=ctrl_color)

        if any(outlier_flags):
            cv2.rectangle(frame, (0, frame_h - 36), (frame_w, frame_h),
                          (0, 0, 180), -1)
            draw_text_with_shadow(frame, "⚠  OUTLIER DETECTED — OBJECT OUT OF GROUP",
                                  (10, frame_h - 12), scale=0.55,
                                  color=(255, 220, 80))

        mask_small = cv2.resize(fg_mask, (frame_w // 5, frame_h // 5))
        mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        h5, w5 = mask_bgr.shape[:2]
        frame[8: 8 + h5, frame_w - w5 - 8: frame_w - 8] = mask_bgr
        draw_text_with_shadow(frame, "FG mask",
                              (frame_w - w5 - 8, 8 + h5 + 14),
                              scale=0.38, color=(160, 160, 160))

        cv2.imshow("FieldVision — Multi-Object Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit requested.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] FieldVision closed.")


if __name__ == "__main__":
    main()
