import cv2
import time

def measure_fps(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cam {cam_id} failed.")
        return
        
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Warmup
    for _ in range(5):
        cap.grab()
        
    start = time.perf_counter()
    count = 30
    for _ in range(count):
        cap.grab()
    duration = time.perf_counter() - start
    
    fps = count / duration
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_str = "".join([chr((fcc >> 8 * i) & 0xFF) for i in range(4)]) if fcc != 0 else "None"
    
    print(f"Cam {cam_id}: {fps:.2f} FPS measured. Codec reports as: {fcc_str}")
    cap.release()

measure_fps(1)
measure_fps(0)
