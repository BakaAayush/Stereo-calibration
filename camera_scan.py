"""
camera_scan.py — Quick utility to find all available camera IDs on this machine.
Run: python camera_scan.py
"""
import cv2

print("Scanning camera indices 0 through 9 ...")
print("=" * 45)
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        ret, _ = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"  ID {i:>2} : FOUND   {w}x{h} @ {fps}fps   frame_ok={ret}")
    else:
        print(f"  ID {i:>2} : ---")
print("=" * 45)
print("Plug/unplug cameras and re-run to see changes.")
