import cv2
import glob
import os
import numpy as np

folder = "calibration_images"
left_files = sorted(glob.glob(os.path.join(folder, "left_*.png")))
right_files = sorted(glob.glob(os.path.join(folder, "right_*.png")))

left_stems = {os.path.basename(f).replace("left_", ""): f for f in left_files}
right_stems = {os.path.basename(f).replace("right_", ""): f for f in right_files}
common = sorted(set(left_stems) & set(right_stems))

print(f"Total pairs found: {len(common)}")

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def check_board_edges(gray, board_size=(7, 10)):
    # Optional: ensure corners are not touching the absolute edges
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not ret:
        return False
    
    # Get min/max x and y of the corners
    x_coords = corners[:, 0, 0]
    y_coords = corners[:, 0, 1]
    
    h, w = gray.shape
    margin = 15 # pixels
    
    if np.min(x_coords) < margin or np.max(x_coords) > w - margin:
        return False
    if np.min(y_coords) < margin or np.max(y_coords) > h - margin:
        return False
        
    return True

deleted_count = 0
laplacians = []

# First pass: calculate all laplacians to find a good dynamic threshold
for k in common:
    img_l = cv2.imread(left_stems[k], cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(right_stems[k], cv2.IMREAD_GRAYSCALE)
    
    if img_l is None or img_r is None:
        continue
        
    fm_l = variance_of_laplacian(img_l)
    fm_r = variance_of_laplacian(img_r)
    laplacians.append((fm_l, fm_r))

if len(laplacians) > 0:
    all_fms = [fm for pair in laplacians for fm in pair]
    median_fm = np.median(all_fms)
    
    # Threshold: anything less than 40% of the median sharpness in this dataset is probably very blurry
    # A generic fixed threshold is hard to set without knowing lighting, so dynamic is better.
    # But let's also set an absolute minimum for 720p webcams
    THRESHOLD = max(median_fm * 0.4, 50.0) 
    print(f"Dataset median sharpness: {median_fm:.2f}. Using dynamic blur threshold: {THRESHOLD:.2f}")
else:
    THRESHOLD = 50.0

for k in common:
    path_l = left_stems[k]
    path_r = right_stems[k]
    
    img_l = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)
    
    fm_l = variance_of_laplacian(img_l)
    fm_r = variance_of_laplacian(img_r)
    
    reason = None
    if fm_l < THRESHOLD or fm_r < THRESHOLD:
        reason = f"Blurry (Left: {fm_l:.1f}, Right: {fm_r:.1f} | Threshold: {THRESHOLD:.1f})"
    elif not check_board_edges(img_l) or not check_board_edges(img_r):
        reason = "Board too close to edge or undetectable"
        
    if reason:
        print(f"Removing pair {k}: {reason}")
        os.remove(path_l)
        os.remove(path_r)
        deleted_count += 1

print(f"\n[DONE] Removed {deleted_count} bad pairs.")
print(f"[REMAINING] {len(common) - deleted_count} high-quality pairs left for calibration.")
