import cv2
import numpy as np
import os
import json
from serial.tools import list_ports
from pydobotplus import Dobot
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 

# --- Dobot Setup ---
# Automatically find and connect to the last detected port
try:
    port = list_ports.comports()[-1].device
    print(f"Connecting to Dobot on port: {port}")
    device = Dobot(port=port)
    device.speed(velocity=100, acceleration=100)  # Set speed once
except IndexError:
    print("Error: No serial ports found. Ensure Dobot is connected.")
    device = None
except Exception as e:
    print(f"Error connecting to Dobot: {e}")
    device = None

# --- Configuration ---
TRANSFORM_FILE = 'data/transform.npy'
CROP_WARP_DATA_FILE = 'data/crop_warp_transform.json'
SOLVED_MAZE_IMAGE = 'images/maze_solution.png'
OUTPUT_WAYPOINTS_FILE = 'data/dobot_waypoints.npy'

# Z_HEIGHT_PATH should be the height where the tool touches the surface (e.g., -40mm)
Z_HEIGHT_PATH = -40
# Z_HEIGHT_LIFT should be a safe height above the surface (e.g., 0mm)
Z_HEIGHT_LIFT = 0

# Pixel threshold for blue color in BGR (The path color)
LOWER_BLUE = np.array([100, 50, 0])
UPPER_BLUE = np.array([255, 150, 100])

# --- Helper Functions (Start/End Detection) ---

def get_center(mask):
    """ Finds the center (x, y) of the largest contour in a binary mask. """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 5: 
        return None
    
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def detect_start_end_points(image):
    """ Detects the start (red) and end (green) points in the image. """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red mask (two hue ranges for wrapping around 180)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Green mask
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    start_pt = get_center(mask_red)
    end_pt = get_center(mask_green)

    if start_pt is None or end_pt is None:
        print("RuntimeError: Could not detect start (red) or end (green) dot.")
        return None, None

    return start_pt, end_pt

# --- Helper Functions (Transformations & Pathing) ---

def order_path_pixels(pixel_coords, start_point=None):
    """ Orders pixel coordinates into a continuous path starting nearest to start_point. """
    if not pixel_coords:
        return []

    remaining_points = set(pixel_coords)
    ordered_path = []

    # Determine starting point by finding the closest path pixel to the target start_point
    if start_point:
        path_array = np.array(list(remaining_points))
        start_point_array = np.array([start_point])
        closest_idx = np.argmin(cdist(path_array, start_point_array))
        current_point = tuple(path_array[closest_idx])
    else:
        current_point = min(remaining_points, key=lambda p: (p[1], p[0]))

    ordered_path.append(current_point)
    remaining_points.remove(current_point)

    MAX_DIST_SQ_CHECK = 8 

    while remaining_points:
        last_x, last_y = current_point
        best_next = None
        min_dist_sq = float('inf')

        search_candidates = list(remaining_points)
        search_candidates.sort(key=lambda p: (p[0] - last_x)**2 + (p[1] - last_y)**2)

        for next_point in search_candidates[:min(50, len(search_candidates))]: 
            dist_sq = (next_point[0] - last_x)**2 + (next_point[1] - last_y)**2
            
            if dist_sq <= MAX_DIST_SQ_CHECK:
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_next = next_point

        if best_next is None:
            if search_candidates:
                closest = search_candidates[0]
                dist_sq_closest = (closest[0] - last_x)**2 + (closest[1] - last_y)**2
                if dist_sq_closest < 2500: 
                    best_next = closest
                else:
                    print(f"Warning: Path break detected. Closest point too far ({np.sqrt(dist_sq_closest):.1f}px). Stopping path search.")
                    break
            else:
                break 

        current_point = best_next
        ordered_path.append(current_point)
        remaining_points.remove(current_point)

    return ordered_path

def transform_point_to_original(point, transform_json="data/crop_warp_transform.json"):
    """
    Correctly undo resize, warp, crop, and rotation in the right order.
    """
    with open(transform_json, 'r') as f:
        transform_data = json.load(f)

    x, y = point

    # 1️⃣ Undo resize
    if transform_data.get("resize_applied", False):
        scale = transform_data.get("resize_scale", 1.0)
        if scale != 0:
            x /= scale
            y /= scale

    # 2️⃣ Undo warp (if applied)
    if transform_data.get("is_warped", False) and transform_data.get("warp_matrix") is not None:
        M = np.array(transform_data["warp_matrix"], dtype=np.float32)
        M_inv = np.linalg.inv(M)
        pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        unwarped = cv2.perspectiveTransform(pts, M_inv)
        x, y = unwarped[0, 0]

    # 3️⃣ Undo crop (crop box is defined in rotated coordinates)
    crop_box = transform_data.get("crop_box", [0, 0, 0, 0])
    crop_x, crop_y = crop_box[:2]
    x += crop_x
    y += crop_y

    # 4️⃣ Undo rotation last (because crop_box is in rotated frame)
    rotation_matrix = transform_data.get("rotation_matrix", None)
    if rotation_matrix is not None:
        M = np.array(rotation_matrix, dtype=np.float32)
        M_affine = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M_affine)
        pt = np.array([x, y, 1], dtype=np.float32)
        original_pt = M_inv @ pt
        x, y = original_pt[0], original_pt[1]

    return (float(x), float(y))


def visualize_point_mapping(processed_img_path, original_img_path, point_processed, transform_json):
    """Visualize where a processed-image point maps onto the original image."""
    # Load images
    proc_img = cv2.imread(processed_img_path)
    orig_img = cv2.imread(original_img_path)
    if proc_img is None or orig_img is None:
        raise FileNotFoundError("Could not load one or both images.")

    # Transform point back to original
    point_orig = transform_point_to_original(point_processed, transform_json)

    # Draw on both images
    proc_vis = proc_img.copy()
    cv2.circle(proc_vis, tuple(map(int, point_processed)), 8, (0, 0, 255), -1)
    cv2.putText(proc_vis, "Processed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    orig_vis = orig_img.copy()
    cv2.circle(orig_vis, tuple(map(int, point_orig)), 10, (0, 255, 0), -1)
    cv2.putText(orig_vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show side by side
    combined = np.hstack([
        cv2.resize(proc_vis, (orig_vis.shape[1] // 2, orig_vis.shape[0] // 2)),
        cv2.resize(orig_vis, (orig_vis.shape[1] // 2, orig_vis.shape[0] // 2))
    ])

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title(f"Point mapping: Processed → Original\nProcessed {point_processed} → Original {point_orig}")
    plt.axis("off")
    plt.show()

    print(f"🟢 Mapped processed point {point_processed} → original point {point_orig}")

def apply_homography(x_orig, y_orig, h_matrix, z_height):
    """ STEP 2: Applies the Homography to original camera coordinates to get Dobot coordinates. """
    H = h_matrix
    cam_coords = np.array([[x_orig, y_orig, 1]], dtype=np.float32).T
    world_coords = np.dot(H, cam_coords)
    world_coords /= world_coords[2, 0]
    xw, yw = world_coords[0, 0], world_coords[1, 0]
    return (xw, yw, z_height)

def thinning(image_binary):
    """ Zhang-Suen thinning algorithm for skeletonization """
    img = image_binary.copy()
    img[img == 255] = 1
    rows, cols = img.shape
    changed = True

    while changed:
        changed = False
        for iteration in range(2):
            image_copy = img.copy()
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    P2, P3, P4 = img[r-1, c], img[r-1, c+1], img[r, c+1]
                    P5, P6, P7 = img[r+1, c+1], img[r+1, c], img[r+1, c-1]
                    P8, P9 = img[r, c-1], img[r-1, c-1]
                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9]

                    transitions = 0
                    for i in range(8):
                        if neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1:
                            transitions += 1
                    sum_neighbors = sum(neighbors)

                    if img[r, c] == 1 and 2 <= sum_neighbors <= 6 and transitions == 1:
                        if iteration == 0:
                            A = P2 * P4 * P6 == 0
                            B = P4 * P6 * P8 == 0
                            if A and B:
                                image_copy[r, c] = 0
                                changed = True
                        else:
                            A = P2 * P4 * P8 == 0
                            B = P2 * P6 * P8 == 0
                            if A and B:
                                image_copy[r, c] = 0
                                changed = True
            img = image_copy
            if not changed:
                break

    img[img == 1] = 255
    return img

def full_pixel_to_dobot_transform(x_final, y_final, h_matrix, transform_data, z_height):
    """ Combines inverse preprocessing (crop/warp) and homography to get final Dobot coords. """
    # Step 1: Reverse crop/warp to get original camera pixel coordinates
    x_orig, y_orig = transform_point_to_original((x_final, y_final), transform_data)
    
    # Step 2: Apply Homography to get Dobot world coordinates
    dobot_wp = apply_homography(x_orig, y_orig, h_matrix, z_height)
    
    return dobot_wp

def generate_dobot_waypoints_with_inverse(h_matrix, solved_image_path, transform_json, z_height, start_color="red"):
    """ 
    Generates optimized Dobot waypoints.
    Can start from either the red or green dot depending on `start_color`.
    Uses HSV-based blue path detection for robustness.
    """
    print(f"Loading solved image: {solved_image_path}")
    img = cv2.imread(solved_image_path)
    if img is None:
        print(f"❌ Error: Could not load image at {solved_image_path}")
        return []

    # --- STEP 1: Detect Start & End Points ---
    red_pt, green_pt = detect_start_end_points(img)
    if red_pt is None or green_pt is None:
        print("❌ Could not detect red or green dots. Aborting.")
        return []

    if start_color.lower() == "red":
        start_pt_final = red_pt
        print("🟥 Starting from RED dot → ending at GREEN dot.")
    elif start_color.lower() == "green":
        start_pt_final = green_pt
        print("🟩 Starting from GREEN dot → ending at RED dot.")
    else:
        print("⚠️ Invalid start color input. Defaulting to RED.")
        start_pt_final = red_pt

    # --- STEP 2: Detect Blue Path (in HSV) ---
    print("🔹 Detecting blue path using HSV color space...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    debug_mask_path = "images/debug_blue_mask.png"
    cv2.imwrite(debug_mask_path, blue_mask)
    blue_pixel_count = np.count_nonzero(blue_mask)
    print(f"Detected {blue_pixel_count} blue pixels (saved mask → {debug_mask_path})")

    if blue_pixel_count == 0:
        print("❌ Error: No blue pixels detected. Check HSV thresholds or image path.")
        return []

    # --- STEP 3: Skeletonize (Thinning) ---
    print("🧠 Applying skeletonization to thin the path...")
    thinned_mask = thinning(blue_mask)
    blue_pixels_y, blue_pixels_x = np.where(thinned_mask > 0)
    pixel_coords_raw = list(zip(blue_pixels_x, blue_pixels_y))

    if not pixel_coords_raw:
        print("❌ Error: No thinned path detected after skeletonization.")
        return []

    # --- STEP 4: Order the Path ---
    print("🧭 Ordering path pixels using nearest-neighbor traversal...")
    ordered_path_pixels = order_path_pixels(pixel_coords_raw, start_point=start_pt_final)
    
    if not ordered_path_pixels:
        print("❌ Error: Could not determine a continuous path.")
        return []

    # --- STEP 4.5: Direction enforcement based on user input ---
    if start_color.lower() == "green":
        print("🟩 User selected GREEN start → Reversing path to go GREEN → RED.")
        ordered_path_pixels.reverse()
    else:
        print("🟥 User selected RED start → Keeping path direction RED → GREEN.")


    # --- STEP 5: Sample & Transform to World Coordinates ---
    sampling_rate = 2
    sampled_coords_final = ordered_path_pixels[::sampling_rate]
    print(f"📏 Sampling path every {sampling_rate} pixels (Total samples: {len(sampled_coords_final)})")

    real_world_waypoints = []
    for x_final, y_final in sampled_coords_final:
        dobot_wp = full_pixel_to_dobot_transform(x_final, y_final, h_matrix, transform_json, z_height)
        real_world_waypoints.append(dobot_wp)

    # --- STEP 6: Optimize Waypoints ---
    optimized_waypoints = []
    if real_world_waypoints:
        optimized_waypoints.append(real_world_waypoints[0])
        min_distance_sq = 25
        for i in range(1, len(real_world_waypoints)):
            current = np.array(real_world_waypoints[i][:2])
            last = np.array(optimized_waypoints[-1][:2])
            distance_sq = np.sum((current - last) ** 2)
            if distance_sq >= min_distance_sq:
                optimized_waypoints.append(real_world_waypoints[i])

    print(f"✅ Thinned pixels: {len(ordered_path_pixels)}")
    print(f"✅ Optimized waypoints: {len(optimized_waypoints)}")

    if start_color.lower() == "green":
        print("➡️ Path direction: GREEN → RED")
        optimized_waypoints.reverse()
    else:
        print("➡️ Path direction: RED → GREEN")

    return optimized_waypoints

def execute_dobot_path(waypoints):
    """ Executes the sequence of waypoints """
    print("\nWaypoints to execute:")

    global device
    if not device:
        print("Cannot execute path: Dobot device not connected.")
        return
    if not waypoints:
        print("Error: Waypoints list is empty.")
        return

    try:
        print(f"Starting path execution of {len(waypoints)} waypoints...")
        x_start, y_start, z_path = waypoints[0]
        z_lift = Z_HEIGHT_LIFT

        # 1. Approach start at safe height
        print(f"1. Approaching start at lift height: ({x_start:.2f}, {y_start:.2f}, {z_lift:.2f})")
        device.move_to(x=x_start, y=y_start, z=z_lift, r=0, wait=True)

        # 2. Plunge to path height
        print(f"2. Plunging to path height: ({x_start:.2f}, {y_start:.2f}, {z_path:.2f})")
        device.move_to(x=x_start, y=y_start, z=z_path, r=0, wait=True)

        # 3. Follow continuous path
        print(f"3. Drawing continuous path of {len(waypoints)-1} segments...")
        for x, y, z in waypoints[1:]:
            device.move_to(x=x, y=y, z=z, r=0, wait=True)

        # 4. Lift-off
        x_end, y_end, _ = waypoints[-1]
        print(f"4. Lifting tool at end point: ({x_end:.2f}, {y_end:.2f}, {z_lift:.2f})")
        device.move_to(x=x_end, y=y_end, z=z_lift, r=0, wait=True)

        # 5. Retreat
        print("5. Retreating to safe position (e.g., origin [200, 0, 0, 0])")
        device.move_to(x=200, y=0, z=0, r=0, wait=True)

        print("✅ Path execution complete!")
        device.home()
    except Exception as e:
        print(f"An error occurred during Dobot execution: {e}")
    finally:
        if device is not None:
            print("Closing Dobot connection.")
            device.close()

def main():
    # with open("data/maze_metadata.json", "r") as f:
    #     meta = json.load(f)

    # Ask user for start point color
    start_color = input("Enter start color (red/green): ").strip().lower()
    if start_color not in ["red", "green"]:
        print("⚠️ Invalid input. Defaulting to RED.")
        start_color = "red"

    if not os.path.exists(TRANSFORM_FILE) or not os.path.exists(CROP_WARP_DATA_FILE):
        print(f"Error: Required files not found.")
        print(f"Ensure '{TRANSFORM_FILE}' (Homography) and '{CROP_WARP_DATA_FILE}' (Crop/Warp Data) exist.")
    elif device is not None:
        H = np.load(TRANSFORM_FILE)

        try:
            final_waypoints = generate_dobot_waypoints_with_inverse(
                H, SOLVED_MAZE_IMAGE, CROP_WARP_DATA_FILE, Z_HEIGHT_PATH, start_color=start_color
            )
        except Exception as e:
            print(f"An error occurred during waypoint generation: {e}")
            final_waypoints = []

        if final_waypoints:
            np.save(OUTPUT_WAYPOINTS_FILE, final_waypoints)
            print(f"\n✅ Waypoints successfully saved to {OUTPUT_WAYPOINTS_FILE}")
            execute_dobot_path(final_waypoints)
        else:
            print("\n🚫 No valid waypoints generated.")
    else:
        print("\n🚫 Dobot not connected.")


if __name__ == "__main__":
    main()