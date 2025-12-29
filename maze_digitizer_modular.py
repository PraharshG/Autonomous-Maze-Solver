import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os


# ==========================
# CONFIGURATION
# ==========================
IMAGE_PATH = "images/isolated_maze_strict.png"
OUTPUT_PATH = "images/digitized_maze_visualized.png"
DEBUG = True
TRANSFORM_JSON = "data/crop_warp_transform.json"

if os.path.exists(TRANSFORM_JSON):
    with open(TRANSFORM_JSON, 'r') as f:
        transform_data = json.load(f)
else:
    transform_data = {}


# ==========================
# STEP 1 — LOAD IMAGE
# ==========================
def load_and_resize_image(image_path, transform_json_path, max_dim=800):
    # Load existing transform data
    with open(transform_json_path, 'r') as f:
        transform_data = json.load(f)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")

    original_height, original_width = image.shape[:2]
    scale = max_dim / max(original_height, original_width)
    resized = False

    if scale < 1:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        image = cv2.resize(image, (new_width, new_height))
        resized = True
    else:
        new_width, new_height = original_width, original_height

    # --- Save resize info back to JSON ---
    transform_data["resize_applied"] = resized
    transform_data["resize_scale"] = scale if resized else 1.0
    transform_data["resized_image_size"] = [new_height, new_width]

    with open(transform_json_path, 'w') as f:
        json.dump(transform_data, f, indent=4)

    return image, transform_data



# ==========================
# STEP 2 — CONVERT TO BINARY
# ==========================
def convert_to_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if walls are white
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)

    maze_grid = (binary == 255).astype(np.uint8)
    return binary, maze_grid


# ==========================
# STEP 3 — DETECT START & END POINTS
# ==========================
def get_center(mask):
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def detect_start_end_points_and_clean(binary, image, debug=True):
    """
    Robust detection of start (red) and end (green) circles in a maze.
    - Works with filled color circles.
    - Always assigns green to the circle that is not red.
    - Prevents removal from intruding into maze walls.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- HSV ranges for color detection ---
    lower_red1, upper_red1 = np.array([0, 100, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 80]), np.array([179, 255, 255])

    # --- Preprocess for circle detection ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=100,
        param2=25,
        minRadius=15,
        maxRadius=100
    )

    start_pt, end_pt = None, None
    start_r, end_r = 0, 0
    red_circle_indices = []

    # ===============================
    # STEP 1 — Identify Circles
    # ===============================
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for i, (x, y, r) in enumerate(circles):
            # Sample small region in the circle center
            region = hsv[max(0, y - 5):y + 5, max(0, x - 5):x + 5]
            if region.size == 0:
                continue
            mean_hsv = np.mean(region.reshape(-1, 3), axis=0)
            hue = mean_hsv[0]

            # Check if it's red
            if (hue < 15) or (hue > 160):
                red_circle_indices.append(i)

        # --- Assign roles ---
        if len(red_circle_indices) >= 1:
            # First red circle is start
            i = red_circle_indices[0]
            x, y, r = circles[i]
            start_pt, start_r = (x, y), r

            # Assign green as "the other circle"
            for j, (x, y, r) in enumerate(circles):
                if j != i:
                    end_pt, end_r = (x, y), r
                    break
        else:
            # No red found — assign by order (first red assumed missing)
            if len(circles) >= 2:
                (x1, y1, r1), (x2, y2, r2) = circles[:2]
                start_pt, start_r = (x1, y1), r1
                end_pt, end_r = (x2, y2), r2
            elif len(circles) == 1:
                # Only one circle: assume it's red
                x, y, r = circles[0]
                start_pt, start_r = (x, y), r

    # ===============================
    # STEP 2 — If no circles found, use color fallback
    # ===============================
    if start_pt is None or end_pt is None:
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                start_pt = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                start_r = int(np.sqrt(cv2.contourArea(largest) / np.pi))
        print("[⚠️] Hough detection failed, fallback to red contour mask.")

    # ===============================
    # STEP 3 — Clean detected regions
    # ===============================
    cleaned_binary = binary.copy()
    mask_remove = np.zeros_like(binary)

    if start_pt:
        cv2.circle(mask_remove, start_pt, start_r + 3, 255, -1)
    if end_pt:
        cv2.circle(mask_remove, end_pt, end_r + 3, 255, -1)

    # Subtract circles from binary (preserving walls)
    cleaned_binary[mask_remove > 0] = 0

    # Optional smoothing
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # ===============================
    # STEP 4 — Debug Visualization
    # ===============================
    if debug:
        vis = cv2.cvtColor(cleaned_binary, cv2.COLOR_GRAY2BGR)
        if start_pt:
            cv2.circle(vis, start_pt, 8, (0, 0, 255), -1)
        if end_pt:
            cv2.circle(vis, end_pt, 8, (0, 255, 0), -1)
        plt.figure(figsize=(8, 8))
        plt.title("Robust Circle Detection (start:red, end:green)")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return cleaned_binary, start_pt, end_pt


# ==========================
# STEP 4 — VISUALIZATION
# ==========================
def visualize_maze(binary, start_pt, end_pt, output_path):
    maze_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Overlay start and end points
    cv2.circle(maze_vis, start_pt, 8, (0, 0, 255), -1)  # Red = start
    cv2.circle(maze_vis, end_pt, 8, (0, 255, 0), -1)    # Green = end

    # Save and display visualization
    cv2.imwrite(output_path, maze_vis)

    plt.figure(figsize=(8, 8))
    plt.title("Digitized Maze Visualization")
    plt.imshow(cv2.cvtColor(maze_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print(f"✅ Visualization saved as {output_path}")


# ==========================
# MAIN FUNCTION
# ==========================
def main():
    image, transform_data = load_and_resize_image(IMAGE_PATH, TRANSFORM_JSON)
    binary, maze_grid = convert_to_binary(image)
    cleaned_binary, start_pt, end_pt = detect_start_end_points_and_clean(binary, image)

    print(f"✅ Start point (red): {start_pt}")
    print(f"✅ End point (green): {end_pt}")
    print(f"✅ Grid size: {maze_grid.shape}")

    visualize_maze(cleaned_binary, start_pt, end_pt, OUTPUT_PATH)
    output_metadata = {
        "start_point": start_pt,
        "end_point": end_pt,
        "grid_shape": maze_grid.shape,
        "transform_data": transform_data
    }

    with open("data/maze_metadata.json", "w") as f:
        json.dump(output_metadata, f, indent=4)

    print("✅ Maze metadata (including transform) saved as maze_metadata.json")


if __name__ == "__main__":
    main()