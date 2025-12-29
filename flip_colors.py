import cv2
import numpy as np
import os

def invert_black_white_keep_red_green(image_path, output_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert to HSV for color-based filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Define color masks ---
    # Detect red (two ranges because hue wraps around)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Detect green
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks for colors we want to KEEP
    keep_mask = cv2.bitwise_or(red_mask, green_mask)

    # --- Detect black and white regions ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < 50).astype(np.uint8) * 255
    white_mask = (gray > 200).astype(np.uint8) * 255

    # --- Invert only black & white pixels that are NOT red/green ---
    invert_mask = cv2.bitwise_and(cv2.bitwise_or(black_mask, white_mask),
                                  cv2.bitwise_not(keep_mask))

    # Create output copy
    output = img.copy()

    # Invert where appropriate
    output[invert_mask == 255] = 255 - output[invert_mask == 255]

    # Save result
    cv2.imwrite(output_path, output)
    print(f"✅ Saved inverted image to {output_path}")

# Example:
if __name__ == "__main__":
    invert_black_white_keep_red_green("images/digitized_maze_visualized.png", "images/maze_inverted.png")