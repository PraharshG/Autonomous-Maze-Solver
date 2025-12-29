from transform_modular import CameraToDobot
from maze_isolation_modular import StrictBlackCropper
from maze_digitizer_modular import load_and_resize_image, convert_to_binary, detect_start_end_points_and_clean, visualize_maze
from flip_colors import invert_black_white_keep_red_green
from agent_writer import generate_code_from_image_and_modules
# Run agent_solution.py after it has been generated
from dobot_solve_v2 import main as dobot_main

import cv2
import numpy as np
from serial.tools import list_ports
from pydobotplus import Dobot
import time
import os
import json
import matplotlib.pyplot as plt
from serial.tools import list_ports
from pydobotplus import Dobot
from scipy.spatial.distance import cdist 
import re
from google import genai
from google.genai import types
from PIL import Image
import subprocess

ISOLATION_IMAGE_PATH = "images/capture_for_isolation.png"
ISOLATION_OUTPUT_PATH = "images/isolated_maze_strict.png"
DIGITIZER_IMAGE_PATH = "images/isolated_maze_strict.png"
DIGITIZER_OUTPUT_PATH = "images/digitized_maze_visualized.png"
DEBUG = True
TRANSFORM_JSON = "data/crop_warp_transform.json"
GEMINI_API_KEY = "AIzaSyCZUNmcJ8eRXbvx_DqnCQEyVpS_uI5Q0ZE" 
GEMINI_IMAGE_PATH = "images/maze_inverted.png"
GEMINI_OUTPUT_FILE = "agent_solution.py"
TRANSFORM_FILE = 'data/transform.npy'
CROP_WARP_DATA_FILE = 'data/crop_warp_transform.json'
SOLVED_MAZE_IMAGE = 'images/maze_solution.png'
OUTPUT_WAYPOINTS_FILE = 'data/dobot_waypoints.npy'
Z_HEIGHT_PATH = -40
Z_HEIGHT_LIFT = 0
LOWER_BLUE = np.array([100, 50, 0])
UPPER_BLUE = np.array([255, 150, 100])
MODULE_PATHS = ["agent_solution.py"]
SYSTEM_INSTRUCTION = (
    "You are an expert Python programmer specializing in image processing and computer vision. "
    "Your response should be the final, complete, self-contained Python script to solve the maze. "
    "If context modules are provided, assume their logic is available. "
    "Output MUST be a single Python markdown code block (```python ... ```) with no extra text."
)

def show_solution_image(image_path="images/maze_solution.png"):
    """
    Opens and continuously displays the solved maze image in a window.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not open {image_path}")
        return None

    cv2.imshow("Maze Solution", img)
    cv2.waitKey(1)  # Just refresh once
    return img


### STEP 1: ISOLATE MAZE FROM BACKGROUND ###

# Connect to Dobot
port = list_ports.comports()[-1].device
print(f"Connecting to Dobot on port: {port}")
device = Dobot(port=port)
device.move_to(225, 0, 145, 0, wait=True)
time.sleep(2)

# Capture image from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to capture frame from camera.")
cv2.imwrite("images/capture_for_isolation.png", frame)

# Run strict black cropper
cropper = StrictBlackCropper(
    image_path=ISOLATION_IMAGE_PATH,
    output_path=ISOLATION_OUTPUT_PATH,
    upper_black_v=70,  # make stricter by lowering
    margin=-5,
    debug=True
)
cropper.process()

### STEP 2: DIGITIZE MAZE AND DETECT START/END POINTS ###

with open(TRANSFORM_JSON, 'r') as f:
    transform_data = json.load(f)

image, transform_data = load_and_resize_image(DIGITIZER_IMAGE_PATH, TRANSFORM_JSON)
binary, maze_grid = convert_to_binary(image)
cleaned_binary, start_pt, end_pt = detect_start_end_points_and_clean(binary, image)

print(f"✅ Start point (red): {start_pt}")
print(f"✅ End point (green): {end_pt}")
print(f"✅ Grid size: {maze_grid.shape}")

visualize_maze(cleaned_binary, start_pt, end_pt, DIGITIZER_OUTPUT_PATH)
output_metadata = {
    "start_point": start_pt,
    "end_point": end_pt,
    "grid_shape": maze_grid.shape,
    "transform_data": transform_data
}

# with open("data/maze_metadata.json", "w") as f:
#     json.dump(output_metadata, f, indent=4)

print("✅ Maze metadata (including transform) saved as maze_metadata.json")

### STEP 3: INVERT COLORS FOR AGENT PROCESSING ###
invert_black_white_keep_red_green("images/digitized_maze_visualized.png", "images/maze_inverted.png")

### STEP 4: GENERATE SOLUTION CODE USING GEMINI API ###
generate_code_from_image_and_modules(GEMINI_API_KEY, GEMINI_IMAGE_PATH, GEMINI_OUTPUT_FILE, MODULE_PATHS, SYSTEM_INSTRUCTION)

### STEP 5: RUN THE GENERATED CODE TO SOLVE THE MAZE WITH DOBOT ###
try:
    # Start the subprocess without waiting for it to finish immediately
    process = subprocess.Popen(['python3', GEMINI_OUTPUT_FILE])

    print(f"Script '{GEMINI_OUTPUT_FILE}' started. Waiting for it to finish...")

    # Wait for the subprocess to complete
    process.wait()

    print(f"Script '{GEMINI_OUTPUT_FILE}' finished with return code: {process.returncode}")

except FileNotFoundError:
    print(f"Error: Python executable or script '{GEMINI_OUTPUT_FILE}' not found.")

# STEP 6: SHOW SOLVED MAZE IMAGE WHILE ROBOT EXECUTES
img = show_solution_image(SOLVED_MAZE_IMAGE)
print("🧭 Displaying solved maze while Dobot executes path...")

# Run the Dobot solver
dobot_main()

# Keep the image displayed until you press a key
cv2.waitKey(0)
cv2.destroyAllWindows()
device.close()