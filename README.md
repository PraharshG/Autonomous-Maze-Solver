# AI-Powered Dobot Maze Solver

This project creates a fully autonomous system that solves physical mazes using a **Dobot Magician Lite** robotic arm. It integrates computer vision, coordinate mapping, and Generative AI (Google Gemini) to perceive the environment, generate solution code dynamically, and physically trace the path.

## 🚀 Features

*   **Computer Vision Pipeline**: Automatically detects, crops, and digitizes a maze from a live camera feed, handling perspective distortion and rotation.
*   **Generative AI Agent**: Uses **Google Gemini 2.5 Flash** to analyze the maze image and *write its own Python code* to solve it (Pathfinding Agent).
*   **Robust Calibration**: Maps camera coordinates to the robot's physical space using Homography and ArUco markers.
*   **Physical Execution**: Converts the digital solution path into precise robotic movements to trace the solution on paper.

## 🛠️ Hardware Requirements

*   **Dobot Magician Lite** (or compatible Dobot arm)
*   **USB Webcam** (mounted overhead)
*   **Printed Maze** (Must have a **Red** start dot and **Green** end dot)
*   **ArUco Markers** (4x4 markers placed at the workspace corners for calibration)

## 📦 Software Requirements

*   Python 3.8+
*   Google Gemini API Key

### Dependencies
Install the required libraries:
```bash
pip install opencv-python numpy pydobotplus pyserial matplotlib scipy google-genai pillow
```

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`main.py`** | **Entry point.** Orchestrates the entire workflow: image capture, vision processing, agent code generation, and robot execution. |
| `transform_modular.py` | Handles camera-to-robot calibration using ArUco markers to generate a Homography matrix. |
| `maze_isolation_modular.py` | Detects the maze paper, crops it (removing background), and corrects perspective/rotation. |
| `maze_digitizer_modular.py` | Converts the visual maze into a binary grid and identifies Start (Red) and End (Green) points. |
| `agent_writer.py` | Interfaces with Google Gemini to generate the `agent_solution.py` script based on the current maze image. |
| `agent_solution.py` | **(Generated)** The script written by the AI agent that performs A* pathfinding on the specific maze. |
| `dobot_solve_v2.py` | Reads the solved path image, converts pixels to robot coordinates, and controls the Dobot. |
| `flip_colors.py` | Utility to preprocess images for the AI agent (inverts colors for better contrast). |

## ⚙️ Setup & Usage

### 1. Calibration
Before running the solver, you must calibrate the camera-to-robot coordinate system.
1.  Place 4 ArUco markers at the corners of your workspace.
2.  Update `marker_physical_coords` in `transform_modular.py` with the real-world (x, y) coordinates of your markers.
3.  Run the calibration script:
    ```bash
    python transform_modular.py
    ```
    This will save the homography matrix to `data/transform.npy`.

### 2. Configuration
*   **API Key**: Open `main.py` (or set an environment variable) and ensure your **Google Gemini API Key** is configured.
*   **Ports**: Ensure your Dobot is connected via USB. The script auto-detects the port, but you may need to adjust permissions on Linux (`sudo chmod 666 /dev/ttyUSB0`).

### 3. Run the Solver
Place a maze in the camera's view and run:
```bash
python main.py
```

### 🔁 Workflow Breakdown
1.  **Capture**: The robot moves the camera to a standard viewing position.
2.  **Isolate**: The system crops the maze and creates `images/isolated_maze_strict.png`.
3.  **Digitize**: It locates the red/green dots and creates a binary representation.
4.  **Generate**: Gemini writes a custom `agent_solution.py` to solve the specific maze structure.
5.  **Solve**: The generated script runs, producing `images/maze_solution.png` with a drawn blue path.
6.  **Execute**: The robot traces the blue path physically on the paper.

## 🤖 The "Agent" Concept
Unlike traditional scripts that hard-code a specific solver, this project uses an **Agent Writer** (`agent_writer.py`). It feeds the maze image and context to an LLM, which then *writes* the Python code necessary to solve that specific instance. This allows the system to potentially adapt to different types of puzzles or constraints without rewriting the core logic.

## ⚠️ Safety Note
*   Always keep your hand near the emergency stop (or USB cable) when the robot is moving.
*   Ensure the `Z_HEIGHT_PATH` in `dobot_solve_v2.py` is set correctly to avoid the pen crashing into the table.

## 📝 License
[MIT License](LICENSE)
