import cv2
import numpy as np
from serial.tools import list_ports
from pydobotplus import Dobot
import time


class CameraToDobot:
    def __init__(self):
        # -----------------------------
        # Initialize Dobot Connection
        # -----------------------------
        port = list_ports.comports()[-1].device
        print(f"Connecting to Dobot on port: {port}")
        self.device = Dobot(port=port)
        print("Connected to Dobot")

        self.z_pickup = -50  # Height to pick up objects
        self.camera_pos = [225.2, -0.2, 152.5, 0]  # x, y, z, r
        self.device.move_to(*self.camera_pos, wait=True)
        time.sleep(2)
        print("Camera positioned above workspace.")

        # -----------------------------
        # Marker Configuration
        # -----------------------------
        self.marker_physical_coords = {
            0: (344.9, -39.2),
            1: (259.6, -39.5),
            2: (343.5, 70.1),
            3: (257.2, 69.6)
        }

        # ArUco Setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.homography = None
        self.start_time = time.time()  # For periodic movement trigger

    # -----------------------------
    # FUNCTION: Estimate Homography
    # -----------------------------
    def estimate_homography(self, corners, ids):
        img_pts, world_pts = [], []

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_physical_coords:
                c = corners[i][0]
                cx = np.mean(c[:, 0])
                cy = np.mean(c[:, 1])
                img_pts.append([cx, cy])
                world_pts.append(self.marker_physical_coords[marker_id])

        if len(img_pts) >= 4:
            H, _ = cv2.findHomography(np.array(img_pts), np.array(world_pts))
            np.save('data/transform.npy', H)
            return H

        return None

    # -----------------------------
    # FUNCTION: Detect Blue Dot
    # -----------------------------
    def detect_blue_dot(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
                return (cx, cy, x, y, w, h)
        return None

    # -----------------------------
    # FUNCTION: Process Frame
    # -----------------------------
    def process_frame(self, frame):
        # Detect ArUco markers
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            if len(ids) >= 4:
                self.homography = self.estimate_homography(corners, ids)

        # Detect Blue Dot
        blue_data = self.detect_blue_dot(frame)
        if blue_data is not None:
            cx, cy, x, y, w, h = blue_data
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            cam_coords = np.array([[cx, cy, 1]], dtype=np.float32).T

            if self.homography is not None:
                world_coords = np.dot(self.homography, cam_coords)
                world_coords /= world_coords[2, 0]
                xw, yw = world_coords[0, 0], world_coords[1, 0]
                text = f"Cam: ({cx:.0f}, {cy:.0f})  Phys: ({xw:.2f}, {yw:.2f})"

                # Perform Dobot movement every 5 seconds
                end = time.time()
                if end - self.start_time > 5:
                    self.move_pick_and_place(xw, yw)
                    self.start_time = time.time()

            else:
                text = f"Cam: ({cx:.0f}, {cy:.0f})  Phys: (N/A)"

            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    # -----------------------------
    # FUNCTION: Pick & Place Routine
    # -----------------------------
    def move_pick_and_place(self, xw, yw):
        """Replicates the original movement block from your code."""
        print(f"Moving Dobot to detected position: ({xw:.2f}, {yw:.2f}, {self.z_pickup})")

        self.device.move_to(xw, yw, self.z_pickup, 0, wait=True)
        print(f"Moved to: ({xw:.2f}, {yw:.2f}, {self.z_pickup})")

        self.device.suck(True)
        time.sleep(2)  # pickup delay

        self.device.home()
        self.device.move_to(290, -162, self.z_pickup, 0, wait=True)  # drop-off
        self.device.suck(False)
        time.sleep(2)  # drop delay

        self.device.home()
        self.device.move_to(*self.camera_pos, wait=True)
        print("Returned to camera position.")

    # -----------------------------
    # FUNCTION: Run Loop
    # -----------------------------
    def run(self):
        print("Starting camera feed...")
        print("Move to camera coordinates")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow("Camera to Physical Mapping", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera feed stopped.")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    system = CameraToDobot()
    system.run()