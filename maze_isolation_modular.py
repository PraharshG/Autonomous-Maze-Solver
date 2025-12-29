import cv2
import numpy as np
import os
import json
import time


class StrictBlackCropper:
    """
    Performs strict-black-based contour cropping and optional perspective warp.
    Now includes automatic rotation correction before cropping.
    """

    def __init__(
        self,
        image_path,
        output_path="images/isolated_maze_strict.png",
        upper_black_v=70,
        margin=-5,
        debug=False,
        transform_save_path="data/crop_warp_transform.json",
    ):
        self.image_path = image_path
        self.output_path = output_path
        self.upper_black_v = upper_black_v
        self.margin = margin
        self.debug = debug

        self.image = None
        self.mask = None
        self.large_contours = []
        self.crop_box = None
        self.transform_save_path = transform_save_path

        self.transform_data = {
            "crop_box": None,
            "is_warped": False,
            "warp_matrix": None,
            "target_dimensions": None,
            "rotation_angle": 0,
            "rotation_matrix": None
        }

    # ==========================
    # Step 1: Load Image
    # ==========================
    def load_image(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")
        return self.image

    # ==========================
    # Step 2: Create Strict Black Mask
    # ==========================
    def create_strict_black_mask(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, self.upper_black_v])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.mask = mask
        return mask

    # ==========================
    # Step 3: Find and Filter Contours
    # ==========================
    def find_large_contours(self):
        h, w = self.image.shape[:2]
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contours found — try increasing upper_black_v slightly.")

        min_area = (h * w) * 0.005
        self.large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not self.large_contours:
            raise RuntimeError("No sufficiently large contours found. Try relaxing thresholds.")

        return self.large_contours

    # ==========================
    # Step 4: Auto Rotate Image if Required
    # ==========================
    def auto_rotate_image(self):
        largest = max(self.large_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        angle = rect[-1]

        # OpenCV's minAreaRect angle correction logic
        if angle < -45:
            angle = 90 + angle

        # Rotate image if significant tilt
        if abs(angle) > 1:  # small angles ignored
            (h, w) = self.image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Save rotation info for reversal
            self.transform_data["rotation_angle"] = angle
            self.transform_data["rotation_matrix"] = M.tolist()

            # Save rotated image for reference
            rotated_path = "images/rotated_before_crop.png"
            cv2.imwrite(rotated_path, rotated)
            print(f"Image rotated by {angle:.2f} degrees. Saved as {rotated_path}")

            self.image = rotated
            # Recreate mask after rotation
            self.create_strict_black_mask()
            self.find_large_contours()

    # ==========================
    # Step 5: Crop Region
    # ==========================
    def crop_strict_region(self):
        orig = self.image.copy()
        h, w = self.image.shape[:2]

        all_points = np.vstack(self.large_contours)
        x, y, ww, hh = cv2.boundingRect(all_points)

        margin = self.margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        ww = min(w - x, ww + 2 * margin)
        hh = min(h - y, hh + 2 * margin)

        cropped = orig[y:y + hh, x:x + ww]
        self.crop_box = (x, y, ww, hh)
        self.transform_data['crop_box'] = self.crop_box

        return cropped

    # ==========================
    # Step 6: Optional Warp
    # ==========================
    def warp_if_possible(self, cropped):
        largest = max(self.large_contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            from_points = np.float32(approx.reshape(4, 2))
            (tl, tr, br, bl) = from_points

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)

            maxWidth = int(max(widthA, widthB))
            maxHeight = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(from_points, dst)
            warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))

            self.transform_data['is_warped'] = True
            self.transform_data['warp_matrix'] = M.tolist()
            self.transform_data['target_dimensions'] = (maxWidth, maxHeight)

            return warped, True

        return cropped, False

    # ==========================
    # Step 7: Debug Visualization
    # ==========================
    def save_debug_images(self):
        if not self.debug:
            return

        debug = self.image.copy()
        cv2.drawContours(debug, self.large_contours, -1, (0, 255, 0), 2)

        if self.crop_box:
            x, y, ww, hh = self.crop_box
            cv2.rectangle(debug, (x, y), (x + ww, y + hh), (0, 0, 255), 3)

        cv2.imwrite("images/debug_strict_black.png", debug)
        cv2.imwrite("images/mask_strict_black.png", self.mask)
        print("Debug images saved: debug_strict_black.png, mask_strict_black.png")

    # ==========================
    # Step 8: Save Transform Data
    # ==========================
    def save_transform_data(self):
        self.transform_data["original_image_size"] = self.image.shape[:2]
        with open(self.transform_save_path, 'w') as f:
            json.dump(self.transform_data, f, indent=4)
        print(f"Transformation data saved to: {self.transform_save_path}")

    # ==========================
    # Main Processing Function
    # ==========================
    def process(self):
        """Run full pipeline: load → mask → contour → rotate → crop → warp"""
        print(f"Processing image: {os.path.basename(self.image_path)}")

        self.load_image()
        self.create_strict_black_mask()
        self.find_large_contours()
        self.auto_rotate_image()
        cropped = self.crop_strict_region()
        result, warped = self.warp_if_possible(cropped)
        self.save_transform_data()
        cv2.imwrite(self.output_path, result)

        if warped:
            print(f"Warped maze saved as: {self.output_path}")
        else:
            print(f"Strict cropped maze saved as: {self.output_path}")

        self.save_debug_images()
        return result


# ==========================
# CLI / Live Capture Entry Point
# ==========================
if __name__ == "__main__":
    from serial.tools import list_ports
    from pydobotplus import Dobot

    port = list_ports.comports()[-1].device
    print(f"Connecting to Dobot on port: {port}")
    device = Dobot(port=port)
    # device.move_to(225, 0, 145, 0, wait=True)
    time.sleep(2)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera.")
    cv2.imwrite("images/capture_for_isolation.png", frame)

    IMAGE_PATH = "images/capture_for_isolation.png"
    OUTPUT_PATH = "images/isolated_maze_strict.png"

    cropper = StrictBlackCropper(
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        upper_black_v=70,
        margin=-5,
        debug=True
    )
    cropper.process()
    device.close()