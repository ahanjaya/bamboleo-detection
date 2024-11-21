import cv2
import numpy as np
import json
from glob import glob
from typing import Dict


class ColorDetector:
    def __init__(
        self,
        config_file: str = "color_config.json",
        image_path: str = None,
        video_path: str = None,
    ):
        self.image_path = image_path
        self.video_path = video_path

        self.config_file = config_file
        self.colors = {
            "yellow": {"lower": [0, 0, 0], "upper": [255, 255, 255]},
        }

        # Try to load saved configuration
        try:
            self.load_config()
        except FileNotFoundError:
            pass

        self.window_name = "Color Detection Trackbars"
        cv2.namedWindow(self.window_name)
        self.create_trackbars()

    def create_trackbars(self) -> None:
        """Create trackbars for each color's HSV ranges"""

        def nothing(x):
            pass

        for color in self.colors:
            # Create trackbars for lower values
            cv2.createTrackbar(
                f"{color}_L_H",
                self.window_name,
                self.colors[color]["lower"][0],
                179,
                nothing,
            )
            cv2.createTrackbar(
                f"{color}_L_S",
                self.window_name,
                self.colors[color]["lower"][1],
                255,
                nothing,
            )
            cv2.createTrackbar(
                f"{color}_L_V",
                self.window_name,
                self.colors[color]["lower"][2],
                255,
                nothing,
            )

            # Create trackbars for upper values
            cv2.createTrackbar(
                f"{color}_U_H",
                self.window_name,
                self.colors[color]["upper"][0],
                179,
                nothing,
            )
            cv2.createTrackbar(
                f"{color}_U_S",
                self.window_name,
                self.colors[color]["upper"][1],
                255,
                nothing,
            )
            cv2.createTrackbar(
                f"{color}_U_V",
                self.window_name,
                self.colors[color]["upper"][2],
                255,
                nothing,
            )

    def setup_trackbars(self) -> None:
        """Set trackbar values to loaded configuration"""
        for color in self.colors:
            cv2.setTrackbarPos(
                f"{color}_L_H", self.window_name, self.colors[color]["lower"][0]
            )
            cv2.setTrackbarPos(
                f"{color}_L_S", self.window_name, self.colors[color]["lower"][1]
            )
            cv2.setTrackbarPos(
                f"{color}_L_V", self.window_name, self.colors[color]["lower"][2]
            )
            cv2.setTrackbarPos(
                f"{color}_U_H", self.window_name, self.colors[color]["upper"][0]
            )
            cv2.setTrackbarPos(
                f"{color}_U_S", self.window_name, self.colors[color]["upper"][1]
            )
            cv2.setTrackbarPos(
                f"{color}_U_V", self.window_name, self.colors[color]["upper"][2]
            )

    def get_trackbar_values(self) -> Dict:
        """Read current trackbar values"""
        current_values = {}

        for color in self.colors:
            current_values[color] = {
                "lower": [
                    cv2.getTrackbarPos(f"{color}_L_H", self.window_name),
                    cv2.getTrackbarPos(f"{color}_L_S", self.window_name),
                    cv2.getTrackbarPos(f"{color}_L_V", self.window_name),
                ],
                "upper": [
                    cv2.getTrackbarPos(f"{color}_U_H", self.window_name),
                    cv2.getTrackbarPos(f"{color}_U_S", self.window_name),
                    cv2.getTrackbarPos(f"{color}_U_V", self.window_name),
                ],
            }

        return current_values

    def save_config(self) -> None:
        """Save current HSV values to config file"""
        current_values = self.get_trackbar_values()
        with open(self.config_file, "w") as f:
            json.dump(current_values, f, indent=4)
        print(f"Configuration saved to {self.config_file}")

    def load_config(self) -> None:
        """Load HSV values from config file"""
        with open(self.config_file, "r") as f:
            self.colors = json.load(f)
        print(f"Configuration loaded from {self.config_file}")

    def get_center_of_mass(self, contour: np.ndarray):
        """Calculate center of mass of a contour"""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        return x, y

    def detect_colors(self, frame: np.ndarray):
        """Detect colors in the frame and return masked image and detected colors"""

        # 1. Detect yellow regions
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_values = self.get_trackbar_values()
        yellow_lower = np.array(current_values["yellow"]["lower"])
        yellow_upper = np.array(current_values["yellow"]["upper"])
        yellow_binary = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
        cv2.imshow("Yellow Binary", yellow_binary)

        contours, _ = cv2.findContours(
            yellow_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)
        yellow_mask = np.zeros_like(yellow_binary)
        cv2.drawContours(yellow_mask, [contour], -1, 255, -1)

        # calculate center of yellow mask
        yellow_mask_center = self.get_center_of_mass(contour)
        cx, cy = yellow_mask_center
        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
        cv2.imshow("Yellow Mask", yellow_mask)

        # Get frame inside yellow region
        yellow_frame = cv2.bitwise_and(frame, frame, mask=yellow_mask)
        hsv_frame = cv2.cvtColor(yellow_frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("Yellow Frame", yellow_frame)

        inv_yellow_binary = cv2.bitwise_not(yellow_binary)
        object_frame = cv2.bitwise_and(yellow_mask, inv_yellow_binary)
        cv2.imshow("Yellow Binary", inv_yellow_binary)
        cv2.imshow("Object Frame", object_frame)

        # detect object
        contours, _ = cv2.findContours(
            object_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return

        # remove small contours less than 500
        contours = [contour for contour in contours if cv2.contourArea(contour) > 200]

        # sort contours by distance from yellow mask center to contour center
        contours = sorted(
            contours,
            key=lambda x: np.linalg.norm(
                np.array(self.get_center_of_mass(x)) - np.array(yellow_mask_center)
            ),
        )

        for idx, contour in enumerate(contours):
            # center of object
            object_center = self.get_center_of_mass(contour)
            x, y = object_center
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"{idx+1}",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    def _init_images(self) -> None:
        self.images = glob(self.image_path + "/*.jpg")
        self.current_image_idx = 0
        self.total_images = len(self.images)

    def _init_video(self) -> None:
        self.cap = cv2.VideoCapture(self.video_path)

        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use 'XVID' or other codecs
        self.out = cv2.VideoWriter(
            "output.avi",
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    def run(self) -> None:
        if self.image_path:
            self._init_images()
        elif self.video_path:
            self._init_video()
        else:
            raise ValueError("Please provide either image or video path")

        while True:
            if self.image_path:
                self.current_image_idx = self.current_image_idx % self.total_images
                frame = cv2.imread(self.images[self.current_image_idx])
            elif self.video_path:
                ret, frame = self.cap.read()
                if not ret:
                    break

            # Resize frame
            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

            # Detect colors and get result
            self.detect_colors(frame)

            if self.video_path:
                # save frame into video
                self.out.write(frame)

            # Display results
            cv2.imshow("Original", frame)

            # Handle keyboard input
            key = cv2.waitKey(20)
            if key == ord("q"):
                break
            elif key == ord("a") and self.image_path:
                self.current_image_idx -= 1
            elif key == ord("d") and self.image_path:
                self.current_image_idx += 1
            elif key == ord("s"):
                detector.save_config()
            elif key == ord("l"):
                detector.load_config()
                detector.setup_trackbars()  # Reset trackbars to loaded values

        if self.video_path:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="color_config.json",
        help="Path to configuration file",
    )
    # parse image or video mutually exclusive group
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, help="Path to image file")
    group.add_argument("--video", type=str, help="Path to video file")
    args = parser.parse_args()

    detector = ColorDetector(args.config, args.image, args.video)
    detector.run()
