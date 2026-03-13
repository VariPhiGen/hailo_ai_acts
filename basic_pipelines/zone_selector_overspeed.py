import cv2
import json
import numpy as np
import argparse

class ZoneDrawer:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise ValueError("Cannot open video source")

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Couldn't read from video source")

        self.original_height, self.original_width = frame.shape[:2]
        self.resize_width = 480
        self.resize_height = int(self.original_height * (self.resize_width / self.original_width))
        self.scale_x = self.original_width / self.resize_width
        self.scale_y = self.original_height / self.resize_height

        self.saved_polygon = []
        self.saved_line = []
        self.current_polygon = []
        self.current_line = []
        self.mode = 'polygon'

        self.window_name = 'Zone & Line Drawer'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'polygon':
                self.current_polygon.append((x, y))
            elif self.mode == 'line' and len(self.current_line) < 2:
                self.current_line.append((x, y))

    def save_zones(self, path="zones_and_lines.json"):
        # Prepare zone and line data
        zone_points = [(int(x * self.scale_x), int(y * self.scale_y)) for x, y in self.saved_polygon]
        line_points = [(int(x * self.scale_x), int(y * self.scale_y)) for x, y in self.saved_line]

        data = {
            "source": self.source,
            "zone": {
                "name": "box",
                "points": zone_points
            },
            "line": {
                "name": "line",
                "points": line_points
            }
        }

        # Save JSON without spaces
        with open(path, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        print(f"✅ Saved JSON to {path}")

        # Reset video to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Could not re-read original frame for saving image.")
            return

        # Draw polygon
        if zone_points:
            cv2.polylines(frame, [np.array(zone_points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw line
        if len(line_points) == 2:
            cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)

        # Save annotated image
        cv2.imwrite("org_img.png", frame)
        print("✅ Saved image with polygon and line to org_img.png")

    def run(self):
        frame = None
        while True:
            if frame is None:
                ret, frame = self.cap.read()
            
            if not ret:
                break

            display_frame = cv2.resize(frame, (self.resize_width, self.resize_height))

            if self.saved_polygon:
                cv2.polylines(display_frame, [np.array(self.saved_polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

            if self.saved_line and len(self.saved_line) == 2:
                cv2.line(display_frame, self.saved_line[0], self.saved_line[1], (0, 0, 255), 2)

            if self.current_polygon:
                cv2.polylines(display_frame, [np.array(self.current_polygon)], isClosed=False, color=(0, 255, 255), thickness=1)

            if len(self.current_line) == 1:
                cv2.circle(display_frame, self.current_line[0], 3, (0, 0, 255), -1)
            elif len(self.current_line) == 2:
                cv2.line(display_frame, self.current_line[0], self.current_line[1], (0, 0, 255), 1)

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_zones()
            elif key == ord('f'):
                if self.mode == 'polygon' and self.current_polygon:
                    self.saved_polygon = self.current_polygon.copy()
                    self.current_polygon.clear()
                    print(f"✅ Polygon saved with 'f': {self.saved_polygon}")
                    self.mode = 'line'
                    print("Switched to LINE mode")
                elif self.mode == 'line' and len(self.current_line) == 2:
                    self.saved_line = self.current_line.copy()
                    self.current_line.clear()
                    print(f"✅ Line saved with 'f': {self.saved_line}")
                    self.mode = 'polygon'
                    print("Switched to POLYGON mode")
            elif key == ord('l'):
                self.mode = 'line'
                print("Switched to LINE mode")
            elif key == ord('c'):
                self.mode = 'polygon'
                print("Switched to POLYGON mode")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Video file path or RTSP stream URL")
    args = parser.parse_args()

    drawer = ZoneDrawer(args.source)
    drawer.run()
