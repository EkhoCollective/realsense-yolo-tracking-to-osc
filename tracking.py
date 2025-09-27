import cv2
import numpy as np
import pyrealsense2 as rs # Intel RealSense SDK
from ultralytics import YOLO
from pythonosc import udp_client
import math
import time # Import the time module
import argparse # Add argparse for command-line arguments

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="YOLOv8 RealSense Tracker")
parser.add_argument("--ip", default="127.0.0.1", help="OSC server IP")
parser.add_argument("--port", type=int, default=5005, help="OSC server port")
parser.add_argument("--height", type=float, default=3.46, help="Camera height in meters")
parser.add_argument("--offset", type=float, default=0.5, help="Camera X-axis offset in meters")
parser.add_argument("--conf", type=float, default=0.4, help="YOLO detection confidence threshold")
parser.add_argument("--no-video", action="store_true", help="Run in headless mode without video output.")
parser.add_argument("--stillness", type=float, default=5.0, help="How long a person must be still (in seconds)")
parser.add_argument("--tilt", type=float, default=64.4, help="Camera tilt angle in degrees")
parser.add_argument("--tolerance", type=int, default=1, help="Grid cell movement tolerance for stillness")
parser.add_argument("--rgb-exposure", type=int, default=1000, help="RGB camera exposure value (-1 for auto)")
parser.add_argument("--yaw", type=float, default=10.0, help="Camera yaw angle in degrees (positive = right)")
parser.add_argument("--rs-width", type=int, default=640, help="RealSense stream width in pixels")
parser.add_argument("--rs-height", type=int, default=480, help="RealSense stream height in pixels")
args = parser.parse_args()
MOVEMENT_TOLERANCE = args.tolerance

CAMERA_TILT_DEGREES = args.tilt
CAMERA_TILT_RADIANS = math.radians(CAMERA_TILT_DEGREES)
CAMERA_YAW_DEGREES = args.yaw
CAMERA_YAW_RADIANS = math.radians(CAMERA_YAW_DEGREES)

# --- OSC Configuration ---
OSC_IP = args.ip
OSC_PORT = args.port
OSC_ADDRESS = "/occupied_squares"
osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
OSC_SEND_INTERVAL = 0.5 # How often we check for still people
STILLNESS_DURATION = args.stillness # How long a person must be still (in seconds)

# --- Grid Configuration ---
CAMERA_HEIGHT_M = args.height # 3.5 meters
# Adjust this value to shift the grid origin. 
# If the camera is 1.5m to the left of the room's center, set this to 1.5.
CAMERA_X_OFFSET_M = args.offset

# --- Grid Visualization Configuration ---
GRID_DIM_METERS = 40  # Visualize a 20m x 20m area
GRID_PIXELS = 500     # Size of the grid visualization window
CELL_PIXELS = GRID_PIXELS // GRID_DIM_METERS
GRID_ORIGIN_OFFSET = GRID_DIM_METERS // 2 # To center (0,0) in the visualization

# --- 1. RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()

# Configure the streams for the color and depth camera
# Choose a lower resolution (640x480) and standard FPS (30) for best performance
W, H = args.rs_width, args.rs_height
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# Start streaming
print("[INFO] Starting RealSense pipeline...")
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"[INFO] Depth Scale is: {depth_scale}")

# --- RGB Exposure Control ---
color_sensor = None
for sensor in profile.get_device().query_sensors():
    if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
        color_sensor = sensor
        break

if color_sensor:
    if args.rgb_exposure >= 0:
        try:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, args.rgb_exposure)
            print(f"[INFO] RGB Exposure set to: {args.rgb_exposure}")
        except Exception as e:
            print(f"[WARNING] Could not set RGB exposure: {e}")
    else:
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        print("[INFO] RGB auto exposure enabled.")
else:
    print("[WARNING] RGB camera not found.")

# --- Create an align object
# rs.align allows us to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

print("[INFO] Camera ready.")

# --- 2. Model Initialization ---
# Load the ultra-efficient YOLOv8-Nano model
model = YOLO('yolov8n.pt')

# --- Helper function for Grid Visualization ---
def draw_grid_visualization(occupied_cells):
    """Creates an image representing the top-down grid view centered at (0,0)."""
    grid_img = np.zeros((GRID_PIXELS, GRID_PIXELS, 3), dtype=np.uint8)
    
    # Draw grid lines
    for i in range(1, GRID_DIM_METERS):
        pos = i * CELL_PIXELS
        cv2.line(grid_img, (pos, 0), (pos, GRID_PIXELS), (40, 40, 40), 1)
        cv2.line(grid_img, (0, pos), (GRID_PIXELS, pos), (40, 40, 40), 1)

    # Center (0,0) in the middle of the grid image
    center_pixel_x = GRID_PIXELS // 2
    center_pixel_y = GRID_PIXELS // 2

    for x, y in occupied_cells:
        # Translate world coordinates to pixel coordinates
        px = center_pixel_x + x * CELL_PIXELS
        py = center_pixel_y - y * CELL_PIXELS  # Flip Y axis

        # Check if the cell is within the drawable area
        if 0 <= px < GRID_PIXELS and 0 <= py < GRID_PIXELS:
            cv2.rectangle(grid_img, (px, py), (px + CELL_PIXELS, py + CELL_PIXELS),
                          (0, 255, 0), -1) # Draw a filled green square
            
    return grid_img

# --- 3. Tracking Loop ---
# Variables for timed OSC sending
last_osc_send_time = time.time()
# New dictionary to track the state of each person
person_states = {} 

# --- Visualization State ---
show_window = not args.no_video
window_name = "YOLOv8 ByteTrack on RealSense"
grid_window_name = "Occupancy Grid"

# --- Tracking Loop ---
STICKY_CELL_DURATION = 1.0  # seconds a person must stay in a new cell before updating

try:
    while True:
        # Initialize a set for the current frame's grid cells for visualization
        current_frame_grid_cells = set()
        # Unified set for cells of people confirmed to be "still"
        still_cells = set()

        # Wait for a new set of frames from the camera and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Get camera intrinsics for coordinate mapping
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert the color frame to a numpy array (OpenCV format)
        color_image = np.asanyarray(color_frame.get_data())

        imgsz = ((W + 31) // 32) * 32  # Ensure width is a multiple of 32
        # Run YOLO tracking on the frame
        results = model.track(
            color_image,
            conf=args.conf,
            classes=[0],      # Track only 'person' class (ID 0)
            imgsz=imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        # --- Annotation and Visualization ---
        # This part runs only if the window is supposed to be shown
        if show_window:
            annotated_frame = results[0].plot()
        else:
            # We still need color_image for the key press handling loop
            annotated_frame = color_image 

        # Get a set of all IDs present in the current frame
        current_frame_ids = set()
        if results[0].boxes.id is not None:
            current_frame_ids = set(results[0].boxes.id.cpu().numpy().astype(int))

            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, clss):
                # Check if the detected object is a person
                if model.names[cls_id] == 'person':
                    x1, y1, x2, y2 = box
                    # Get the center of the bounding box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Get the distance to the center of the bounding box
                    distance_m = depth_frame.get_distance(cx, cy)

                    # If a valid distance is found
                    if 0 < distance_m < 10: # Check for a reasonable distance
                        # Deproject 2D pixel to 3D point in camera coordinates
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], distance_m)
                        
                        # Get coordinates relative to the camera and apply the offset
                        # Adjust for camera tilt: project the detected point onto the floor plane
                        # Camera is at height CAMERA_HEIGHT_M, tilted CAMERA_TILT_RADIANS from vertical
                        # point_3d[2] is the Z (forward) distance from camera, point_3d[1] is Y (vertical)
                        # We want the horizontal distance from the camera base to the detected point on the floor

                        # Calculate the vertical distance from camera to detected point
                        vertical_distance = CAMERA_HEIGHT_M - point_3d[1]
                        # Calculate the horizontal distance using tilt
                        floor_y = vertical_distance * math.tan(CAMERA_TILT_RADIANS) + point_3d[2] * math.cos(CAMERA_TILT_RADIANS)
                        floor_x = point_3d[0] + CAMERA_X_OFFSET_M

                        # --- Apply yaw rotation ---
                        # Rotate (floor_x, floor_y) around the camera origin by CAMERA_YAW_RADIANS
                        rotated_x = floor_x * math.cos(CAMERA_YAW_RADIANS) - floor_y * math.sin(CAMERA_YAW_RADIANS)
                        rotated_y = floor_x * math.sin(CAMERA_YAW_RADIANS) + floor_y * math.cos(CAMERA_YAW_RADIANS)

                        # For visualization, calculate grid position on every frame
                        grid_x = math.floor(rotated_x)
                        grid_y = math.floor(rotated_y)
                        current_frame_grid_cells.add((grid_x, grid_y))

                        # --- Update Person State for Stillness Detection ---
                        current_cell = (grid_x, grid_y)
                        current_time_for_state = time.time()

                        if track_id not in person_states:
                            # New person detected
                            person_states[track_id] = {
                                'origin_cell': current_cell,
                                'current_cell': current_cell,
                                'sticky_cell': current_cell,
                                'sticky_since': current_time_for_state,
                                'still_since': current_time_for_state,
                                'osc_sent': False
                            }
                        else:
                            # Existing person, check if they moved beyond tolerance
                            origin_cell = person_states[track_id]['origin_cell']
                            dx = abs(current_cell[0] - origin_cell[0])
                            dy = abs(current_cell[1] - origin_cell[1])
                            if dx > MOVEMENT_TOLERANCE or dy > MOVEMENT_TOLERANCE:
                                # Person moved outside tolerance, reset origin and timer
                                person_states[track_id]['origin_cell'] = current_cell
                                person_states[track_id]['still_since'] = current_time_for_state
                                person_states[track_id]['osc_sent'] = False
                                # Also update sticky cell immediately
                                person_states[track_id]['sticky_cell'] = current_cell
                                person_states[track_id]['sticky_since'] = current_time_for_state
                            else:
                                # If within tolerance, check if they've stayed in a new cell long enough
                                if current_cell != person_states[track_id]['sticky_cell']:
                                    # Entered a new adjacent cell
                                    if person_states[track_id].get('sticky_candidate') == current_cell:
                                        # Already tracking this candidate cell
                                        if current_time_for_state - person_states[track_id].get('sticky_candidate_since', current_time_for_state) > STICKY_CELL_DURATION:
                                            # Stayed long enough, update sticky cell
                                            person_states[track_id]['sticky_cell'] = current_cell
                                            person_states[track_id]['sticky_since'] = current_time_for_state
                                            person_states[track_id].pop('sticky_candidate', None)
                                            person_states[track_id].pop('sticky_candidate_since', None)
                                    else:
                                        # Start tracking candidate cell
                                        person_states[track_id]['sticky_candidate'] = current_cell
                                        person_states[track_id]['sticky_candidate_since'] = current_time_for_state
                                else:
                                    # Reset candidate if back to sticky cell
                                    person_states[track_id].pop('sticky_candidate', None)
                                    person_states[track_id].pop('sticky_candidate_since', None)
                    person_states[track_id]['current_cell'] = current_cell
                    
                    # --- Visualization & Stillness Logic ---
                    is_still = (current_time_for_state - person_states[track_id]['still_since']) > STILLNESS_DURATION
                    if is_still:
                        still_cells.add(person_states[track_id]['sticky_cell'])

                    # Draw info on the frame only if window is visible
                    if show_window:
                        viz_color = (0, 255, 0) if is_still else (0, 255, 255)
                        label = f"ID {track_id}: ({grid_x}, {grid_y}) @ {distance_m:.2f}m"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, viz_color, 2)
                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

        # --- Timed OSC Sending & Visualization ---
        current_time = time.time()
        if current_time - last_osc_send_time > OSC_SEND_INTERVAL:
            # Use the unified still_cells set for OSC
            if still_cells:
                occupied_grid_cells = [coord for cell in sorted(list(still_cells)) for coord in cell]
                print(f"[INFO] Sending OSC message for STILL persons: {occupied_grid_cells}")
                osc_client.send_message(OSC_ADDRESS, occupied_grid_cells)
            # Prune old tracks
            person_states = {tid: state for tid, state in person_states.items() if tid in current_frame_ids}
            last_osc_send_time = current_time

        # --- Window Display and Control ---
        if show_window:
            # Display the main camera feed
            cv2.imshow(window_name, annotated_frame)
            
            # Create and display the grid visualization using ONLY the still cells
            grid_image = draw_grid_visualization(still_cells)
            cv2.imshow(grid_window_name, grid_image)

            # Check if window was closed by clicking the 'X'
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                show_window = False
                cv2.destroyAllWindows() # Close both windows
        
        key = cv2.waitKey(1) & 0xFF

        # 'q' to quit the entire application
        if key == ord('q'):
            print("[INFO] 'q' pressed, shutting down.")
            break
        
        # 'h' to hide/show the window
        if key == ord('h'):
            show_window = not show_window
            if not show_window:
                cv2.destroyAllWindows()

finally:
    # --- 4. Cleanup ---
    pipeline.stop()       # Stop the RealSense pipeline
    cv2.destroyAllWindows()
    print("[INFO] Pipeline stopped and resources released.")