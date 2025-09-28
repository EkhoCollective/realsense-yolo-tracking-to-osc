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
parser.add_argument("--tilt", type=float, default=50, help="Camera tilt angle in degrees")
parser.add_argument("--tolerance", type=int, default=1, help="Grid cell movement tolerance for stillness")
parser.add_argument("--rgb-exposure", type=int, default=1000, help="RGB camera exposure value (-1 for auto)")
parser.add_argument("--yaw", type=float, default=-0, help="Camera yaw angle in degrees (positive = right)")
parser.add_argument("--rs-width", type=int, default=640, help="RealSense stream width in pixels")
parser.add_argument("--rs-height", type=int, default=480, help="RealSense stream height in pixels")
parser.add_argument("--wall-idx-offset", type=int, default=3, help="Threshold for using opposite wall segment for projection")
parser.add_argument("--extra-wall", type=float, default=1.0, help="Extra wall length (in meters) to add to the longer end of the segmented wall")
parser.add_argument("--num-segments", type=int, default=11, help="Number of wall segments for tracking")
parser.add_argument("--projection_width", type=int, default=4800, help="Resolution width for wall projection")
parser.add_argument("--projection_height", type=int, default=1200, help="Resolution height for wall projection")
parser.add_argument("--sampling_height", type=float, default=0.25, help="Relative height (0.0-1.0) for wall segment sampling (0=top, 1=bottom)")
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
STICKY_CELL_DURATION = 0.5 # seconds

WALL_IDX_OFFSET = args.wall_idx_offset

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
    """Creates an image representing the top-down grid view centered at (0,0), with wall curve."""
    grid_img = np.zeros((GRID_PIXELS, GRID_PIXELS, 3), dtype=np.uint8)
    
    # Draw grid lines
    for i in range(1, GRID_DIM_METERS):
        pos = i * CELL_PIXELS
        cv2.line(grid_img, (pos, 0), (pos, GRID_PIXELS), (40, 40, 40), 1)
        cv2.line(grid_img, (0, pos), (GRID_PIXELS, pos), (40, 40, 40), 1)

    # Center (0,0) in the middle of the grid image
    center_pixel_x = GRID_PIXELS // 2
    center_pixel_y = GRID_PIXELS // 2

    # Draw occupied cells
    for x, y in occupied_cells:
        px = center_pixel_x + x * CELL_PIXELS
        py = center_pixel_y - y * CELL_PIXELS  # Flip Y axis
        if 0 <= px < GRID_PIXELS and 0 <= py < GRID_PIXELS:
            cv2.rectangle(grid_img, (px, py), (px + CELL_PIXELS, py + CELL_PIXELS),
                          (0, 255, 0), -1) # Draw a filled green square

    # --- Draw wall curve ---
    wall_pts = []
    for wx, wy in WALL_SEGMENTS:
        if wx is None or wy is None:
            continue  # Skip invalid wall points
        px = center_pixel_x + int(wx * CELL_PIXELS)
        py = center_pixel_y - int(wy * CELL_PIXELS)
        wall_pts.append((px, py))
        cv2.circle(grid_img, (px, py), 5, (0, 0, 255), -1)
    for i in range(1, len(wall_pts)):
        cv2.line(grid_img, wall_pts[i-1], wall_pts[i], (0, 0, 255), 2)

    return grid_img

# --- Wall Definition ---
# Wall: straight from (-3, 11) to (-3, 12.5), then quadratic curve to (0, 14)
NUM_SEGMENTS = args.num_segments

def sample_wall_curve(depth_frame, depth_intrinsics, num_points=200):
    """Samples depth along a line in the image to estimate the wall curve."""
    img_w = depth_intrinsics.width
    img_h = depth_intrinsics.height
    y = img_h // 4  # Top quarter of the image
    wall_curve = []
    for i in range(num_points):
        x = int((img_w - 1) * (i / (num_points - 1)))  # Ensure x is in [0, img_w-1]
        depth = depth_frame.get_distance(x, y)
        if 0 < depth < 10:  # Valid depth
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
            wall_curve.append((pt_3d[0] + CAMERA_X_OFFSET_M, pt_3d[2]))  # (X, Z) as (x, y) in room
        else:
            wall_curve.append((None, None))  # Mark invalid
    return wall_curve

def sample_room_edges(depth_frame, depth_intrinsics, num_points=11):
    """Samples depth along all four edges of the image to estimate the room shape."""
    img_w = depth_intrinsics.width
    img_h = depth_intrinsics.height
    wall_curve = []

    # Top edge
    y_top = 0
    for i in range(num_points):
        x = int((img_w - 1) * (i / (num_points - 1)))
        depth = depth_frame.get_distance(x, y_top)
        if 0 < depth < 10:
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y_top], depth)
            wall_curve.append((pt_3d[0] + CAMERA_X_OFFSET_M, pt_3d[2]))
        else:
            wall_curve.append((None, None))

    # Right edge
    x_right = img_w - 1
    for i in range(num_points):
        y = int((img_h - 1) * (i / (num_points - 1)))
        depth = depth_frame.get_distance(x_right, y)
        if 0 < depth < 10:
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_right, y], depth)
            wall_curve.append((pt_3d[0] + CAMERA_X_OFFSET_M, pt_3d[2]))
        else:
            wall_curve.append((None, None))

    # Bottom edge
    y_bottom = img_h - 1
    for i in range(num_points):
        x = int((img_w - 1) * (1 - i / (num_points - 1)))  # Reverse direction for continuity
        depth = depth_frame.get_distance(x, y_bottom)
        if 0 < depth < 10:
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y_bottom], depth)
            wall_curve.append((pt_3d[0] + CAMERA_X_OFFSET_M, pt_3d[2]))
        else:
            wall_curve.append((None, None))

    # Left edge
    x_left = 0
    for i in range(num_points):
        y = int((img_h - 1) * (1 - i / (num_points - 1)))  # Reverse direction for continuity
        depth = depth_frame.get_distance(x_left, y)
        if 0 < depth < 10:
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_left, y], depth)
            wall_curve.append((pt_3d[0] + CAMERA_X_OFFSET_M, pt_3d[2]))
        else:
            wall_curve.append((None, None))

    return wall_curve

def sample_furthest_points(depth_frame, depth_intrinsics, num_points=20, sampling_height=0.25):
    """Samples the furthest valid depth points across the image width at a specific row."""
    img_w = depth_intrinsics.width
    img_h = depth_intrinsics.height
    y = int(img_h * sampling_height)
    wall_curve = []
    wall_pixels = []
    for i in range(num_points):
        x = int((img_w - 1) * (i / (num_points - 1)))
        depth = depth_frame.get_distance(x, y)
        if 0 < depth < 10:
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
            wall_curve.append((pt_3d[0] + CAMERA_X_OFFSET_M, pt_3d[2]))
            wall_pixels.append((x, y))
        else:
            wall_curve.append((None, None))
            wall_pixels.append((None, None))
    return wall_curve, wall_pixels

def closest_wall_segment(px, py):
    """Returns the index of the closest valid wall segment to (px, py)."""
    min_dist = float('inf')
    min_idx = 0
    for idx, (wx, wy) in enumerate(WALL_SEGMENTS):
        if wx is None or wy is None:
            continue  # Skip invalid wall points
        dist = math.hypot(px - wx, py - wy)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx
    return min_idx

def draw_wall_visualization(occupied_segments):
    """Draws a horizontal row of rectangles for wall segments after wall-idx-offset, highlighting occupied segments."""
    start_idx = WALL_IDX_OFFSET + 1
    num_display_segments = NUM_SEGMENTS - start_idx
    width = args.projection_width
    height = 40
    seg_w = width // num_display_segments if num_display_segments > 0 else width
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(num_display_segments):
        seg_idx = start_idx + i
        color = (0, 255, 0) if seg_idx in occupied_segments else (40, 40, 40)
        cv2.rectangle(img, (i * seg_w, 0), ((i + 1) * seg_w - 2, height - 2), color, -1)
        cv2.putText(img, str(seg_idx + 1), (i * seg_w + 10, height // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return img

def average_wall_curve(pipeline, align, num_points=20, sample_duration=5.0, sampling_height=0.25):
    """Samples the furthest wall points for sample_duration seconds and averages them."""
    print("[INFO] Sampling wall segments for averaging...")
    start_time = time.time()
    samples = [[] for _ in range(num_points)]
    pixel_samples = [[] for _ in range(num_points)]
    while time.time() - start_time < sample_duration:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        wall_curve, wall_pixels = sample_furthest_points(depth_frame, depth_intrinsics, num_points, sampling_height)
        for i, ((wx, wy), (px, py)) in enumerate(zip(wall_curve, wall_pixels)):
            if wx is not None and wy is not None:
                samples[i].append((wx, wy))
            if px is not None and py is not None:
                pixel_samples[i].append((px, py))
    averaged_curve = []
    averaged_pixels = []
    for seg_samples, pix_samples in zip(samples, pixel_samples):
        if seg_samples:
            avg_x = sum(x for x, y in seg_samples) / len(seg_samples)
            avg_y = sum(y for x, y in seg_samples) / len(seg_samples)
            averaged_curve.append((avg_x, avg_y))
        else:
            averaged_curve.append((None, None))
        if pix_samples:
            avg_px = int(sum(x for x, y in pix_samples) / len(pix_samples))
            avg_py = int(sum(y for x, y in pix_samples) / len(pix_samples))
            averaged_pixels.append((avg_px, avg_py))
        else:
            averaged_pixels.append((None, None))
    print("[INFO] Wall segments averaged and frozen for tracking.")
    return averaged_curve, averaged_pixels

# --- 3. Tracking Loop ---
# Variables for timed OSC sending
last_osc_send_time = time.time()
# New dictionary to track the state of each person
person_states = {} 

# --- Visualization State ---
show_window = not args.no_video
window_name = "YOLOv8 ByteTrack on RealSense"
grid_window_name = "Occupancy Grid"

EXTRA_WALL = args.extra_wall

def extend_wall_curve(wall_curve, extra_length=1.0):
    """Extends the wall curve by adding a straight segment to the longer end."""
    # Find the last valid segment
    for i in reversed(range(len(wall_curve))):
        wx, wy = wall_curve[i]
        if wx is not None and wy is not None:
            last_valid = (wx, wy)
            break
    else:
        return wall_curve  # No valid segment, return as is

    # Estimate direction of last segment (straight extension)
    # If there are at least two valid points, use their vector
    for j in reversed(range(i)):
        wx2, wy2 = wall_curve[j]
        if wx2 is not None and wy2 is not None:
            dx = last_valid[0] - wx2
            dy = last_valid[1] - wy2
            norm = math.hypot(dx, dy)
            if norm == 0:
                dx, dy = 0, 1  # Default direction
            else:
                dx /= norm
                dy /= norm
            break
    else:
        dx, dy = 0, 1  # Default direction if only one valid point

    # Add extra segment
    extended = wall_curve.copy()
    extra_point = (last_valid[0] + dx * extra_length, last_valid[1] + dy * extra_length)
    extended.append(extra_point)
    return extended

# --- Sample and average wall curve BEFORE tracking loop ---
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
depth_frame = aligned_frames.get_depth_frame()
depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

WALL_SEGMENTS, WALL_SEGMENT_PIXELS = average_wall_curve(
    pipeline, align, NUM_SEGMENTS, sample_duration=5.0, sampling_height=args.sampling_height
)
WALL_SEGMENTS = extend_wall_curve(WALL_SEGMENTS, EXTRA_WALL)
print("[INFO] Wall segments frozen for tracking.")

# --- Tracking Loop ---
try:
    while True:
        # Initialize a set for the current frame's grid cells for visualization
        current_frame_grid_cells = set()
        # Unified set for cells of people confirmed to be "still"
        still_cells = set()
        still_segments = set()  # Use this for wall segments

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
            # Highlight wall segment pixels
            for px, py in WALL_SEGMENT_PIXELS:
                if px is not None and py is not None:
                    cv2.circle(annotated_frame, (px, py), 8, (0, 0, 255), 2)  # Red circles
        else:
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

                        # --- Wall segment calculation ---
                        wall_idx = closest_wall_segment(rotated_x, rotated_y)

                        # Heuristic: Use opposite wall segment if projection is longer on that side
                        projection_wall_idx = wall_idx
                        if wall_idx <= WALL_IDX_OFFSET:
                            projection_wall_idx = NUM_SEGMENTS - wall_idx - 1

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
                            still_segments.add(projection_wall_idx)
                            still_cells.add(current_cell)

                        # Draw info on the frame only if window is visible
                        if show_window:
                            viz_color = (0, 255, 0) if is_still else (0, 255, 255)
                            label = f"ID {track_id}: seg {wall_idx+1} @ {distance_m:.2f}m"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, viz_color, 2)
                            cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

        # --- Timed OSC Sending & Visualization ---
        current_time = time.time()
        if current_time - last_osc_send_time > OSC_SEND_INTERVAL:
            if still_segments:
                occupied_segments = sorted(list(still_segments))
                print(f"[INFO] Sending OSC message for STILL persons: {occupied_segments}")
                osc_client.send_message(OSC_ADDRESS, occupied_segments)
            person_states = {tid: state for tid, state in person_states.items() if tid in current_frame_ids}
            last_osc_send_time = current_time

        # --- Window Display and Control ---
        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow("Wall Segments", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Movement Grid", cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, annotated_frame)
            wall_image = draw_wall_visualization(still_segments)
            cv2.imshow("Wall Segments", wall_image)
            stopped_grid_image = draw_grid_visualization(still_cells)
            cv2.imshow("Movement Grid", stopped_grid_image)
            
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                show_window = False
                cv2.destroyAllWindows()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] 'q' pressed, shutting down.")
            break
        if key == ord('h'):
            show_window = not show_window
            if not show_window:
                cv2.destroyAllWindows()

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline stopped and resources released.")