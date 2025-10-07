import cv2
import numpy as np
import pyrealsense2 as rs # Intel RealSense SDK
from ultralytics import YOLO
from pythonosc import udp_client
import math
import time # Import the time module
import argparse # Add argparse for command-line arguments¨
import os
import pandas as pd
import torch

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
parser.add_argument("--rs-height", type=int, default=484, help="RealSense stream height in pixels")
parser.add_argument("--wall-idx-offset", type=int, default=0, help="Threshold for using opposite wall segment for projection")
parser.add_argument("--extra-wall", type=float, default=0, help="Extra wall length (in meters) to add to the longer end of the segmented wall")
parser.add_argument("--num-segments", type=int, default=11, help="Number of wall segments for tracking")
parser.add_argument("--projection_width", type=int, default=4800, help="Resolution width for wall projection")
parser.add_argument("--projection_height", type=int, default=1200, help="Resolution height for wall projection")
parser.add_argument("--sampling_height", type=float, default=0.25, help="Relative height (0.0-1.0) for wall segment sampling (0=top, 1=bottom)")
parser.add_argument("--calibrate-wall", action="store_true", help="Enable interactive wall calibration mode")
parser.add_argument("--replay-path", type=str, default=None, help="Folder with recorded RGB/depth frames for replay")
parser.add_argument("--osc-log", type=str, default=None, help="Path to log OSC output for evaluation")
parser.add_argument("--orientation-tracking", action="store_true", help="Enable orientation tracking using pose estimation")
parser.add_argument("--cone-angle", type=float, default=75, help="Cone angle in degrees for orientation-based wall segment assignment")
parser.add_argument("--occlusion-forgiveness", type=float, default=3.0, help="Seconds to retain an occluded person's state before removal.")
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

start_time = time.time()

# Configure the streams for the color and depth camera
# Choose a lower resolution (640x480) and standard FPS (30) for best performance
W, H = args.rs_width, args.rs_height
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 15)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 15)

# Start streaming
print("[INFO] Starting RealSense pipeline...")
profile = pipeline.start(config)
threshold_filter = rs.threshold_filter()
threshold_filter.set_option(rs.option.max_distance, 2.0) # 10 m maximum
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
# Load the ultra-efficient YOLOv8-L model
if torch.cuda.is_available():
    print("[INFO] Loading models onto GPU...")
    model = YOLO('yolov8l.pt').cuda()
    pose_model = None
    if args.orientation_tracking:
        pose_model = YOLO('yolov8l-pose.pt').cuda()
else:
    print("[INFO] Loading models onto CPU...")
    model = YOLO('yolov8l.pt')
    pose_model = None
    if args.orientation_tracking:
        pose_model = YOLO('yolov8l-pose.pt')


def get_facing_direction(keypoints, depth_frame, depth_intrinsics):
    """
    Estimate facing direction (unit vector) from pose keypoints in world coordinates.
    Uses nose and mid-hip for direction, applies camera tilt and yaw.
    """
    # Get pixel coordinates
    nose_px, nose_py = keypoints[0][:2]
    left_hip_px, left_hip_py = keypoints[11][:2]
    right_hip_px, right_hip_py = keypoints[12][:2]
    mid_hip_px = (left_hip_px + right_hip_px) / 2
    mid_hip_py = (left_hip_py + right_hip_py) / 2

    # Get depth for each keypoint
    nose_depth = depth_frame.get_distance(int(nose_px), int(nose_py))
    mid_hip_depth = depth_frame.get_distance(int(mid_hip_px), int(mid_hip_py))

    # Convert to world coordinates
    nose_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [nose_px, nose_py], nose_depth)
    mid_hip_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [mid_hip_px, mid_hip_py], mid_hip_depth)

    # Apply camera offset
    nose_3d[0] += CAMERA_X_OFFSET_M
    mid_hip_3d[0] += CAMERA_X_OFFSET_M

    # Project onto floor plane using tilt
    nose_vertical = CAMERA_HEIGHT_M - nose_3d[1]
    mid_hip_vertical = CAMERA_HEIGHT_M - mid_hip_3d[1]
    nose_floor_y = nose_vertical * math.tan(CAMERA_TILT_RADIANS) + nose_3d[2] * math.cos(CAMERA_TILT_RADIANS)
    mid_hip_floor_y = mid_hip_vertical * math.tan(CAMERA_TILT_RADIANS) + mid_hip_3d[2] * math.cos(CAMERA_TILT_RADIANS)
    nose_floor_x = nose_3d[0]
    mid_hip_floor_x = mid_hip_3d[0]

    # Apply yaw rotation
    nose_rot_x = nose_floor_x * math.cos(CAMERA_YAW_RADIANS) - nose_floor_y * math.sin(CAMERA_YAW_RADIANS)
    nose_rot_y = nose_floor_x * math.sin(CAMERA_YAW_RADIANS) + nose_floor_y * math.cos(CAMERA_YAW_RADIANS)
    mid_hip_rot_x = mid_hip_floor_x * math.cos(CAMERA_YAW_RADIANS) - mid_hip_floor_y * math.sin(CAMERA_YAW_RADIANS)
    mid_hip_rot_y = mid_hip_floor_x * math.sin(CAMERA_YAW_RADIANS) + mid_hip_floor_y * math.cos(CAMERA_YAW_RADIANS)

    # Facing vector in world coordinates (from mid-hip to nose)
    dx = nose_rot_x - mid_hip_rot_x
    dy = nose_rot_y - mid_hip_rot_y
    norm = math.hypot(dx, dy)
    if norm == 0:
        return (0, 1)
    return (dx / norm, dy / norm)


def is_wall_in_cone(person_pos, facing_vec, wall_segments, cone_angle_deg=args.cone_angle):
    """
    Returns the index of the nearest wall segment within the person's cone of vision.
    person_pos: (x, y) in world coordinates
    facing_vec: (dx, dy) unit vector
    wall_segments: list of (x, y)
    cone_angle_deg: cone angle in degrees
    """
    cone_angle_rad = math.radians(cone_angle_deg / 2)
    best_idx = None
    min_dist = float('inf')
    for idx, (wx, wy) in enumerate(wall_segments):
        if wx is None or wy is None:
            continue
        vec_to_wall = (wx - person_pos[0], wy - person_pos[1])
        dist = math.hypot(*vec_to_wall)
        if dist == 0:
            continue
        vec_to_wall_norm = (vec_to_wall[0] / dist, vec_to_wall[1] / dist)
        dot = facing_vec[0] * vec_to_wall_norm[0] + facing_vec[1] * vec_to_wall_norm[1]
        angle = math.acos(max(-1, min(1, dot)))
        if angle < cone_angle_rad and dist < min_dist:
            min_dist = dist
            best_idx = idx
    return best_idx

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


def calibrate_wall_points(pipeline, align, avg_duration=1.0):
    """
    Interactive calibration: user clicks on wall points in the color image.
    For each clicked pixel, averages depth over avg_duration seconds.
    Saves both pixel and world coordinates to wall_calibration.npz.
    """
    print("[INFO] Wall calibration mode: Click on wall points. Press 's' to save, 'q' to quit.")
    clicked_pixels = []
    world_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixels.append((x, y))
            print(f"[INFO] Clicked pixel: ({x}, {y})")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        display_img = color_image.copy()
        for px, py in clicked_pixels:
            cv2.circle(display_img, (px, py), 8, (0, 0, 255), -1)
        cv2.imshow("Calibrate Wall", display_img)
        cv2.setMouseCallback("Calibrate Wall", mouse_callback)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # For each clicked pixel, average depth over avg_duration seconds
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            for px, py in clicked_pixels:
                depths = []
                start_time = time.time()
                while time.time() - start_time < avg_duration:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    depth = depth_frame.get_distance(px, py)
                    if 0 < depth < 10:
                        depths.append(depth)
                if depths:
                    avg_depth = sum(depths) / len(depths)
                    pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [px, py], avg_depth)
                    pt_3d[0] += CAMERA_X_OFFSET_M
                    vertical_distance = CAMERA_HEIGHT_M - pt_3d[1]
                    floor_y = vertical_distance * math.tan(CAMERA_TILT_RADIANS) + pt_3d[2] * math.cos(CAMERA_TILT_RADIANS)
                    floor_x = pt_3d[0]
                    rotated_x = floor_x * math.cos(CAMERA_YAW_RADIANS) - floor_y * math.sin(CAMERA_YAW_RADIANS)
                    rotated_y = floor_x * math.sin(CAMERA_YAW_RADIANS) + floor_y * math.cos(CAMERA_YAW_RADIANS)
                    world_points.append((rotated_x, rotated_y))
                else:
                    world_points.append((None, None))
            np.savez("wall_calibration.npz", pixels=np.array(clicked_pixels), world=np.array(world_points))
            print(f"[INFO] Saved {len(world_points)} wall points to wall_calibration.npz")
            break
    cv2.destroyWindow("Calibrate Wall")

if args.calibrate_wall:
    calibrate_wall_points(pipeline, align)
    print("[INFO] Calibration complete. Exiting.")
    pipeline.stop()
    cv2.destroyAllWindows()
    exit(0)

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

def sample_wall_vertical_lines(depth_frame, depth_intrinsics, num_lines=20, min_height=0.1, max_height=0.9, samples_per_line=30):
    """
    Samples vertical lines across the image width.
    For each line (column), checks multiple heights and picks the furthest valid depth.
    """
    img_w = depth_intrinsics.width
    img_h = depth_intrinsics.height
    wall_curve = []
    wall_pixels = []
    for i in range(num_lines):
        x = int((img_w - 1) * (i / (num_lines - 1)))
        max_depth = 0
        best_y = None
        for j in range(samples_per_line):
            frac = min_height + (max_height - min_height) * (j / (samples_per_line - 1))
            y = int(img_h * frac)
            depth = depth_frame.get_distance(x, y)
            if 0 < depth < 10 and depth > max_depth:
                max_depth = depth
                best_y = y
        if max_depth > 0 and best_y is not None:
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, best_y], max_depth)
            wall_curve.append((pt_3d[0] + CAMERA_X_OFFSET_M, pt_3d[2]))
            wall_pixels.append((x, best_y))
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
    """Draws rectangles for wall segments, all with equal pixel width."""
    start_idx = WALL_IDX_OFFSET
    num_display_segments = NUM_SEGMENTS - start_idx
    width = args.projection_width
    height = 40

    # Calculate equal pixel width for each segment
    if num_display_segments > 0:
        segment_pixel_width = width // num_display_segments
    else:
        segment_pixel_width = width

    img = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = 0
    for i in range(num_display_segments):
        seg_idx = start_idx + i
        color = (0, 255, 0) if seg_idx in occupied_segments else (40, 40, 40)
        cv2.rectangle(img, (x_offset, 0), (x_offset + segment_pixel_width - 2, height - 2), color, -1)
        cv2.putText(img, str(seg_idx + 1), (x_offset + 10, height // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        x_offset += segment_pixel_width
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
        wall_curve, wall_pixels = sample_wall_vertical_lines(depth_frame, depth_intrinsics, num_points, sampling_height)
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

def compute_equal_segments(wall_curve, num_segments):
    """Divides the wall curve into num_segments segments of equal length."""
    # Filter out invalid points
    valid_points = [pt for pt in wall_curve if pt[0] is not None and pt[1] is not None]
    if len(valid_points) < 2:
        return [(None, None)] * num_segments

    # Compute cumulative distances along the curve
    distances = [0.0]
    for i in range(1, len(valid_points)):
        prev = valid_points[i-1]
        curr = valid_points[i]
        d = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        distances.append(distances[-1] + d)
    total_length = distances[-1]
    segment_length = total_length / (num_segments - 1)

    # Find segment points at equal intervals
    segment_points = []
    target_dist = 0.0
    idx = 0
    for seg in range(num_segments):
        while idx < len(distances) - 1 and distances[idx] < target_dist:
            idx += 1
        if idx == 0:
            segment_points.append(valid_points[0])
        else:
            # Linear interpolation between valid_points[idx-1] and valid_points[idx]
            prev_pt = valid_points[idx-1]
            curr_pt = valid_points[idx]
            prev_dist = distances[idx-1]
            curr_dist = distances[idx]
            if curr_dist == prev_dist:
                interp_pt = curr_pt
            else:
                ratio = (target_dist - prev_dist) / (curr_dist - prev_dist)
                interp_x = prev_pt[0] + ratio * (curr_pt[0] - prev_pt[0])
                interp_y = prev_pt[1] + ratio * (curr_pt[1] - prev_pt[1])
                interp_pt = (interp_x, interp_y)
            segment_points.append(interp_pt)
        target_dist += segment_length
    return segment_points

# --- Sample and average wall curve BEFORE tracking loop ---
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
depth_frame = aligned_frames.get_depth_frame()
depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

# --- Sample a dense wall curve (200 points) ---
dense_wall_curve, dense_wall_pixels = sample_wall_vertical_lines(
    depth_frame, depth_intrinsics, num_lines=200, min_height=args.sampling_height, max_height=args.sampling_height
)

# --- Divide the wall into equal-length segments ---
WALL_SEGMENTS = compute_equal_segments(dense_wall_curve, NUM_SEGMENTS)



def get_vertical_line_depths(depth_frame, x, y_start, y_end=0):
    """
    Returns a list of (y, depth) for all pixels in column x from y_start up to y_end (exclusive).
    y_start should be the wall segment pixel's y value, y_end is typically 0 (top of image).
    """
    depths = []
    for y in range(y_start, y_end - 1, -1):  # Go upwards
        depth = depth_frame.get_distance(x, y)
        depths.append((y, depth))
    return depths


# --- Map each dense measurement point to its segment ---
def assign_points_to_segments(dense_curve, segments):
    """Returns a list mapping each dense point index to its closest segment index."""
    mapping = []
    for wx, wy in dense_curve:
        min_dist = float('inf')
        min_idx = 0
        for idx, (sx, sy) in enumerate(segments):
            if sx is None or sy is None or wx is None or wy is None:
                continue
            dist = math.hypot(wx - sx, wy - sy)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        mapping.append(min_idx)
    return mapping

DENSE_POINT_TO_SEGMENT = assign_points_to_segments(dense_wall_curve, WALL_SEGMENTS)

# --- Helper: Find nearest dense measurement point ---
def nearest_dense_point_idx(px, py, dense_curve):
    min_dist = float('inf')
    min_idx = 0
    for idx, (wx, wy) in enumerate(dense_curve):
        if wx is None or wy is None:
            continue
        dist = math.hypot(px - wx, py - wy)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx
    return min_idx

# --- Tracking Loop ---
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


def get_vertical_line_world_points(depth_frame, depth_intrinsics, x, y_start, y_end=0):
    """
    Returns a list of (X, Y, Z) world coordinates for all pixels in column x from y_start up to y_end (exclusive).
    Applies camera tilt and yaw corrections.
    """
    points = []
    for y in range(y_start, y_end - 1, -1):  # Go upwards
        depth = depth_frame.get_distance(x, y)
        if 0 < depth < 10:
            pt_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
            # Apply camera offset
            pt_3d[0] += CAMERA_X_OFFSET_M
            # Project onto floor plane using tilt
            vertical_distance = CAMERA_HEIGHT_M - pt_3d[1]
            floor_y = vertical_distance * math.tan(CAMERA_TILT_RADIANS) + pt_3d[2] * math.cos(CAMERA_TILT_RADIANS)
            floor_x = pt_3d[0]
            # Apply yaw rotation
            rotated_x = floor_x * math.cos(CAMERA_YAW_RADIANS) - floor_y * math.sin(CAMERA_YAW_RADIANS)
            rotated_y = floor_x * math.sin(CAMERA_YAW_RADIANS) + floor_y * math.cos(CAMERA_YAW_RADIANS)
            points.append((rotated_x, rotated_y))
    return points

# --- Sample and average wall curve BEFORE tracking loop ---
frames = pipeline.wait_for_frames()
aligned_frames = align.process(frames)
depth_frame = aligned_frames.get_depth_frame()
depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics


if os.path.exists("wall_calibration.npz"):
    data = np.load("wall_calibration.npz")
    WALL_SEGMENTS = data["world"]
    WALL_SEGMENT_PIXELS = data["pixels"].tolist()
    NUM_SEGMENTS = len(WALL_SEGMENTS)  # Override segment count
    print(f"[INFO] Loaded {NUM_SEGMENTS} wall calibration points from wall_calibration.npz")
else:
    # Fallback: sample and average wall curve as before
    WALL_SEGMENTS, WALL_SEGMENT_PIXELS = average_wall_curve(
        pipeline, align, NUM_SEGMENTS, sample_duration=5.0, sampling_height=args.sampling_height
    )
    WALL_SEGMENTS = extend_wall_curve(WALL_SEGMENTS, EXTRA_WALL)


if args.osc_log:
    osc_log_file = open(args.osc_log, "w")
    osc_log_file.write("Frame_Num," + ",".join(str(i+1) for i in range(NUM_SEGMENTS - WALL_IDX_OFFSET)) + "\n")
else:
    osc_log_file = None
# --- Tracking Loop ---
try:
    if args.replay_path:
        rgb_files = sorted([f for f in os.listdir(args.replay_path) if f.startswith("rgb_") and f.endswith(".png")])
        depth_files = sorted([f for f in os.listdir(args.replay_path) if f.startswith("depth_") and f.endswith(".npy")])
        assert len(rgb_files) == len(depth_files), "Mismatch between RGB and depth files!"
        replay_idx = 0

    def get_next_frame():
        if args.replay_path:
            if replay_idx >= len(rgb_files):
                return None, None
            rgb_img = cv2.imread(os.path.join(args.replay_path, rgb_files[replay_idx]))
            depth_img = np.load(os.path.join(args.replay_path, depth_files[replay_idx]))
            return rgb_img, depth_img
        else:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return None, None
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            return color_image, depth_image

    frame_counter = 0
    while True:
        if args.replay_path:
            color_image, depth_image = get_next_frame()
            if color_image is None or depth_image is None:
                break
            else:
                frame_counter += 1
            replay_idx += 1
            # You may need to mock depth_frame/color_frame objects if you use RealSense API calls
            # Otherwise, adapt your code to use numpy arrays directly
        else:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
        # ...rest of your tracking code using color_image and depth_image...

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

        # If orientation tracking is enabled, run pose estimation
        pose_results = None
        if args.orientation_tracking and pose_model is not None:
            pose_results = pose_model.predict(color_image, conf=args.conf, imgsz=imgsz, verbose=False)

        # --- Annotation and Visualization ---
        # This part runs only if the window is supposed to be shown
        if show_window:
            annotated_frame = results[0].plot()
            # --- Draw vertical sampling lines and reference verticals ---
            keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
                
                # Iterate over each detected person's pose
            for person_keypoints in keypoints_data:
                    # Iterate over each keypoint for the person
                for x, y, conf in person_keypoints:
                    if conf > 0.5: # Draw only if confidence is above a threshold
                        cv2.circle(annotated_frame, (int(x), int(y)), 3, (255, 0, 255), -1) # Draw a small magenta circle
            for idx, (px, py) in enumerate(WALL_SEGMENT_PIXELS):
                if px is not None and py is not None:
                    # Draw a vertical line at each sampled column
                    cv2.line(annotated_frame, (px, 0), (px, annotated_frame.shape[0]), (0, 180, 255), 1)
                    # Draw the sampled point as before
                    cv2.circle(annotated_frame, (px, py), 6, (255, 0, 0), 2)
                    # Highlight active segments
                    if idx in still_segments:
                        cv2.circle(annotated_frame, (px, py), 14, (0, 255, 0), -1)
                        cv2.circle(annotated_frame, (px, py), 18, (0, 255, 255), 2)
                    else:
                        cv2.circle(annotated_frame, (px, py), 8, (0, 0, 255), 2)
                    # --- Draw all points above the wall segment pixel ---
                    vertical_depths = get_vertical_line_depths(depth_frame, px, py)
                    for y, depth in vertical_depths:
                        if 0 < depth < 10:
                            cv2.circle(annotated_frame, (px, y), 2, (0, 255, 255), -1)  # Yellow dots for reference line

        # Get a set of all IDs present in the current frame
        current_frame_ids = set()
        if results[0].boxes.id is not None:
            current_frame_ids = set(results[0].boxes.id.cpu().numpy().astype(int))
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, clss):
                if model.names[cls_id] == 'person':
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    distance_m = depth_frame.get_distance(cx, cy)
                    if 0 < distance_m < 10:
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], distance_m)
                        vertical_distance = CAMERA_HEIGHT_M - point_3d[1]
                        floor_y = vertical_distance * math.tan(CAMERA_TILT_RADIANS) + point_3d[2] * math.cos(CAMERA_TILT_RADIANS)
                        floor_x = point_3d[0] + CAMERA_X_OFFSET_M
                        rotated_x = floor_x * math.cos(CAMERA_YAW_RADIANS) - floor_y * math.sin(CAMERA_YAW_RADIANS)
                        rotated_y = floor_x * math.sin(CAMERA_YAW_RADIANS) + floor_y * math.cos(CAMERA_YAW_RADIANS)
                        grid_x = math.floor(rotated_x)
                        grid_y = math.floor(rotated_y)
                        current_frame_grid_cells.add((grid_x, grid_y))

                        # --- Update Person State for Stillness Detection ---
                        current_cell = (grid_x, grid_y)
                        current_time_for_state = time.time()

                        # Helper function to get segment index to avoid code repetition
                        def get_current_segment_idx(track_id, person_pos):
                            if args.orientation_tracking and pose_results is not None:
                                pose_keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
                                best_pose_idx = None
                                min_nose_dist = float('inf')
                                for i, kps in enumerate(pose_keypoints_data):
                                    nose_x, nose_y, _ = kps[0]
                                    dist = math.hypot(cx - nose_x, cy - nose_y)
                                    if dist < min_nose_dist:
                                        min_nose_dist = dist
                                        best_pose_idx = i
                                if best_pose_idx is not None:
                                    keypoints_with_conf = pose_keypoints_data[best_pose_idx]
                                    try:
                                        prev_stable_vec = person_states.get(track_id, {}).get('facing_vec')
                                        stable_vec = get_facing_direction(keypoints_with_conf, depth_frame, depth_intrinsics, prev_vec=prev_stable_vec)
                                        if track_id in person_states:
                                            person_states[track_id]['facing_vec'] = stable_vec
                                        segment_idx = is_wall_in_cone(person_pos, stable_vec, WALL_SEGMENTS)
                                        # If cone finds nothing, fall back to closest
                                        if segment_idx is None:
                                            segment_idx = closest_wall_segment(person_pos[0], person_pos[1])
                                        return segment_idx
                                    except Exception as e:
                                        print(f"[WARN] Error computing facing direction for ID {track_id}: {e}")
                                        return closest_wall_segment(person_pos[0], person_pos[1])
                                else:
                                    print(f"[WARN] No pose keypoints matched for ID {track_id}, using closest segment.")
                                    return closest_wall_segment(person_pos[0], person_pos[1])
                            else:
                                return closest_wall_segment(person_pos[0], person_pos[1])

                        if track_id not in person_states:
                            # New person detected, calculate initial segment
                            person_pos = (rotated_x, rotated_y)
                            closest_segment_idx = get_current_segment_idx(track_id, person_pos)

                            person_states[track_id] = {
                                'origin_cell': current_cell,
                                'current_cell': current_cell,
                                'sticky_cell': current_cell,
                                'sticky_since': current_time_for_state,
                                'still_since': current_time_for_state,
                                'osc_sent': False,
                                'segment': closest_segment_idx,
                                'facing_vec': None,
                                'last_seen': current_time_for_state
                            }
                        else:
                            # Existing person, check if they moved beyond tolerance
                            origin_cell = person_states[track_id]['origin_cell']
                            dx = abs(current_cell[0] - origin_cell[0])
                            dy = abs(current_cell[1] - origin_cell[1])
                            if dx > MOVEMENT_TOLERANCE or dy > MOVEMENT_TOLERANCE:
                                # Person moved outside tolerance, reset and re-evaluate segment
                                person_pos = (rotated_x, rotated_y)
                                closest_segment_idx = get_current_segment_idx(track_id, person_pos)
                                person_states[track_id]['segment'] = closest_segment_idx
                                person_states[track_id]['origin_cell'] = current_cell
                                person_states[track_id]['still_since'] = current_time_for_state
                                person_states[track_id]['osc_sent'] = False
                                person_states[track_id]['sticky_cell'] = current_cell
                            else:
                                # Person has not moved, do not update segment.
                                # Update facing vector if available
                                if args.orientation_tracking:
                                     get_current_segment_idx(track_id, (rotated_x, rotated_y))

                                # Sticky cell logic for minor drifts
                                if current_cell != person_states[track_id]['sticky_cell']:
                                    if person_states[track_id].get('sticky_candidate') == current_cell:
                                        if current_time_for_state - person_states[track_id].get('sticky_candidate_since', current_time_for_state) > STICKY_CELL_DURATION:
                                            person_states[track_id]['sticky_cell'] = current_cell
                                            person_states[track_id]['sticky_since'] = current_time_for_state
                                            person_states[track_id].pop('sticky_candidate', None)
                                            person_states[track_id].pop('sticky_candidate_since', None)
                                    else:
                                        person_states[track_id]['sticky_candidate'] = current_cell
                                        person_states[track_id]['sticky_candidate_since'] = current_time_for_state
                                else:
                                    person_states[track_id].pop('sticky_candidate', None)
                                    person_states[track_id].pop('sticky_candidate_since', None)

                        person_states[track_id]['current_cell'] = current_cell
                        person_states[track_id]['last_seen'] = current_time_for_state
                        
                        
                        # --- Visualization & Stillness Logic ---
                        is_still = (current_time_for_state - person_states[track_id]['still_since']) > STILLNESS_DURATION
                        if is_still:
                            still_segments.add(person_states[track_id]['segment'])
                            still_cells.add(current_cell)
                        # Draw info on the frame only if window is visible
                        if show_window:
                            current_segment = person_states[track_id].get('segment')
                            if current_segment is not None:
                                viz_color = (0, 255, 0) if is_still else (0, 255, 255)
                                label = f"ID {track_id}: seg {current_segment+1} @ {distance_m:.2f}m"
                                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, viz_color, 2)
                                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

        # --- Timed OSC Sending & Visualization ---
        current_time = time.time()
        if current_time - last_osc_send_time > OSC_SEND_INTERVAL:
            # --- Handle Occlusion and State Cleanup ---
            # Check for people who are still and those who are occluded but within forgiveness period
            active_person_states = {}
            for tid, state in person_states.items():
                is_occluded = tid not in current_frame_ids
                time_since_seen = current_time - state['last_seen']

                if not is_occluded:
                    # Person is visible, keep them
                    active_person_states[tid] = state
                elif is_occluded and time_since_seen < args.occlusion_forgiveness:
                    # Person is occluded, but within the forgiveness period. Keep them.
                    # We can also reset their stillness timer if we don't want occluded people to become "still"
                    # state['still_since'] = current_time 
                    active_person_states[tid] = state
                # else: person is occluded and forgiveness period has passed, so they are dropped.

            person_states = active_person_states
            
            # --- Determine still segments from the remaining active states ---
            still_segments.clear() # Clear before recalculating
            for tid, state in person_states.items():
                is_still = (current_time - state['still_since']) > STILLNESS_DURATION
                if is_still:
                    still_segments.add(state['segment'])


            osc_list = [1 if idx in still_segments else 0 for idx in range(WALL_IDX_OFFSET, NUM_SEGMENTS)]
            print(f"[INFO] Sending OSC message: {osc_list}")
            osc_client.send_message(OSC_ADDRESS, osc_list)
            if osc_log_file:
                osc_log_file.write(f"{frame_counter}," + ",".join(str(x) for x in osc_list) + "\n")
            last_osc_send_time = current_time

        if args.orientation_tracking and pose_results is not None and show_window:
            stopped_grid_image = draw_grid_visualization(still_cells)
            pose_keypoints = pose_results[0].keypoints.xy.cpu().numpy()  
            pose_keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
            if results[0].boxes.id is not None:
                for box, track_id, cls_id in zip(boxes, ids, clss):
                    if model.names[cls_id] == 'person':
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        best_pose_idx = None
                        min_nose_dist = float('inf')
                        for i, kps in enumerate(pose_keypoints_data):
                            nose_x, nose_y, _ = kps[0]
                            dist = math.hypot(cx - nose_x, cy - nose_y)
                            if dist < min_nose_dist:
                                min_nose_dist = dist
                                best_pose_idx = i
                        if best_pose_idx is not None:
                            keypoints_with_conf = pose_keypoints_data[best_pose_idx]
                            try:
                                # Use the smoothed vector from the person's state
                                facing_vec = person_states.get(track_id, {}).get('facing_vec')
                                if facing_vec is None:
                                    # Fallback if not available
                                    facing_vec = get_facing_direction(keypoints_with_conf, depth_frame, depth_intrinsics)
                            except Exception as e:
                                print(f"[ERROR] Facing vector calculation failed: {e}")
                                continue
                            # Use world coordinates for cone visualization
                            # Get world position of nose
                            nose_px, nose_py = keypoints_with_conf[0][:2]
                            if not (0 <= int(nose_px) < depth_intrinsics.width and 0 <= int(nose_py) < depth_intrinsics.height):
                                continue # Skip this person's cone visualization if nose is out of bounds
                            nose_depth = depth_frame.get_distance(int(nose_px), int(nose_py))
                            if nose_depth <= 0: # Also check for invalid depth
                                continue
                            nose_depth = depth_frame.get_distance(int(nose_px), int(nose_py))
                            nose_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [nose_px, nose_py], nose_depth)
                            nose_x = nose_3d[0] + CAMERA_X_OFFSET_M
                            nose_y = CAMERA_HEIGHT_M - nose_3d[1]
                            nose_z = nose_3d[2]
                            nose_floor_y = nose_y * math.tan(CAMERA_TILT_RADIANS) + nose_z * math.cos(CAMERA_TILT_RADIANS)
                            nose_rot_x = nose_x * math.cos(CAMERA_YAW_RADIANS) - nose_floor_y * math.sin(CAMERA_YAW_RADIANS)
                            nose_rot_y = nose_x * math.sin(CAMERA_YAW_RADIANS) + nose_floor_y * math.cos(CAMERA_YAW_RADIANS)
                            # Draw cone in grid visualization
                            cone_length = 5  # meters
                            cone_angle = args.cone_angle
                            angle_rad = math.atan2(facing_vec[1], facing_vec[0])
                            left_angle = angle_rad - math.radians(cone_angle / 2)
                            right_angle = angle_rad + math.radians(cone_angle / 2)
                            left_x = nose_rot_x + cone_length * math.cos(left_angle)
                            left_y = nose_rot_y + cone_length * math.sin(left_angle)
                            right_x = nose_rot_x + cone_length * math.cos(right_angle)
                            right_y = nose_rot_y + cone_length * math.sin(right_angle)
                            # Convert to grid pixels
                            center_pixel_x = GRID_PIXELS // 2
                            center_pixel_y = GRID_PIXELS // 2
                            nose_px_grid = int(center_pixel_x + nose_rot_x * CELL_PIXELS)
                            nose_py_grid = int(center_pixel_y - nose_rot_y * CELL_PIXELS)
                            left_px_grid = int(center_pixel_x + left_x * CELL_PIXELS)
                            left_py_grid = int(center_pixel_y - left_y * CELL_PIXELS)
                            right_px_grid = int(center_pixel_x + right_x * CELL_PIXELS)
                            right_py_grid = int(center_pixel_y - right_y * CELL_PIXELS)
                            pts = np.array([[nose_px_grid, nose_py_grid], [left_px_grid, left_py_grid], [right_px_grid, right_py_grid]], np.int32)
                            cv2.polylines(stopped_grid_image, [pts], isClosed=True, color=(255, 200, 0), thickness=2)

        # --- Window Display and Control ---
        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow("Wall Segments", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Movement Grid", cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, annotated_frame)
            wall_image = draw_wall_visualization(still_segments)
            cv2.imshow("Wall Segments", wall_image)
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
    if osc_log_file:
        osc_log_file.close()
    print("[INFO] Pipeline stopped and resources released.")

def seconds_to_mmss(seconds):
    seconds = float(seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

def evaluate_accuracy(osc_log_path, gt_path="Mock_Tracking_File.csv"):
    if not os.path.exists(osc_log_path) or not os.path.exists(gt_path):
        print("[WARNING] OSC log or ground truth file not found.")
        return
    osc_df = pd.read_csv(osc_log_path).fillna(0)
    gt_df = pd.read_csv(gt_path).fillna(0)
    # Use Frame_Num as index
    osc_df.set_index("Frame_Num", inplace=True)
    gt_df.set_index("Frame_Num", inplace=True)
    common_steps = osc_df.index.intersection(gt_df.index)
    if len(common_steps) == 0:
        print("[WARNING] No matching frame numbers between OSC log and ground truth.")
        print(f"OSC log frames: {list(osc_df.index)}")
        print(f"Ground truth frames: {list(gt_df.index)}")
        return
    osc_df = osc_df.loc[common_steps]
    gt_df = gt_df.loc[common_steps]
    # Align columns
    common_cols = osc_df.columns.intersection(gt_df.columns)
    if len(common_cols) == 0:
        print("[WARNING] No matching columns between OSC log and ground truth.")
        return
    osc_arr = osc_df[common_cols].values
    gt_arr = gt_df[common_cols].values
    matches = (osc_arr == gt_arr)
    accuracy = matches.mean() if matches.size > 0 else 0.0
    print(f"[RESULT] Evaluation accuracy: {accuracy:.3f}")

# After closing osc_log_file
if args.osc_log:
    evaluate_accuracy(args.osc_log)