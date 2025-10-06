import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os
import argparse

parser = argparse.ArgumentParser(description="RealSense Recorder & Evaluator")
parser.add_argument("--eval-path", type=str, default=None, help="Path to folder containing annotated data for evaluation")
args = parser.parse_args()

SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

def record_mode():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    recording = False
    frame_count = 0

    print("[INFO] Press 'r' to start/stop recording, 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            display_img = color_image.copy()
            if recording:
                cv2.putText(display_img, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                # Save RGB and depth frames
                rgb_path = os.path.join(SAVE_DIR, f"rgb_{frame_count:05d}.png")
                depth_path = os.path.join(SAVE_DIR, f"depth_{frame_count:05d}.npy")
                cv2.imwrite(rgb_path, color_image)
                np.save(depth_path, depth_image)
                frame_count += 1

            cv2.imshow("Recorder", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                recording = not recording
                print("[INFO] Recording:", recording)
            if key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped and resources released.")

def evaluate_mode(eval_path):
    print(f"[INFO] Evaluating annotated data in: {eval_path}")
    rgb_files = sorted([f for f in os.listdir(eval_path) if f.startswith("rgb_") and f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(eval_path) if f.startswith("depth_") and f.endswith(".npy")])
    assert len(rgb_files) == len(depth_files), "Mismatch between RGB and depth files!"

    for idx, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        rgb_img = cv2.imread(os.path.join(eval_path, rgb_file))
        depth_img = np.load(os.path.join(eval_path, depth_file))
        # Example: visualize, print stats, or annotate
        cv2.imshow("RGB", rgb_img)
        depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imshow("Depth", depth_vis)
        print(f"[INFO] Frame {idx}: {rgb_file}, {depth_file}, Depth min/max: {depth_img.min():.2f}/{depth_img.max():.2f}")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if args.eval_path:
    evaluate_mode(args.eval_path)
else:
    record_mode()