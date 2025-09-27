# YOLOv8 RealSense Tracker

This project implements a YOLOv8 RealSense Tracker using Intel RealSense cameras and the YOLOv8 model for object detection. The tracker identifies and tracks people in a specified area, sending occupancy data via OSC (Open Sound Control).

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/EkhoCollective/realsense-yolo-tracking-to-osc.git
   cd realsense-yolo-tracking-to-osc
   ```

2. **Build the Docker Image**
Note that USB passthrough may require additional configuration on Windows hosts.
   ```bash
   docker build -t realsense-tracking-osc .
   ```

3. **Run the Docker Container**
   ```bash
   docker run --rm -it realsense-tracking-osc
   ```

## Usage

To run the tracker, you can specify various command-line arguments. Here are some common options:

- `--ip`: OSC server IP (default: `127.0.0.1`)
- `--port`: OSC server port (default: `5005`)
- `--height`: Camera height in meters (default: `3.46`)
- `--offset`: Camera X-axis offset in meters (default: `0.5`)
- `--conf`: YOLO detection confidence threshold (default: `0.4`)
- `--no-video`: Run in headless mode without video output.
- `--stillness`: How long a person must be still (in seconds) (default: `5.0`)
- `--tilt`: Camera tilt angle in degrees (default: `64.4`)
- `--tolerance`: Grid cell movement tolerance for stillness (default: `1`)
- `--rgb-exposure`: RGB camera exposure value (-1 for auto, default: `1000`)
- `--yaw`: Camera yaw angle in degrees (positive = right, default: `10.0`)

## Dependencies

The project requires the following Python packages:

- `opencv-python`
- `numpy`
- `pyrealsense2`
- `ultralytics`
- `python-osc`

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the GPL v3 License. See the LICENSE file for more details.