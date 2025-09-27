# YOLOv8 RealSense Tracker

This project implements a YOLOv8 RealSense Tracker using Intel RealSense cameras and the YOLOv8 model for object detection. The tracker identifies and tracks people in a specified area, sending occupancy data via OSC (Open Sound Control).

## Project Structure

- `tracking.py`: Contains the main logic for the YOLOv8 RealSense Tracker, including argument parsing, camera setup, model initialization, tracking loop, and OSC communication.
- `Dockerfile`: Defines the environment for the application, specifying the base image, dependencies, and commands to run the application.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `README.md`: Documentation for the project, including setup instructions and usage.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd tracking-project
   ```

2. **Build the Docker Image**
   ```bash
   docker build -t tracking-project .
   ```

3. **Run the Docker Container**
   ```bash
   docker run --rm -it tracking-project
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