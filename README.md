# YOLO RealSense Tracker

This project implements a YOLO RealSense Tracker using Intel RealSense cameras and the YOLOv8 or YOLOOv11 models for object detection. The tracker identifies and tracks people in a specified area, sending occupancy data via OSC (Open Sound Control).

---

## Installation

### 1. **Check CUDA Version and Install PyTorch**

Before installing other requirements, ensure you have the correct PyTorch version for your CUDA installation:

```bash
# Check your CUDA version
nvidia-smi
```

Then install PyTorch with the appropriate CUDA version from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

For example:
```bash

# For CUDA 12.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. **Install Other Requirements**

```bash
pip install -r requirements.txt
```

---

## Running `project_with_evals.py`

This script is an advanced version of the tracker with additional evaluation and configuration options.

### 1. **Run the Script**

```bash
python project_with_evals.py [arguments]
```

### 2. **Windows Startup Script**

For automatic startup on Windows, you can use the provided `startup.bat` file:

1. **Configure the batch file:**
   - Edit `startup.bat` and update the following variables:
     - `PROJECT_DIR`: Path to your project folder
     - `CONDA_ENV_NAME`: Name of your conda environment
   
2. **Test the script:**
   ```cmd
   startup.bat
   ```

3. **Add to Windows startup (optional):**
   - Copy `startup.bat` to your Windows Startup folder:
     - Press `Win + R`, type `shell:startup`, and press Enter
     - Copy the batch file to this folder

The batch file will automatically activate your conda environment, wait for system stabilization, and start the tracker with predefined arguments.

### 3. **Arguments and Their Explanations**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ip` | str | `127.0.0.1` | OSC server IP address |
| `--port` | int | `5005` | OSC server port |
| `--height` | float | `3.46` | Camera height in meters |
| `--offset` | float | `0.5` | Camera X-axis offset in meters |
| `--conf` | float | `0.4` | YOLO detection confidence threshold |
| `--no-video` | flag |  | Run in headless mode (no video output) |
| `--stillness` | float | `5.0` | Seconds a person must be still to trigger |
| `--tilt` | float | `50` | Camera tilt angle in degrees |
| `--tolerance` | int | `1` | Grid cell movement tolerance for stillness |
| `--rgb-exposure` | int | `1000` | RGB camera exposure (-1 for auto) |
| `--yaw` | float | `0` | Camera yaw angle in degrees (positive = right) |
| `--rs-width` | int | `640` | RealSense stream width in pixels |
| `--rs-height` | int | `484` | RealSense stream height in pixels |
| `--wall-idx-offset` | int | `0` | Threshold for using opposite wall segment for projection |
| `--extra-wall` | float | `0` | Extra wall length (meters) to add to the longer end |
| `--num-segments` | int | `11` | Number of wall segments for tracking |
| `--projection_width` | int | `4800` | Resolution width for wall projection |
| `--projection_height` | int | `1200` | Resolution height for wall projection |
| `--sampling_height` | float | `0.25` | Relative height (0.0-1.0) for wall segment sampling (0=top, 1=bottom) |
| `--calibrate-wall` | flag |  | Enable interactive wall calibration mode |
| `--replay-path` | str |  | Folder with recorded RGB/depth frames for replay |
| `--osc-log` | str |  | Path to log OSC output for evaluation |
| `--orientation-tracking` | flag |  | Enable orientation tracking using pose estimation |
| `--cone-angle` | float | `75` | Cone angle (degrees) for orientation-based wall segment assignment |
| `--occlusion-forgiveness` | float | `3.0` | Seconds to retain an occluded person's state before removal |
| `--top-n-segments` | int | `3` | Number of closest segments to consider before applying orientation filter |
| `--reverse-osc` | flag |  | Reverse OSC signal (0 becomes 1, 1 becomes 0) |
| `--project-all` | flag |  | Project to all wall segments within any vision cone (ignores stillness) |
| `--head-to-center-offset` | float | `0.0` | Meters to add to depth to approximate center of mass from head position |
| `--min-distance` | float | `0.0` | Minimum distance (meters) from wall segment to assign a person |
| `--max-distance` | float | `inf` | Maximum distance (meters) from wall segment to assign a person |
| `--decouple-segments` | str | `""` | Comma-separated list of segment indices to decouple (0-based) |
| `--decouple-min-distance` | float | `0.0` | Minimum distance (meters) for decoupled segments |
| `--decouple-max-distance` | float | `3.0` | Maximum distance (meters) for decoupled segments |
| `--decouple-forget-time` | float | `5.0` | Time (seconds) to forget a decoupled segment after person leaves range |
| `--decouple-zones` | str | `""` | Distance zones for decoupled segments, e.g. `'0:0.5-1.5,2.0-3.0;1:1.0-2.0'` |
| `--eval-file-path` | str | `Mock_Tracking_File.csv` | Path to ground truth evaluation CSV file |

**Flags** (no value needed):  
`--no-video`, `--calibrate-wall`, `--orientation-tracking`, `--reverse-osc`, `--project-all`

### 4. **Example Usage**

```bash
python project_with_evals.py --rs-height 480 --rs-width 640 --rgb-exposure 1000 --stillness 1.0 --tilt 50 --conf 0.2 --max-distance 2.0 --min-distance 0.3 --decouple-segment 1 --decouple-zones "1:0.1-2.0,6.0-8.0" --no-video
```

---

## Calibrating the Wall (`--calibrate-wall`)

Before running the tracker, you should calibrate the wall segments to match your physical setup. This ensures accurate mapping between the camera view and the real-world wall.

### **How to Calibrate**

1. **Start Calibration Mode:**

   ```bash
   python project_with_evals.py --calibrate-wall
   ```

2. **Calibration Steps:**
   - A window will open showing the camera's color image.
   - **Click** on the wall points in the image, in order, where you want to define the wall segments.
   - Each click will mark a calibration point.
   - When you have clicked all desired wall points, **press `s`** to save the calibration.
   - To exit without saving, **press `q`**.

3. **What Happens Next:**
   - The script will average the depth at each clicked point for a short time to improve accuracy.
   - The pixel and corresponding world coordinates are saved to `wall_calibration.npz`.
   - On future runs, the tracker will use this calibration file for accurate wall segment mapping.

**Tip:**  
For best results, click points evenly along the wall you want to track, starting from one end to the other.

---

## Recording Evaluation Data with `recorder.py`

You can record synchronized RGB and depth frames for evaluation or annotation using `recorder.py`.

### 1. **Start Recording**

```bash
python recorder.py
```

- Press **`r`** to start or stop recording frames.
- Press **`q`** to quit the recorder.
- Recorded frames are saved in the `recordings/` folder as `rgb_XXXXX.png` and `depth_XXXXX.npy`.

### 2. **Evaluate or Annotate Recorded Data**

To step through and visualize recorded frames (for annotation or inspection):

```bash
python recorder.py --eval-path recordings 
```

- Use the window to view RGB and depth images.
- Press **`q`** to quit evaluation mode.

---

### 3. **Evaluating Tracking Accuracy**
You can evaluate the tracking accuracy by comparing the OSC log file generated during tracking with a ground truth CSV file by setting the `--eval-file-path` argument. An example ground truth file is `Mock_Tracking_File.csv`.

```bash
python project_with_evals.py --osc-log osc_log.txt --eval-file-path Mock_Tracking_File.csv
```

## License

This project is licensed under the AGPL v3 License. See the LICENSE file for more details.