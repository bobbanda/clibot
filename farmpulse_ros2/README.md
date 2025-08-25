# Farmpulse ROS2 Intruder Detection System

A ROS2-based intruder detection system for farm security using YOLO object detection.

## Features

- Real-time person detection using YOLOv8
- Configurable detection zones (Region of Interest)
- Alert system with cooldown period
- Automatic image saving for detected intruders
- ROS2 topic publishing for integration with other systems
- Support for USB cameras and IP cameras (RTSP)

## System Requirements

- ROS2 Humble
- Python 3.10+
- OpenCV
- Ultralytics YOLO

## Installation

1. Clone this package into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
cp -r /workspace/farmpulse_ros2 .
```

2. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select farmpulse_ros2 --symlink-install
source install/setup.bash
```

## Usage

### Basic Usage

Run the intruder detection node:
```bash
ros2 run farmpulse_ros2 farmpulse_node
```

### Using Launch File

Run with custom parameters:
```bash
ros2 launch farmpulse_ros2 intruder_detection_launch.py camera_source:=/dev/video0
```

### Monitor Alerts

In a separate terminal, run the monitor to see alerts:
```bash
ros2 run farmpulse_ros2 intruder_monitor
```

## Configuration

Edit the configuration file at `config/intruder_detection.yaml`:

- `camera_source`: Camera device or RTSP URL
- `detection_confidence`: Minimum confidence for detection (0.0-1.0)
- `alert_threshold`: Minimum confidence to trigger alerts
- `cooldown_period`: Time between alerts (seconds)
- `roi_enabled`: Enable region of interest
- `roi_points`: Polygon points for ROI

## Topics

### Published Topics

- `/farmpulse/camera/image_raw` - Raw camera feed
- `/farmpulse/detection/image` - Annotated detection image
- `/farmpulse/camera/image_compressed` - Compressed detection image
- `/farmpulse/intruder/alert` - High-priority intruder alerts
- `/farmpulse/intruder/detections` - All detections data
- `/farmpulse/system/status` - System status updates

## Customization

### Using Different YOLO Models

- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

### Setting up Region of Interest

Define a polygon area to monitor:
```yaml
roi_enabled: true
roi_points: [100, 100, 500, 100, 500, 400, 100, 400]  # Rectangle example
```

### Using IP Cameras

For RTSP streams:
```yaml
camera_source: "rtsp://username:password@192.168.1.100:554/stream"
```

## Troubleshooting

1. **Camera not found**: Check device permissions and camera connection
2. **YOLO model download fails**: Ensure internet connection is available
3. **Low FPS**: Use a smaller YOLO model or reduce image resolution

## Integration Examples

### Email Alerts
Modify the `alert_callback` in `intruder_monitor.py` to send emails

### Database Logging
Add database connection to store detection history

### Cloud Upload
Integrate with cloud services to upload detection images