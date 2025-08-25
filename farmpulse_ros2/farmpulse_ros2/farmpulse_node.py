#!/usr/bin/env python3

import sys
import site
import cv2
import numpy as np
from datetime import datetime
import threading
import time

# Add your virtual environment site-packages path
sys.path.insert(0, '/home/ubuntu/farmpulse_projects/farmpulse_venv/lib/python3.10/site-packages')

# Now you can import ultralytics
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header, String, Bool
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import json

class IntruderDetection:
    """Class to hold intruder detection data"""
    def __init__(self, bbox, confidence, timestamp):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.timestamp = timestamp
        self.center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

class FarmPulseNode(Node):
    def __init__(self):
        super().__init__('farmpulse_node')
        
        # Declare parameters
        self.declare_parameter('camera_source', '/dev/video0')  # Can be video device or RTSP URL
        self.declare_parameter('detection_confidence', 0.5)
        self.declare_parameter('alert_threshold', 0.7)  # Higher confidence for alerts
        self.declare_parameter('cooldown_period', 30.0)  # Seconds between alerts
        self.declare_parameter('roi_enabled', False)  # Region of Interest
        self.declare_parameter('roi_points', [])  # List of points defining ROI polygon
        self.declare_parameter('save_detections', True)  # Save detection images
        self.declare_parameter('detection_folder', '/home/ubuntu/farmpulse_detections')
        self.declare_parameter('model_size', 'yolov8n.pt')  # n, s, m, l, x variants
        
        # Get parameters
        self.camera_source = self.get_parameter('camera_source').value
        self.detection_confidence = self.get_parameter('detection_confidence').value
        self.alert_threshold = self.get_parameter('alert_threshold').value
        self.cooldown_period = self.get_parameter('cooldown_period').value
        self.roi_enabled = self.get_parameter('roi_enabled').value
        self.roi_points = self.get_parameter('roi_points').value
        self.save_detections = self.get_parameter('save_detections').value
        self.detection_folder = self.get_parameter('detection_folder').value
        self.model_size = self.get_parameter('model_size').value
        
        self.get_logger().info(f"Farmpulse intruder detection node started")
        self.get_logger().info(f"Camera source: {self.camera_source}")
        self.get_logger().info(f"Detection confidence: {self.detection_confidence}")
        self.get_logger().info(f"Alert threshold: {self.alert_threshold}")
        
        # Initialize YOLO model
        try:
            self.model = YOLO(self.model_size)
            self.get_logger().info(f"YOLO model loaded: {self.model_size}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {str(e)}")
            raise
        
        # Initialize camera
        self.cap = None
        self.bridge = CvBridge()
        self.init_camera()
        
        # Publishers
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.image_pub = self.create_publisher(Image, 'farmpulse/camera/image_raw', qos_profile)
        self.detection_image_pub = self.create_publisher(Image, 'farmpulse/detection/image', qos_profile)
        self.compressed_image_pub = self.create_publisher(CompressedImage, 'farmpulse/camera/image_compressed', qos_profile)
        self.alert_pub = self.create_publisher(String, 'farmpulse/intruder/alert', 10)
        self.detection_pub = self.create_publisher(String, 'farmpulse/intruder/detections', 10)
        self.status_pub = self.create_publisher(String, 'farmpulse/system/status', 10)
        
        # Alert management
        self.last_alert_time = 0
        self.detection_count = 0
        self.false_positive_filter = []
        
        # Create detection folder if saving is enabled
        if self.save_detections:
            import os
            os.makedirs(self.detection_folder, exist_ok=True)
            self.get_logger().info(f"Saving detections to: {self.detection_folder}")
        
        # Start detection timer
        self.create_timer(0.1, self.detection_callback)  # 10 FPS
        
        # Status timer
        self.create_timer(5.0, self.publish_status)
        
        self.get_logger().info("Intruder detection system initialized successfully")
    
    def init_camera(self):
        """Initialize camera connection"""
        try:
            # Check if source is an integer (webcam index)
            if self.camera_source.isdigit():
                source = int(self.camera_source)
            else:
                source = self.camera_source
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {self.camera_source}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.get_logger().info("Camera initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"Camera initialization failed: {str(e)}")
            raise
    
    def detection_callback(self):
        """Main detection loop"""
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warn("Camera not available")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from camera")
            return
        
        # Publish raw image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw image: {str(e)}")
        
        # Run YOLO detection
        try:
            results = self.model(frame, conf=self.detection_confidence, classes=[0])  # class 0 is 'person'
            
            # Process detections
            detections = []
            detection_frame = frame.copy()
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        
                        # Check if detection is within ROI (if enabled)
                        if self.roi_enabled and self.roi_points:
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            if not self.point_in_roi(center_x, center_y):
                                continue
                        
                        # Create detection object
                        detection = IntruderDetection(
                            bbox=[x1, y1, x2, y2],
                            confidence=confidence,
                            timestamp=self.get_clock().now().nanoseconds
                        )
                        detections.append(detection)
                        
                        # Draw on frame
                        color = (0, 0, 255) if confidence > self.alert_threshold else (0, 255, 0)
                        cv2.rectangle(detection_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"Intruder: {confidence:.2f}"
                        cv2.putText(detection_frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ROI if enabled
            if self.roi_enabled and self.roi_points:
                self.draw_roi(detection_frame)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(detection_frame, timestamp, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add detection count
            self.detection_count += len(detections)
            cv2.putText(detection_frame, f"Total Detections: {self.detection_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Publish detection image
            detection_img_msg = self.bridge.cv2_to_imgmsg(detection_frame, "bgr8")
            detection_img_msg.header.stamp = self.get_clock().now().to_msg()
            self.detection_image_pub.publish(detection_img_msg)
            
            # Publish compressed image
            _, buffer = cv2.imencode('.jpg', detection_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.get_clock().now().to_msg()
            compressed_msg.format = "jpeg"
            compressed_msg.data = buffer.tobytes()
            self.compressed_image_pub.publish(compressed_msg)
            
            # Process detections for alerts
            if detections:
                self.process_detections(detections, detection_frame)
            
        except Exception as e:
            self.get_logger().error(f"Detection error: {str(e)}")
    
    def process_detections(self, detections, frame):
        """Process detections and send alerts if necessary"""
        current_time = time.time()
        
        # Publish detection data
        detection_data = {
            'timestamp': datetime.now().isoformat(),
            'detections': [
                {
                    'bbox': d.bbox,
                    'confidence': d.confidence,
                    'center': d.center
                }
                for d in detections
            ]
        }
        
        detection_msg = String()
        detection_msg.data = json.dumps(detection_data)
        self.detection_pub.publish(detection_msg)
        
        # Check for high-confidence detections that warrant an alert
        high_conf_detections = [d for d in detections if d.confidence > self.alert_threshold]
        
        if high_conf_detections and (current_time - self.last_alert_time) > self.cooldown_period:
            # Send alert
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'INTRUDER_DETECTED',
                'confidence': max(d.confidence for d in high_conf_detections),
                'num_intruders': len(high_conf_detections),
                'location': 'farm_perimeter'  # Could be enhanced with zone detection
            }
            
            alert_msg = String()
            alert_msg.data = json.dumps(alert_data)
            self.alert_pub.publish(alert_msg)
            
            self.last_alert_time = current_time
            self.get_logger().warn(f"INTRUDER ALERT: {len(high_conf_detections)} person(s) detected!")
            
            # Save detection image if enabled
            if self.save_detections:
                self.save_detection_image(frame, high_conf_detections)
    
    def save_detection_image(self, frame, detections):
        """Save detection image to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.detection_folder}/intruder_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.get_logger().info(f"Saved detection image: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save detection image: {str(e)}")
    
    def point_in_roi(self, x, y):
        """Check if a point is inside the ROI polygon"""
        if not self.roi_points or len(self.roi_points) < 6:  # Need at least 3 points (6 values)
            return True
        
        # Convert flat list to points
        points = []
        for i in range(0, len(self.roi_points), 2):
            points.append([self.roi_points[i], self.roi_points[i+1]])
        
        # Use OpenCV pointPolygonTest
        result = cv2.pointPolygonTest(np.array(points, dtype=np.int32), (x, y), False)
        return result >= 0
    
    def draw_roi(self, frame):
        """Draw ROI polygon on frame"""
        if not self.roi_points or len(self.roi_points) < 6:
            return
        
        # Convert flat list to points
        points = []
        for i in range(0, len(self.roi_points), 2):
            points.append([int(self.roi_points[i]), int(self.roi_points[i+1])])
        
        # Draw polygon
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
        
        # Add ROI label
        cv2.putText(frame, "ROI", (points[0][0], points[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def publish_status(self):
        """Publish system status"""
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'node_status': 'active',
            'camera_status': 'connected' if self.cap and self.cap.isOpened() else 'disconnected',
            'total_detections': self.detection_count,
            'model': self.model_size,
            'roi_enabled': self.roi_enabled
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)
    
    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = FarmPulseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()