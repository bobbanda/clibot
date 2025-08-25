#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
import cv2
from cv_bridge import CvBridge
from datetime import datetime

class IntruderMonitor(Node):
    """Simple monitoring node to display alerts and detections"""
    
    def __init__(self):
        super().__init__('intruder_monitor')
        
        # Subscribers
        self.alert_sub = self.create_subscription(
            String,
            'farmpulse/intruder/alert',
            self.alert_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            String,
            'farmpulse/intruder/detections',
            self.detection_callback,
            10
        )
        
        self.status_sub = self.create_subscription(
            String,
            'farmpulse/system/status',
            self.status_callback,
            10
        )
        
        # For displaying images (optional)
        self.image_sub = self.create_subscription(
            Image,
            'farmpulse/detection/image',
            self.image_callback,
            10
        )
        
        self.bridge = CvBridge()
        self.display_images = False  # Set to True to display detection images
        
        self.get_logger().info("Intruder Monitor started - listening for alerts...")
        
    def alert_callback(self, msg):
        """Handle intruder alerts"""
        try:
            alert_data = json.loads(msg.data)
            
            # Print alert with formatting
            self.get_logger().error("="*60)
            self.get_logger().error("🚨 INTRUDER ALERT! 🚨")
            self.get_logger().error(f"Time: {alert_data['timestamp']}")
            self.get_logger().error(f"Confidence: {alert_data['confidence']:.2%}")
            self.get_logger().error(f"Number of intruders: {alert_data['num_intruders']}")
            self.get_logger().error(f"Location: {alert_data['location']}")
            self.get_logger().error("="*60)
            
            # Here you could add:
            # - Send SMS/Email notifications
            # - Trigger alarm system
            # - Save to database
            # - Send to cloud service
            
        except Exception as e:
            self.get_logger().error(f"Error processing alert: {str(e)}")
    
    def detection_callback(self, msg):
        """Handle detection data"""
        try:
            detection_data = json.loads(msg.data)
            num_detections = len(detection_data['detections'])
            
            if num_detections > 0:
                self.get_logger().info(f"[{detection_data['timestamp']}] Detected {num_detections} person(s)")
                
                for i, det in enumerate(detection_data['detections']):
                    self.get_logger().debug(
                        f"  Person {i+1}: Confidence={det['confidence']:.2f}, "
                        f"Position=({det['center'][0]:.0f}, {det['center'][1]:.0f})"
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Error processing detection: {str(e)}")
    
    def status_callback(self, msg):
        """Handle system status updates"""
        try:
            status_data = json.loads(msg.data)
            
            # Log status periodically
            self.get_logger().info(
                f"System Status - Camera: {status_data['camera_status']}, "
                f"Total Detections: {status_data['total_detections']}, "
                f"Model: {status_data['model']}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error processing status: {str(e)}")
    
    def image_callback(self, msg):
        """Display detection images (optional)"""
        if self.display_images:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imshow("Intruder Detection", cv_image)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().error(f"Error displaying image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    monitor = IntruderMonitor()
    
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()