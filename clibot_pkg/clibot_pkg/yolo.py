import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np
import colorsys
import random
import sys
sys.path.append('/usr/lib/python3.10/site-packages')
sys.path.append('/usr/local/share/pynq-venv/lib/python3.10/site-packages')
from pynq_dpu import DpuOverlay
import matplotlib.pyplot as plt
import cvzone
import math

class YOLODPUNode(Node):
    def __init__(self):
        super().__init__('yolo_dpu_node')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
		
        # Load DPU overlay and model
        self.overlay = DpuOverlay("dpu.bit")
        self.overlay.load_model('/root/jupyter_notebooks/pynq-dpu/tf_yolov3_voc.xmodel')

        # Anchors and class names
        anchor_list = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
        anchor_float = [float(x) for x in anchor_list]
        self.anchors = np.array(anchor_float).reshape(-1, 2)
        
        classes_path = '/root/jupyter_notebooks/pynq-dpu/img/voc_classes.txt'
        self.class_names = self.get_class(classes_path)

        # Prepare colors for bounding boxes
        num_classes = len(self.class_names)
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(0)
        random.shuffle(self.colors)
        random.seed(None)

        # Initialize DPU runner
        self.dpu = self.overlay.runner
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()
        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut0 = (tuple(outputTensors[0].dims))
        self.shapeOut1 = (tuple(outputTensors[1].dims))
        self.shapeOut2 = (tuple(outputTensors[2].dims))
        self.outputSize0 = int(outputTensors[0].get_data_size() / self.shapeIn[0])
        self.outputSize1 = int(outputTensors[1].get_data_size() / self.shapeIn[0])
        self.outputSize2 = int(outputTensors[2].get_data_size() / self.shapeIn[0])

        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        self.output_data = [np.empty(self.shapeOut0, dtype=np.float32, order="C"), 
                       np.empty(self.shapeOut1, dtype=np.float32, order="C"),
                       np.empty(self.shapeOut2, dtype=np.float32, order="C")]
        self.image = self.input_data[0]

        # Publisher for controlling TurtleSim
        self.cmd_vel_publisher = self.create_publisher(Twist, 'turtle1/cmd_vel', 10)

    def get_class(self, classes_path):
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def letterbox_image(self, image, size):
        ih, iw, _ = image.shape
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.ones((h,w,3), np.uint8) * 128
        h_start = (h-nh)//2
        w_start = (w-nw)//2
        new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
        return new_image

    def pre_process(self, image, model_image_size):
        image = image[...,::-1]
        image_h, image_w, _ = image.shape
        if model_image_size != (None, None):
            assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = self.letterbox_image(image, tuple(reversed(model_image_size)))
        else:
            new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
            boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0) 	
        return image_data

    def _get_feats(self, feats, anchors, num_classes, input_shape):
        num_anchors = len(anchors)
        anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
        grid_size = np.shape(feats)[1:3]
        nu = num_classes + 5
        predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
        grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = np.concatenate([grid_x, grid_y], axis = -1)
        grid = np.array(grid, dtype=np.float32)

        box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
        box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
        box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
        box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
        return box_xy, box_wh, box_confidence, box_class_probs

    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape, dtype=np.float32)
        image_shape = np.array(image_shape, dtype=np.float32)
    
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
    
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
    
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[..., 0:1],  
            box_mins[..., 1:2],  
            box_maxes[..., 0:1],  
            box_maxes[..., 1:2]  
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape])
        return boxes

    def boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, num_classes, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = np.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = np.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores

    def draw_bbox(self, image, bboxes):
            num_classes = len(self.class_names)
            image_h, image_w, _ = image.shape
            hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
            no_objects_detected = True
            
            for bbox in bboxes:
                no_objects_detected = False  # Objects are detected
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                bbox_thick = int(1.0 * (image_h + image_w) / 600)
                c1, c2 = (coor[0], coor[1]), (coor[1], coor[0]) 
        
                # Draw bounding box and label
                img = cvzone.cornerRect(
                    image,  # The image to draw on
                    (coor[0], coor[1], coor[3], coor[2]),  # The position and dimensions of the rectangle (x, y, width, height)
                    l=30,  # Length of the corner edges
                    t=5,  # Thickness of the corner edges
                    rt=1,  # Thickness of the rectangle
                    colorR=(255, 0, 255),  # Color of the rectangle
                    colorC=(0, 255, 0)  # Color of the corner edges
                )
                color = (255, 0, 255)
                scale = 5
                img, bbox = cvzone.putTextRect(
                    image, 
                    '{}: {:.2f}'.format(self.class_names[class_ind], score),
                    (coor[0], coor[1] - 20),  # Image and starting position of the rectangle
                    scale=2, 
                    thickness=2,  # Font scale and thickness
                    colorT=(255, 255, 255), colorR=(255, 0, 255),  # Text color and Rectangle color
                    font=cv2.FONT_HERSHEY_PLAIN,  # Font type
                    offset=10,  # Offset of text inside the rectangle
                    border=2, colorB=(0, 255, 0)  # Border thickness and color
                )
                
                # Calculate distance from the center of the object to the bottom-left corner of the window
                x1, y1 = c1
                x2, y2 = (0, image_h)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                length = math.hypot(x2 - x1, y2 - y1)
        
                # Draw distance and control TurtleSim
                if img is not None:
                    cvzone.putTextRect(
                        image, 
                        '{}'.format(int(length)),
                        (cx, cy - 10),  # Image and starting position of the rectangle
                        scale=1, 
                        thickness=1,  # Font scale and thickness
                        colorT=(255, 255, 255), 
                        colorR=(255, 0, 255),
                        font=cv2.FONT_HERSHEY_PLAIN,  # Font type
                        offset=10,  # Offset of text inside the rectangle
                        border=2, 
                        colorB=(0, 255, 0)  # Border thickness and color
                    )
                    cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
                    cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
                    cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)
        
                # Stop movement if the distance is below a threshold (e.g., 100 pixels)
                if length < 300:
                    self.get_logger().info('Object too close: Stopping TurtleSim')
                    msg = Twist()
                    msg.linear.x = 0.0
                    msg.linear.y = 0.0
                    msg.linear.z = 0.0
                    msg.angular.x = 0.0
                    msg.angular.y = 0.0
                    msg.angular.z = 0.0
                    self.cmd_vel_publisher.publish(msg)
                else:
                    self.get_logger().info('Object far away: Moving TurtleSim')
                    msg = Twist()
                    msg.linear.x = 2.0
                    msg.linear.y = 0.0
                    msg.linear.z = 0.0
                    msg.angular.x = 0.0
                    msg.angular.y = 0.0
                    msg.angular.z = 0.0
                    self.cmd_vel_publisher.publish(msg)
        
            # If no objects detected, move forward
            if no_objects_detected:
                self.get_logger().info('No objects detected: Moving forward')
                msg = Twist()
                msg.linear.x = 2.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.cmd_vel_publisher.publish(msg)
            return image


    def nms_boxes(self, boxes, scores, iou_threshold=0.45):
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def evaluate(self, yolo_outputs, image_shape, class_names, anchors):
        score_thresh = 0.7
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = np.shape(yolo_outputs[0])[1:3]
        input_shape = np.array(input_shape) * 32

        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(
                yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), 
                input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = np.concatenate(boxes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        mask = box_scores >= score_thresh
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(class_names)):
            class_boxes = boxes[mask[:, c]]
            class_box_scores = box_scores[:, c]
            class_box_scores = class_box_scores[mask[:, c]]
            nms_index = self.nms_boxes(class_boxes, class_box_scores) 
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            classes = np.ones_like(class_box_scores, dtype=np.int32) * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = np.concatenate(boxes_, axis=0)
        scores_ = np.concatenate(scores_, axis=0)
        classes_ = np.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_

    def process_frame(self, input_image):
        # Pre-process frame
        image_size = input_image.shape[:2]
        image_data = np.array(self.pre_process(input_image, (416, 416)), dtype=np.float32)
    
        # Fetch data to DPU and trigger it
        self.image[0, ...] = image_data.reshape(self.shapeIn[1:])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        # Retrieve output data
        conv_out0 = np.reshape(self.output_data[0], self.shapeOut0)
        conv_out1 = np.reshape(self.output_data[1], self.shapeOut1)
        conv_out2 = np.reshape(self.output_data[2], self.shapeOut2)
        yolo_outputs = [conv_out0, conv_out1, conv_out2]
    
        # Decode output from YOLOv3
        boxes, scores, classes = self.evaluate(yolo_outputs, image_size, self.class_names, self.anchors)
    
        # Combine boxes, scores, and classes for drawing
        bboxes = np.concatenate([boxes, scores[:, np.newaxis], classes[:, np.newaxis]], axis=1)
    
        # Draw bounding boxes on the original image
        result_image = self.draw_bbox(input_image, bboxes)
    
        return result_image
    def listener_callback(self, msg):
        self.get_logger().info('Receiving video frame')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        processed_image = self.process_frame(cv_image)
        cv2.imshow("YOLOv3 DPU", processed_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    yolo_dpu_node = YOLODPUNode()
    rclpy.spin(yolo_dpu_node)
    yolo_dpu_node.destroy_node()
    rclpy.shutdown()


