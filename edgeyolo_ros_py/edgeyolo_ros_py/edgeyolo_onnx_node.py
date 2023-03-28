import onnxruntime as ort
import numpy as np
import os
import cv2
import time
from .edgeyolo_onnx import EdgeYOLO_ONNX
from .utils import multiclass_nms, draw_rectangle

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from bboxes_ex_msgs.msg import BoundingBoxes
from bboxes_ex_msgs.msg import BoundingBox


class EdgeYOLO_ONNX_Node(Node):
    def __init__(self, node_name="edgeyolo_onnx_node"):
        super().__init__(node_name)

        model_path = self.declare_parameter(
            "model_path",
            os.path.join(os.path.dirname(__file__), "../../weights/edgeyolo_tiny_coco_416x416.onnx")
        ).value
        self.nms_th = self.declare_parameter("nms_th", 0.3).value
        self.score_th = self.declare_parameter("score_th", 0.4).value
        self.imshow = self.declare_parameter("imshow", True).value

        self.cv_bridge = CvBridge()
        self.model = EdgeYOLO_ONNX(model_path)

        self.create_subscription(Image, "image_raw", self.msg_callback, 10)
        self.image_publisher = self.create_publisher(Image, "edgeyolo_ros_py/image_raw", 10)
        self.boxes_publisher = self.create_publisher(BoundingBoxes, "edgeyolo_ros_py/bounding_boxes", 10)

    def msg_callback(self, msg: Image):
        img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        t = time.time()
        boxes, scores, cls = self.model.inference(img, nms_th=self.nms_th, score_th=self.score_th)
        dt = time.time() - t
        self.get_logger().info(f"Inference: {1/dt:.3f} FPS")

        h, w = img.shape[:2]
        boxes_msg = BoundingBoxes()
        boxes_msg.header = msg.header
        for box, score, c in zip(boxes, scores, cls):
            box_msg = BoundingBox()
            box_msg.class_id = str(c)
            box_msg.probability = score
            box_msg.img_height = h
            box_msg.img_width = w
            box_msg.xmin = int(min(max(box[0], 0), w - 1))
            box_msg.ymin = int(min(max(box[1], 0), h - 1))
            box_msg.xmax = int(min(max(box[2], 0), w - 1))
            box_msg.ymax = int(min(max(box[3], 0), h - 1))
            boxes_msg.bounding_boxes.append(box_msg)
        self.boxes_publisher.publish(boxes_msg)

        draw_rectangle(img, boxes, scores, cls)
        pub_img_msg = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
        self.image_publisher.publish(pub_img_msg)

        if self.imshow:
            cv2.imshow(self.get_name(), img)
            key = cv2.waitKey(10)


def main():
    rclpy.init()

    node = EdgeYOLO_ONNX_Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

