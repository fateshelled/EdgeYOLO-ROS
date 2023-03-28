import onnxruntime as ort
import numpy as np
import cv2
import time
from .utils import multiclass_nms, draw_rectangle


class EdgeYOLO_ONNX:
    def __init__(self, model_path: str) -> None:
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.sess = ort.InferenceSession(
            model_path,
            providers=providers)

        input = self.sess.get_inputs()[0]
        self.input_name = input.name
        self.input_height = input.shape[2]
        self.input_width = input.shape[3]

        self.output_name = self.sess.get_outputs()[0].name

    def static_resize(self, image: np.ndarray):
        ret_img = np.full([self.input_height, self.input_width, 3],
                          fill_value=114, dtype=np.uint8)
        r = min(self.input_height / image.shape[0], self.input_width / image.shape[1])
        resized = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        ret_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized

        ret_img = ret_img.transpose((2, 0, 1))
        ret_img = np.ascontiguousarray(ret_img, dtype=np.float32)
        return ret_img, r

    def inference(self, image: np.ndarray, nms_th=0.3, score_th=0.4):
        blob, r = self.static_resize(image)
        output = self.sess.run([self.output_name],
                               {self.input_name: blob[None, :, :, :]})[0]

        boxes = output[0, :, :4]
        scores = output[0, :, 4:5] * output[0, :, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= r

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_th, score_thr=score_th)
        if dets is None:
            return [], [], []
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        return final_boxes, final_scores, final_cls_inds


if __name__ == "__main__":

    model = EdgeYOLO_ONNX("edgeyolo_tiny_coco_416x416.onnx")
    nms_th = 0.3
    score_th = 0.4

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        res, image = cap.read()
        if res is False:
            break

        t = time.time()
        boxes, scores, cls = model.inference(image, nms_th=nms_th, score_th=score_th)
        dt = time.time() - t
        print(f"Inference {1 / dt} FPS")

        drawn = image.copy()
        draw_rectangle(drawn, boxes, scores, cls)
        cv2.imshow("EdgeYOLO onnxruntime", drawn)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


