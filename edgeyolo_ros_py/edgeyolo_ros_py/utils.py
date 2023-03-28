import numpy as np
import cv2


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

COLORS = [
    [  0, 114, 189],
    [217,  83,  25],
    [237, 177,  32],
    [126,  47, 142],
    [119, 172,  48],
    [ 77, 190, 238],
    [162,  20,  47],
    [ 77,  77,  77],
    [153, 153, 153],
    [255,   0,   0],
    [255, 128,   0],
    [191, 191,   0],
    [  0, 255,   0],
    [  0,   0, 255],
    [170,   0, 255],
    [ 85,  85,   0],
    [ 85, 170,   0],
    [ 85, 255,   0],
    [170,  85,   0],
    [170, 170,   0],
    [170, 255,   0],
    [255,  85,   0],
    [255, 170,   0],
    [255, 255,   0],
    [  0,  85, 128],
    [  0, 170, 128],
    [  0, 255, 128],
    [ 85,   0, 128],
    [ 85,  85, 128],
    [ 85, 170, 128],
    [ 85, 255, 128],
    [170,   0, 128],
    [170,  85, 128],
    [170, 170, 128],
    [170, 255, 128],
    [255,   0, 128],
    [255,  85, 128],
    [255, 170, 128],
    [255, 255, 128],
    [  0,  85, 255],
    [  0, 170, 255],
    [  0, 255, 255],
    [ 85,   0, 255],
    [ 85,  85, 255],
    [ 85, 170, 255],
    [ 85, 255, 255],
    [170,   0, 255],
    [170,  85, 255],
    [170, 170, 255],
    [170, 255, 255],
    [255,   0, 255],
    [255,  85, 255],
    [255, 170, 255],
    [ 85,   0,   0],
    [128,   0,   0],
    [170,   0,   0],
    [212,   0,   0],
    [255,   0,   0],
    [  0,  43,   0],
    [  0,  85,   0],
    [  0, 128,   0],
    [  0, 170,   0],
    [  0, 212,   0],
    [  0, 255,   0],
    [  0,   0,  43],
    [  0,   0,  85],
    [  0,   0, 128],
    [  0,   0, 170],
    [  0,   0, 212],
    [  0,   0, 255],
    [  0,   0,   0],
    [ 36,  36,  36],
    [ 73,  73,  73],
    [109, 109, 109],
    [146, 146, 146],
    [182, 182, 182],
    [219, 219, 219],
    [  0, 114, 189],
    [ 80, 183, 189],
    [128, 128,   0]
]

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def draw_rectangle(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, names=COCO_NAMES):
    h, w = img.shape[:2]
    for box, score, c in zip(boxes, scores, cls):
        x0 = int(min(max(box[0], 0), w - 1))
        y0 = int(min(max(box[1], 0), h - 1))
        x1 = int(min(max(box[2], 0), w - 1))
        y1 = int(min(max(box[3], 0), h - 1))

        text = f"{names[int(c)]}: {score * 100:.2f}"
        font_face = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.0
        (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, 1)
        cv2.rectangle(img, (x0, y0), (x1, y1), color=COLORS[int(c)], thickness=1)
        cv2.rectangle(img, (x0, y0), (x0 + text_w, y0 + text_h), color=COLORS[int(c)], thickness=-1)

        if 0.3 * COLORS[int(c)][0] + 0.59 * COLORS[int(c)][0] + 0.11 * COLORS[int(c)][0] > 128:
            cv2.putText(img, text, (x0, y0 + text_h), font_face, font_scale, (0, 0, 0))
        else:
            cv2.putText(img, text, (x0, y0 + text_h), font_face, font_scale, (255, 255, 255))