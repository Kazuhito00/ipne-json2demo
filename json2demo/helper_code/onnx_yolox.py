"""
YOLOX Object Detection ヘルパー関数テンプレート
"""

ONNX_HELPER_YOLOX = """
class YOLOXDetector:
    def __init__(self, model_path, nms_th=0.45, nms_score_th=0.1):
        self.nms_th = nms_th
        self.nms_score_th = nms_score_th
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
    def __call__(self, image, score_th=0.3):
        h, w = image.shape[:2]
        input_h, input_w = self.input_shape
        ratio = min(input_h / h, input_w / w)
        resized = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded = np.ones((input_h, input_w, 3), dtype=np.uint8) * 114
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None]
        outputs = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(outputs, ratio, w, h, score_th)
    def _postprocess(self, outputs, ratio, max_w, max_h, score_th):
        # Grid and stride calculation for YOLOX
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]
        hsizes = [self.input_shape[0] // s for s in strides]
        wsizes = [self.input_shape[1] // s for s in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        predictions = outputs[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = self._multiclass_nms(boxes_xyxy, scores, self.nms_th, self.nms_score_th)
        if dets is None:
            return np.array([]), np.array([]), np.array([])
        bboxes, scores_out, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, max_w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, max_h)
        return bboxes, scores_out, class_ids
    def _nms(self, boxes, scores, nms_thr):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
    def _multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        valid_mask = cls_scores > score_thr
        if valid_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_mask]
        valid_boxes = boxes[valid_mask]
        valid_cls_inds = cls_inds[valid_mask]
        keep = self._nms(valid_boxes, valid_scores, nms_thr)
        if not keep:
            return None
        return np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def get_color(index):
    temp_index = abs(int(index + 35)) * 3
    return ((29 * temp_index) % 255, (17 * temp_index) % 255, (37 * temp_index) % 255)

def draw_object_detection_info(image, score_th, bboxes, scores, class_ids, class_names, thickness=3):
    debug_image = image.copy()
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if score_th > score:
            continue
        color = get_color(class_id)
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness=thickness)
        score_text = '%.2f' % score
        text = '%s:%s(%s)' % (int(class_id), str(class_names[int(class_id)]), score_text)
        cv2.putText(debug_image, text, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=thickness)
    return debug_image
"""
