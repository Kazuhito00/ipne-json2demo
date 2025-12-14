"""
DEIMv2 Object Detection ヘルパー関数テンプレート
"""

ONNX_HELPER_DEIMV2 = '''
class DEIMv2Detector:
    """DEIMv2 Object Detector for COCO classes"""
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic shapes (None or string values)
        self.input_h = int(input_shape[2]) if len(input_shape) > 2 and isinstance(input_shape[2], (int, float)) else 640
        self.input_w = int(input_shape[3]) if len(input_shape) > 3 and isinstance(input_shape[3], (int, float)) else 640
    def __call__(self, image, score_th=0.3):
        h, w = image.shape[:2]
        # Preprocess
        resized = cv2.resize(image, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        # Inference
        outputs = self.session.run(None, {self.input_name: blob})
        # Postprocess - DEIMv2 outputs: [labels, boxes, scores]
        labels = outputs[0][0] if len(outputs) > 0 else np.array([])
        boxes = outputs[1][0] if len(outputs) > 1 else np.array([])
        scores = outputs[2][0] if len(outputs) > 2 else np.array([])
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        # Scale boxes to original image size
        scale_x = w / self.input_w
        scale_y = h / self.input_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        # Filter by score
        mask = scores >= score_th
        return boxes[mask], scores[mask], labels[mask]

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
'''
