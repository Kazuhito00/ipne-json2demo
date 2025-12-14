"""
ONNX Runtime ヘルパー関数テンプレート (その他モデル用)
"""

from json2demo.helper_code.onnx_yolox import ONNX_HELPER_YOLOX
from json2demo.helper_code.onnx_deimv2 import ONNX_HELPER_DEIMV2

# Wholebody34検出器 (長いため別途定義)
ONNX_HELPER_WHOLEBODY34 = '''
class DEIMv2Wholebody34Detector:
    """DEIMv2 Wholebody34 Detector with extended attributes"""
    def __init__(self, model_path, obj_score_th=0.35, attr_score_th=0.70, keypoint_th=0.35):
        import re
        self._obj_score_th = obj_score_th
        self._attr_score_th = attr_score_th
        self._keypoint_th = keypoint_th
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        try:
            self.input_h = int(input_shape[2])
            self.input_w = int(input_shape[3])
        except (ValueError, TypeError):
            match = re.search(r'(\\d+)x(\\d+)', model_path)
            if match:
                self.input_h = int(match.group(1))
                self.input_w = int(match.group(2))
            else:
                self.input_h, self.input_w = 640, 640

    def __call__(self, image, score_th=0.3):
        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.input_w, self.input_h))
        blob = resized.transpose(2, 0, 1).astype(np.float32)[None]
        outputs = self.session.run(None, {self.input_name: blob})
        boxes_raw = outputs[0][0] if len(outputs) > 0 else np.array([])
        if len(boxes_raw) == 0:
            return []
        min_th = min(self._obj_score_th, self._attr_score_th, self._keypoint_th)
        boxes_raw = boxes_raw[boxes_raw[:, 5] > min_th]
        results = []
        for box in boxes_raw:
            classid = int(box[0])
            x1 = int(max(0, box[1]) * w)
            y1 = int(max(0, box[2]) * h)
            x2 = int(min(box[3], 1.0) * w)
            y2 = int(min(box[4], 1.0) * h)
            score = float(box[5])
            results.append({
                'classid': classid, 'score': score,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) // 2, 'cy': (y1 + y2) // 2,
                'generation': -1, 'gender': -1, 'handedness': -1, 'head_pose': -1,
                'is_used': False,
            })
        obj_classes = {0, 5, 6, 7, 16, 17, 18, 19, 20, 26, 27, 28, 33}
        attr_classes = {1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15}
        keypoint_classes = {21, 22, 23, 24, 25, 29, 30, 31, 32}
        results = [r for r in results if not (r['classid'] in obj_classes and r['score'] < self._obj_score_th)]
        results = [r for r in results if not (r['classid'] in attr_classes and r['score'] < self._attr_score_th)]
        results = [r for r in results if not (r['classid'] in keypoint_classes and r['score'] < self._keypoint_th)]
        self._merge_attributes(results)
        results = [r for r in results if r['classid'] not in {1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 27, 28}]
        return results

    def _calc_iou(self, a, b):
        ix1, iy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
        ix2, iy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
        area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
        return inter / float(area_a + area_b - inter)

    def _merge_attributes(self, results):
        body_boxes = [r for r in results if r['classid'] == 0]
        gen_boxes = [r for r in results if r['classid'] in {1, 2}]
        for base in body_boxes:
            best = self._find_best_match(base, gen_boxes)
            if best:
                base['generation'] = 0 if best['classid'] == 1 else 1
                best['is_used'] = True
        gender_boxes = [r for r in results if r['classid'] in {3, 4}]
        for base in body_boxes:
            best = self._find_best_match(base, gender_boxes)
            if best:
                base['gender'] = 0 if best['classid'] == 3 else 1
                best['is_used'] = True
        head_boxes = [r for r in results if r['classid'] == 7]
        pose_boxes = [r for r in results if r['classid'] in {8, 9, 10, 11, 12, 13, 14, 15}]
        for base in head_boxes:
            best = self._find_best_match(base, pose_boxes)
            if best:
                base['head_pose'] = best['classid'] - 8
                best['is_used'] = True
        hand_boxes = [r for r in results if r['classid'] == 26]
        hand_lr_boxes = [r for r in results if r['classid'] in {27, 28}]
        for base in hand_boxes:
            best = self._find_best_match(base, hand_lr_boxes)
            if best:
                base['handedness'] = 0 if best['classid'] == 27 else 1
                best['is_used'] = True

    def _find_best_match(self, base, targets):
        best, best_score, best_iou = None, 0.0, 0.0
        for t in targets:
            if t['is_used']:
                continue
            dist = ((base['cx'] - t['cx'])**2 + (base['cy'] - t['cy'])**2)**0.5
            if dist > 10.0:
                continue
            if t['score'] >= best_score:
                iou = self._calc_iou(base, t)
                if iou > best_iou:
                    best, best_score, best_iou = t, t['score'], iou
        return best

WHOLEBODY34_CLASSES = {
    0: 'body', 5: 'body_with_wheelchair', 6: 'body_with_crutches',
    7: 'head', 16: 'face', 17: 'eye', 18: 'nose', 19: 'mouth', 20: 'ear',
    21: 'left_hand_kpt', 22: 'right_hand_kpt',
    23: 'left_foot_kpt', 24: 'right_foot_kpt', 25: 'face_kpt',
    26: 'hand', 29: 'left_foot_kpt2', 30: 'right_foot_kpt2',
    31: 'left_hand_kpt2', 32: 'right_hand_kpt2', 33: 'foot'
}

def draw_wholebody34_detection_info(image, results):
    """Draw Wholebody34 detection results on image"""
    output = image.copy()
    if not results:
        return output
    keypoint_classes = {21, 22, 23, 24, 25, 29, 30, 31, 32}
    for r in results:
        x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
        cx, cy = r['cx'], r['cy']
        classid = r['classid']
        score = r['score']
        label = WHOLEBODY34_CLASSES.get(classid, f'class_{classid}')
        # Color by class type
        if classid == 0:  # body
            color = (0, 255, 0)
        elif classid == 7:  # head
            color = (255, 0, 0)
        elif classid == 26:  # hand
            color = (0, 255, 255)
        elif classid in keypoint_classes:
            color = (0, 128, 255)
        else:
            color = (255, 255, 0)
        # Draw keypoints as dots, others as rectangles
        if classid in keypoint_classes:
            cv2.circle(output, (cx, cy), 5, color, -1)
        else:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            # Add attributes if available
            attrs = []
            if r.get('generation', -1) >= 0:
                attrs.append('Adult' if r['generation'] == 0 else 'Child')
            if r.get('gender', -1) >= 0:
                attrs.append('M' if r['gender'] == 0 else 'F')
            if r.get('head_pose', -1) >= 0:
                poses = ['Front', 'Right-Front', 'Right', 'Right-Back', 'Back', 'Left-Back', 'Left', 'Left-Front']
                attrs.append(poses[r['head_pose']])
            if r.get('handedness', -1) >= 0:
                attrs.append('L' if r['handedness'] == 0 else 'R')
            text = f'{label}:{score:.2f}'
            if attrs:
                text += f' ({",".join(attrs)})'
            cv2.putText(output, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(output, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return output
'''

# その他のONNXヘルパー
ONNX_HELPERS = {
    "SemanticSegmentation_DeepLab": """
def _get_color_map_list(num_classes):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3 + 2] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map

class DeepLabV3Segmenter:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_detail = self.session.get_inputs()[0]
        self.input_name = self.input_detail.name
        self.output_detail = self.session.get_outputs()[0]
        self.input_shape = self.input_detail.shape[1:3]
    def __call__(self, image):
        h, w = image.shape[:2]
        input_h, input_w = self.input_shape
        resized = cv2.resize(image, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(rgb, axis=0).astype(np.float32)
        blob = blob / 127.5 - 1
        result = self.session.run(None, {self.input_name: blob})
        segmentation_map = np.squeeze(result[0])
        segmentation_map = cv2.resize(segmentation_map, (w, h), interpolation=cv2.INTER_LINEAR)
        segmentation_map = segmentation_map.transpose(2, 0, 1)
        segmentation_map = np.argmax(segmentation_map, axis=0)
        class_num = self.output_detail.shape[3]
        segmentation_map_list = []
        for index in range(0, class_num):
            mask = np.where(segmentation_map == index, 1.0, 0.0)
            segmentation_map_list.append(mask)
        return {'segmentation_map': np.array(segmentation_map_list), 'class_num': class_num}
    def get_class_num(self):
        return self.output_detail.shape[3]

def draw_deeplab_segmentation(image, segmentation, score_th=0.5):
    if segmentation is None:
        return image.copy()
    output = image.copy()
    seg_map = segmentation['segmentation_map']
    class_num = segmentation['class_num']
    seg_map = np.where(seg_map > score_th, 0, 1)
    color_map = _get_color_map_list(class_num)
    for index, mask in enumerate(seg_map):
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (color_map[index * 3 + 0], color_map[index * 3 + 1], color_map[index * 3 + 2])
        mask = np.stack((mask,) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, output, bg_image)
        output = cv2.addWeighted(output, 0.5, mask_image, 0.5, 1.0)
    return output
""",
    "PoseEstimation_MoveNet": """
MOVENET_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

class MoveNetDetector:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = input_shape[1] if input_shape[1] else 192
    def __call__(self, image, score_th=0.3):
        h, w = image.shape[:2]
        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(img, axis=0).astype(np.int32)
        outputs = self.session.run(None, {self.input_name: blob})[0]
        keypoints = outputs[0, 0, :, :]
        points = []
        for kp in keypoints:
            y, x, conf = kp
            points.append((int(x * w), int(y * h), float(conf)))
        return {'landmarks': points}

def draw_movenet_pose(image, pose, score_th=0.3):
    output = image.copy()
    if pose is None:
        return output
    points = pose['landmarks']
    for x, y, conf in points:
        if conf > score_th:
            cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
    for i, j in MOVENET_CONNECTIONS:
        if i < len(points) and j < len(points):
            if points[i][2] > score_th and points[j][2] > score_th:
                cv2.line(output, points[i][:2], points[j][:2], (0, 255, 0), 2)
    return output
""",
    "LLIE": """
class LLIEModel:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
    def __call__(self, image):
        h, w = image.shape[:2]
        input_h, input_w = self.input_shape
        resized = cv2.resize(image, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        result = self.session.run(None, {self.input_name: blob})[0]
        output = np.squeeze(result)
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        if output.shape[0] == 3:
            output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return cv2.resize(output, (w, h))
""",
    "MonocularDepthEstimation": """
class DepthEstimator:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
    def __call__(self, image):
        h, w = image.shape[:2]
        input_h, input_w = self.input_shape
        resized = cv2.resize(image, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        result = self.session.run(None, {self.input_name: blob})[0]
        depth = np.squeeze(result)
        depth = (depth * 255).astype(np.uint8)
        return cv2.resize(depth, (w, h)) if len(depth.shape) == 2 else depth
""",
    "Classification": """
IMAGENET_CLASSES = {imagenet_class_names}

class Classifier:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
    def __call__(self, image, top_k=5):
        resized = cv2.resize(image, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(rgb, axis=0).astype(np.float32)
        result = self.session.run(None, {self.input_name: blob})[0]
        result = np.squeeze(result)
        top_indices = np.argsort(result)[::-1][:top_k]
        return result[top_indices], top_indices

def draw_classification(image, scores, class_ids):
    output = image.copy()
    h, w = output.shape[:2]
    for i, (score, cls_id) in enumerate(zip(scores[:3], class_ids[:3])):
        name = IMAGENET_CLASSES.get(cls_id, str(cls_id))
        label = f"[{cls_id}] {name}: {score:.3f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = h - 20 - (2 - i) * 25
        cv2.putText(output, label, (w - tw - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return output
""",
    "FaceDetection_YuNet": """
class YuNetFaceDetector:
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]
    def __init__(self, model_path, input_shape=(160, 120), nms_th=0.3):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = input_shape
        self.nms_th = nms_th
        self.priors = self._generate_priors()
    def _generate_priors(self):
        w, h = self.input_shape
        feature_map_2th = [int(int((h + 1) / 2) / 2), int(int((w + 1) / 2) / 2)]
        feature_map_3th = [feature_map_2th[0] // 2, feature_map_2th[1] // 2]
        feature_map_4th = [feature_map_3th[0] // 2, feature_map_3th[1] // 2]
        feature_map_5th = [feature_map_4th[0] // 2, feature_map_4th[1] // 2]
        feature_map_6th = [feature_map_5th[0] // 2, feature_map_5th[1] // 2]
        feature_maps = [feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th]
        priors = []
        for k, f in enumerate(feature_maps):
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in self.MIN_SIZES[k]:
                        s_kx, s_ky = min_size / w, min_size / h
                        cx = (j + 0.5) * self.STEPS[k] / w
                        cy = (i + 0.5) * self.STEPS[k] / h
                        priors.append([cx, cy, s_kx, s_ky])
        return np.array(priors, dtype=np.float32)
    def __call__(self, image, score_th=0.6):
        h, w = image.shape[:2]
        resized = cv2.resize(image, self.input_shape)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = rgb.transpose(2, 0, 1).reshape(1, 3, self.input_shape[1], self.input_shape[0])
        loc, conf, iou = self.session.run(None, {self.input_name: blob})
        bboxes, landmarks, scores = self._decode(loc, conf, iou, score_th)
        keep = self._nms(bboxes, scores, self.nms_th)
        bboxes = [bboxes[i] for i in keep]
        landmarks = [landmarks[i] for i in keep]
        scores = [scores[i] for i in keep]
        scale_x, scale_y = w / self.input_shape[0], h / self.input_shape[1]
        for i in range(len(bboxes)):
            bboxes[i] = [int(bboxes[i][0] * scale_x), int(bboxes[i][1] * scale_y),
                         int(bboxes[i][2] * scale_x), int(bboxes[i][3] * scale_y)]
            for j in range(len(landmarks[i])):
                landmarks[i][j] = [int(landmarks[i][j][0] * scale_x), int(landmarks[i][j][1] * scale_y)]
        return bboxes, landmarks, scores
    def _nms(self, bboxes, scores, nms_th):
        if len(bboxes) == 0:
            return []
        bboxes_arr = np.array(bboxes)
        scores_arr = np.array(scores)
        x1, y1, x2, y2 = bboxes_arr[:, 0], bboxes_arr[:, 1], bboxes_arr[:, 2], bboxes_arr[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores_arr.argsort()[::-1]
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
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_th)[0]
            order = order[inds + 1]
        return keep
    def _decode(self, loc, conf, iou, score_th):
        cls_scores = conf[:, 1]
        iou_scores = np.clip(iou[:, 0], 0, 1)
        scores = np.sqrt(cls_scores * iou_scores)
        scale = np.array(self.input_shape)
        bboxes = np.hstack(((self.priors[:, 0:2] + loc[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
                            (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE[1])) * scale))
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2
        bboxes[:, 2:4] += bboxes[:, 0:2]
        landmarks = []
        for k in range(5):
            lm = (self.priors[:, 0:2] + loc[:, 4+k*2:6+k*2] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale
            landmarks.append(lm)
        landmarks = np.stack(landmarks, axis=1)
        mask = scores > score_th
        return bboxes[mask].astype(np.int32).tolist(), landmarks[mask].astype(np.int32).tolist(), scores[mask].tolist()

def draw_face_detections(image, bboxes, landmarks, scores, score_th=0.6):
    output = image.copy()
    for bbox, lms, score in zip(bboxes, landmarks, scores):
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        for lm in lms:
            cv2.circle(output, (lm[0], lm[1]), 3, (0, 255, 0), -1)
    return output
""",
}
