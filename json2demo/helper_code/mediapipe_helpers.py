"""
MediaPipe Deep Learning ヘルパー関数テンプレート
"""

DL_HELPERS = {
    "FaceDetection_MediaPipe": '''
def create_face_detection_model(model_name):
    """Create face detection model"""
    if not MEDIAPIPE_AVAILABLE:
        print(f"FaceDetection: mediapipe not installed, model '{model_name}' cannot be loaded")
        return None
    try:
        mp_face_detection = mp.solutions.face_detection
        if "~5m" in model_name:
            return mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
        else:
            return mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
    except Exception as e:
        print(f"FaceDetection: Failed to load model '{model_name}': {e}")
        return None

def run_face_detection(model, image, score_th=0.5):
    """Run face detection (processing only, no drawing)"""
    if model is None:
        return None
    h, w = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_image)
    if not results.detections:
        return None
    faces = []
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)
        keypoints = []
        for keypoint in detection.location_data.relative_keypoints:
            kx = int(keypoint.x * w)
            ky = int(keypoint.y * h)
            keypoints.append((kx, ky))
        faces.append({'bbox': (x1, y1, x2, y2), 'keypoints': keypoints, 'score': detection.score[0]})
    return faces

def draw_face_detection_mp(image, faces, score_th=0.5):
    """Draw face detection results on image"""
    output = image.copy()
    if faces is None:
        return output
    for face in faces:
        if face['score'] < score_th:
            continue
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for kx, ky in face['keypoints']:
            cv2.circle(output, (kx, ky), 5, (0, 255, 0), -1)
    return output
''',
    "FaceDetection_FaceMesh": '''
def create_facemesh_model():
    """Create FaceMesh model"""
    if not MEDIAPIPE_AVAILABLE:
        print("FaceMesh: mediapipe not installed")
        return None
    return mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def run_facemesh_detection(model, image):
    """Run FaceMesh detection"""
    if model is None:
        return None
    h, w = image.shape[:2]
    results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    return [[(int(lm.x * w), int(lm.y * h)) for lm in face.landmark] for face in results.multi_face_landmarks]

def draw_facemesh(image, faces):
    """Draw FaceMesh results on image"""
    output = image.copy()
    if faces:
        for landmarks in faces:
            for x, y in landmarks:
                cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
    return output
''',
    "HandDetection": '''
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def create_hands_model(model_name):
    """Create hand detection model"""
    if not MEDIAPIPE_AVAILABLE:
        print(f"HandDetection: mediapipe not installed, model '{model_name}' cannot be loaded")
        return None
    try:
        mp_hands = mp.solutions.hands
        complexity = 1 if "Complexity1" in model_name else 0
        return mp_hands.Hands(model_complexity=complexity, max_num_hands=2,
                              min_detection_confidence=0.7, min_tracking_confidence=0.5)
    except Exception as e:
        print(f"HandDetection: Failed to load model '{model_name}': {e}")
        return None

def run_hands_detection(model, image, score_th=0.5):
    """Run hand detection (processing only, no drawing)"""
    if model is None:
        return None
    h, w = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_image)
    if not results.multi_hand_landmarks:
        return None
    hands = []
    for hand_landmarks in results.multi_hand_landmarks:
        points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        hands.append({'landmarks': points})
    return hands

def draw_hands_detection(image, hands, score_th=0.5):
    """Draw hand detection results on image"""
    output = image.copy()
    if hands is None:
        return output
    for hand in hands:
        points = hand['landmarks']
        for x, y in points:
            cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
        for i, j in HAND_CONNECTIONS:
            if i < len(points) and j < len(points):
                cv2.line(output, points[i], points[j], (0, 255, 0), 2)
    return output
''',
    "PoseEstimation": '''
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

def create_pose_model(model_name):
    """Create pose estimation model"""
    if not MEDIAPIPE_AVAILABLE:
        print(f"PoseEstimation: mediapipe not installed, model '{model_name}' cannot be loaded")
        return None
    try:
        mp_pose = mp.solutions.pose
        if "Complexity2" in model_name:
            complexity = 2
        elif "Complexity1" in model_name:
            complexity = 1
        else:
            complexity = 0
        return mp_pose.Pose(model_complexity=complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:
        print(f"PoseEstimation: Failed to load model '{model_name}': {e}")
        return None

def run_pose_estimation(model, image, score_th=0.5):
    """Run pose estimation (processing only, no drawing)"""
    if model is None:
        return None
    h, w = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_image)
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        vis = lm.visibility
        points.append((x, y, vis))
    return {'landmarks': points}

def draw_pose_estimation(image, pose, score_th=0.5):
    """Draw pose estimation results on image"""
    output = image.copy()
    if pose is None:
        return output
    points = pose['landmarks']
    for x, y, vis in points:
        if vis > score_th:
            cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
    for i, j in POSE_CONNECTIONS:
        if i < len(points) and j < len(points):
            if points[i][2] > score_th and points[j][2] > score_th:
                cv2.line(output, points[i][:2], points[j][:2], (0, 255, 0), 2)
    return output
''',
    "SemanticSegmentation": '''
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

def create_selfie_segmentation_model(model_name):
    """Create selfie segmentation model"""
    if not MEDIAPIPE_AVAILABLE:
        print(f"SemanticSegmentation: mediapipe not installed, model '{model_name}' cannot be loaded")
        return None
    try:
        mp_selfie = mp.solutions.selfie_segmentation
        model_selection = 1 if "LandScape" in model_name else 0
        return mp_selfie.SelfieSegmentation(model_selection=model_selection)
    except Exception as e:
        print(f"SemanticSegmentation: Failed to load model '{model_name}': {e}")
        return None

def run_selfie_segmentation(model, image):
    """Run selfie segmentation (processing only, no drawing)"""
    if model is None:
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_image)
    if results.segmentation_mask is None:
        return None
    segmentation_map = np.expand_dims(results.segmentation_mask, 0)
    return {'segmentation_map': segmentation_map, 'class_num': 1}

def draw_selfie_segmentation(image, segmentation, score_th=0.5):
    """Draw selfie segmentation results on image"""
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
''',
}
