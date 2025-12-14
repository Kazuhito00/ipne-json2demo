"""
Deep Learningノード用テンプレート

MediaPipe、ONNX Runtime を使用するDLノードを定義
"""

DL_NODE_TEMPLATES = {
    # MediaPipe FaceDetection
    "FaceDetection_MediaPipe": {
        "init": '# Load face detection model\n_face{node_id}_model = create_face_detection_model("{model}")',
        "process": "_face{node_id}_results = run_face_detection(_face{node_id}_model, {input}, {score_threshold})",
        "draw": "{output} = draw_face_detection_mp({input}, _face{node_id}_results, {score_threshold})",
        "cleanup": "",
        "params": {"model": "MediaPipe FaceDetection(~2m)", "score_threshold": 0.5},
        "output_type": "image",
        "requires_dl": True,
    },
    "FaceDetection_FaceMesh": {
        "init": "# Load face mesh model\n_facemesh{node_id}_model = create_facemesh_model()",
        "process": "_facemesh{node_id}_results = run_facemesh_detection(_facemesh{node_id}_model, {input})",
        "draw": "{output} = draw_facemesh({input}, _facemesh{node_id}_results)",
        "cleanup": "",
        "params": {},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX YuNet FaceDetection
    "FaceDetection_YuNet": {
        "init": (
            '# Load face detection model\n'
            '_face{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_face{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _face{node_id}_model = YuNetFaceDetector(_face{node_id}_model_path)\n'
            'else:\n'
            '    _face{node_id}_model = None\n'
            '    if not os.path.exists(_face{node_id}_model_path):\n'
            '        print("[FaceDetection] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _face{node_id}_model is not None:\n"
            "    _face{node_id}_bboxes, _face{node_id}_landmarks, _face{node_id}_scores = _face{node_id}_model({input}, {score_threshold})\n"
            "else:\n"
            "    _face{node_id}_bboxes, _face{node_id}_landmarks, _face{node_id}_scores = [], [], []"
        ),
        "draw": (
            "if _face{node_id}_model is not None:\n"
            "    {output} = draw_face_detections({input}, _face{node_id}_bboxes, _face{node_id}_landmarks, _face{node_id}_scores, {score_threshold})\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "YuNet", "score_threshold": 0.6},
        "output_type": "image",
        "requires_dl": True,
    },
    # MediaPipe PoseEstimation
    "PoseEstimation": {
        "init": '# Load pose estimation model\n_pose{node_id}_model = create_pose_model("{model}")',
        "process": "_pose{node_id}_results = run_pose_estimation(_pose{node_id}_model, {input}, {score_threshold})",
        "draw": "{output} = draw_pose_estimation({input}, _pose{node_id}_results, {score_threshold})",
        "cleanup": "",
        "params": {"model": "MediaPipe Pose(Complexity0)", "score_threshold": 0.5},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX MoveNet PoseEstimation
    "PoseEstimation_MoveNet": {
        "init": (
            '# Load pose estimation model\n'
            '_movenet{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_movenet{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _movenet{node_id}_model = MoveNetDetector(_movenet{node_id}_model_path)\n'
            'else:\n'
            '    _movenet{node_id}_model = None\n'
            '    if not os.path.exists(_movenet{node_id}_model_path):\n'
            '        print("[PoseEstimation] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _movenet{node_id}_model is not None:\n"
            "    _movenet{node_id}_results = _movenet{node_id}_model({input}, {score_threshold})\n"
            "else:\n"
            "    _movenet{node_id}_results = None"
        ),
        "draw": (
            "if _movenet{node_id}_model is not None:\n"
            "    {output} = draw_movenet_pose({input}, _movenet{node_id}_results, {score_threshold})\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "MoveNet(SinglePose Lightning)", "score_threshold": 0.3},
        "output_type": "image",
        "requires_dl": True,
    },
    # MediaPipe SemanticSegmentation
    "SemanticSegmentation": {
        "init": '# Load selfie segmentation model\n_seg{node_id}_model = create_selfie_segmentation_model("{model}")',
        "process": "_seg{node_id}_results = run_selfie_segmentation(_seg{node_id}_model, {input})",
        "draw": "{output} = draw_selfie_segmentation({input}, _seg{node_id}_results, {score_threshold})",
        "cleanup": "",
        "params": {
            "model": "MediaPipe SelfieSegmentation(Normal)",
            "score_threshold": 0.5,
        },
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX DeepLab SemanticSegmentation
    "SemanticSegmentation_DeepLab": {
        "init": (
            '# Load semantic segmentation model\n'
            '_seg{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_seg{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _seg{node_id}_model = DeepLabV3Segmenter(_seg{node_id}_model_path)\n'
            'else:\n'
            '    _seg{node_id}_model = None\n'
            '    if not os.path.exists(_seg{node_id}_model_path):\n'
            '        print("[SemanticSegmentation] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _seg{node_id}_model is not None:\n"
            "    _seg{node_id}_results = _seg{node_id}_model({input})\n"
            "else:\n"
            "    _seg{node_id}_results = None"
        ),
        "draw": (
            "if _seg{node_id}_model is not None:\n"
            "    {output} = draw_deeplab_segmentation({input}, _seg{node_id}_results, {score_threshold})\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {
            "model": "DeepLabV3",
            "score_threshold": 0.5,
        },
        "output_type": "image",
        "requires_dl": True,
    },
    # MediaPipe HandDetection
    "HandDetection": {
        "init": '# Load hand detection model\n_hands{node_id}_model = create_hands_model("{model}")',
        "process": "_hands{node_id}_results = run_hands_detection(_hands{node_id}_model, {input}, {score_threshold})",
        "draw": "{output} = draw_hands_detection({input}, _hands{node_id}_results, {score_threshold})",
        "cleanup": "",
        "params": {"model": "MediaPipe Hands(Complexity0)", "score_threshold": 0.5},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX YOLOX ObjectDetection
    "ObjectDetection": {
        "init": (
            '# Load object detection model\n'
            '_od{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_od{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _od{node_id}_model = YOLOXDetector(_od{node_id}_model_path)\n'
            'else:\n'
            '    _od{node_id}_model = None\n'
            '    if not os.path.exists(_od{node_id}_model_path):\n'
            '        print("[ObjectDetection] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _od{node_id}_model is not None:\n"
            "    _od{node_id}_boxes, _od{node_id}_scores, _od{node_id}_class_ids = _od{node_id}_model({input}, {score_threshold})\n"
            "else:\n"
            "    _od{node_id}_boxes, _od{node_id}_scores, _od{node_id}_class_ids = [], [], []"
        ),
        "draw": (
            "if _od{node_id}_model is not None:\n"
            "    {output} = draw_object_detection_info({input}, {score_threshold}, _od{node_id}_boxes, _od{node_id}_scores, _od{node_id}_class_ids, COCO_CLASSES)\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "YOLOX-Nano(416x416)", "score_threshold": 0.3},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX DEIMv2 ObjectDetection
    "ObjectDetection_DEIMv2": {
        "init": (
            '# Load object detection model\n'
            '_od{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_od{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _od{node_id}_model = DEIMv2Detector(_od{node_id}_model_path)\n'
            'else:\n'
            '    _od{node_id}_model = None\n'
            '    if not os.path.exists(_od{node_id}_model_path):\n'
            '        print("[ObjectDetection] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _od{node_id}_model is not None:\n"
            "    _od{node_id}_boxes, _od{node_id}_scores, _od{node_id}_class_ids = _od{node_id}_model({input}, {score_threshold})\n"
            "else:\n"
            "    _od{node_id}_boxes, _od{node_id}_scores, _od{node_id}_class_ids = [], [], []"
        ),
        "draw": (
            "if _od{node_id}_model is not None:\n"
            "    {output} = draw_object_detection_info({input}, {score_threshold}, _od{node_id}_boxes, _od{node_id}_scores, _od{node_id}_class_ids, COCO_CLASSES)\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "DEIMv2-Atto(COCO)", "score_threshold": 0.3},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX DEIMv2 Wholebody34 ObjectDetection
    "ObjectDetection_Wholebody34": {
        "init": (
            '# Load object detection model\n'
            '_od{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_od{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _od{node_id}_model = DEIMv2Wholebody34Detector(_od{node_id}_model_path)\n'
            'else:\n'
            '    _od{node_id}_model = None\n'
            '    if not os.path.exists(_od{node_id}_model_path):\n'
            '        print("[ObjectDetection] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _od{node_id}_model is not None:\n"
            "    _od{node_id}_results = _od{node_id}_model({input}, {score_threshold})\n"
            "else:\n"
            "    _od{node_id}_results = []"
        ),
        "draw": (
            "if _od{node_id}_model is not None:\n"
            "    {output} = draw_wholebody34_detection_info({input}, _od{node_id}_results)\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "DEIMv2-Wholebody34-Atto(320x320)", "score_threshold": 0.3},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX Classification
    "Classification": {
        "init": (
            '# Load classification model\n'
            '_cls{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_cls{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _cls{node_id}_model = Classifier(_cls{node_id}_model_path)\n'
            'else:\n'
            '    _cls{node_id}_model = None\n'
            '    if not os.path.exists(_cls{node_id}_model_path):\n'
            '        print("[Classification] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _cls{node_id}_model is not None:\n"
            "    _cls{node_id}_scores, _cls{node_id}_class_ids = _cls{node_id}_model({input})\n"
            "else:\n"
            "    _cls{node_id}_scores, _cls{node_id}_class_ids = [], []"
        ),
        "draw": (
            "if _cls{node_id}_model is not None:\n"
            "    {output} = draw_classification({input}, _cls{node_id}_scores, _cls{node_id}_class_ids)\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "MobileNetV3 Small"},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX MonocularDepthEstimation
    "MonocularDepthEstimation": {
        "init": (
            '# Load depth estimation model\n'
            '_depth{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_depth{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _depth{node_id}_model = DepthEstimator(_depth{node_id}_model_path)\n'
            'else:\n'
            '    _depth{node_id}_model = None\n'
            '    if not os.path.exists(_depth{node_id}_model_path):\n'
            '        print("[MonocularDepthEstimation] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _depth{node_id}_model is not None:\n"
            "    {output} = _depth{node_id}_model({input})\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "FSRE-Depth(320x192)"},
        "output_type": "image",
        "requires_dl": True,
    },
    # ONNX LLIE (Low-Light Image Enhancement)
    "LLIE": {
        "init": (
            '# Load low-light image enhancement model\n'
            '_llie{node_id}_model_path = "{model_path}"\n'
            'if os.path.exists(_llie{node_id}_model_path) and ONNX_AVAILABLE:\n'
            '    _llie{node_id}_model = LLIEModel(_llie{node_id}_model_path)\n'
            'else:\n'
            '    _llie{node_id}_model = None\n'
            '    if not os.path.exists(_llie{node_id}_model_path):\n'
            '        print("[LLIE] Model: {model}, Path: {model_path} not found")'
        ),
        "process": (
            "if _llie{node_id}_model is not None:\n"
            "    {output} = _llie{node_id}_model({input})\n"
            "else:\n"
            "    {output} = {input}.copy()"
        ),
        "cleanup": "",
        "params": {"model": "TBEFN(320x180)"},
        "output_type": "image",
        "requires_dl": True,
    },
}
