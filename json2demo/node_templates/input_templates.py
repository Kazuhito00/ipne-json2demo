"""
入力ノード用テンプレート

WebCam, ScreenCapture, Image, Video, RTSPInput等の入力ノードを定義
"""

INPUT_NODE_TEMPLATES = {
    "WebCam": {
        "init": (
            "# Initialize webcam capture\n"
            "cap{node_id} = cv2.VideoCapture(args.camera)\n"
            "cap{node_id}.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)\n"
            "cap{node_id}.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)\n"
            "_cap_fps{node_id} = cap{node_id}.get(cv2.CAP_PROP_FPS)\n"
            "if _cap_fps{node_id} <= 0:\n"
            "    _cap_fps{node_id} = 30.0"
        ),
        "process": "ret, {output} = cap{node_id}.read()\nif not ret:\n    break",
        "cleanup": "cap{node_id}.release()",
        "params": {},
        "output_type": "image",
    },
    "ScreenCapture": {
        "init": (
            "# Initialize screen capture\n"
            "_cap_fps{node_id} = 30.0"
        ),
        "process": (
            "_pil_image{node_id} = ImageGrab.grab(all_screens=True)\n"
            "{output} = cv2.cvtColor(np.array(_pil_image{node_id}, dtype=np.uint8), cv2.COLOR_RGB2BGR)"
        ),
        "cleanup": "",
        "params": {},
        "output_type": "image",
    },
    "Image": {
        "init": (
            '# Load static image\n'
            '{output}_path = args.image\n'
            '{output} = cv2.imread({output}_path)\n'
            'if {output} is None:\n'
            '    raise FileNotFoundError(f"Image not found: {{{output}_path}}")'
        ),
        "process": "",
        "cleanup": "",
        "params": {},
        "output_type": "image",
        "uses_args": "image",
    },
    "Video": {
        "init": (
            '# Initialize video capture\n'
            'cap{node_id} = cv2.VideoCapture(args.video)\n'
            'if not cap{node_id}.isOpened():\n'
            '    raise FileNotFoundError(f"Video not found or cannot open: {{args.video}}")\n'
            'skip_frame{node_id} = {skip_frame}\n'
            '_cap_fps{node_id} = cap{node_id}.get(cv2.CAP_PROP_FPS)\n'
            'if _cap_fps{node_id} <= 0:\n'
            '    _cap_fps{node_id} = 30.0'
        ),
        "process": (
            "for _ in range(skip_frame{node_id}):\n"
            "    cap{node_id}.read()\n"
            "ret, {output} = cap{node_id}.read()\n"
            "if not ret:\n"
            "    cap{node_id}.set(cv2.CAP_PROP_POS_FRAMES, 0)\n"
            "    ret, {output} = cap{node_id}.read()"
        ),
        "cleanup": "cap{node_id}.release()",
        "params": {"skip_frame": 1},
        "output_type": "image",
        "uses_args": "video",
    },
    "RTSPInput": {
        "init": (
            '# Initialize RTSP stream capture\n'
            'cap{node_id} = cv2.VideoCapture("{rtsp_url}")\n'
            '_cap_fps{node_id} = cap{node_id}.get(cv2.CAP_PROP_FPS)\n'
            'if _cap_fps{node_id} <= 0:\n'
            '    _cap_fps{node_id} = 30.0'
        ),
        "process": "ret, {output} = cap{node_id}.read()\nif not ret:\n    continue",
        "cleanup": "cap{node_id}.release()",
        "params": {"rtsp_url": "rtsp://localhost:8554/stream"},
        "output_type": "image",
    },
    "VideoSetFramePos": {
        "init": (
            '# Open video and read single frame at specified position\n'
            'cap{node_id} = cv2.VideoCapture(args.video)\n'
            'if not cap{node_id}.isOpened():\n'
            '    raise FileNotFoundError(f"Video not found or cannot open: {{args.video}}")\n'
            '_total_frames{node_id} = int(cap{node_id}.get(cv2.CAP_PROP_FRAME_COUNT))\n'
            '_cap_fps{node_id} = cap{node_id}.get(cv2.CAP_PROP_FPS)\n'
            'if _cap_fps{node_id} <= 0:\n'
            '    _cap_fps{node_id} = 30.0\n'
            '\n'
            '# Clamp frame position to valid range\n'
            '_frame_pos{node_id} = max(0, min(args.frame_pos, _total_frames{node_id} - 1))\n'
            'cap{node_id}.set(cv2.CAP_PROP_POS_FRAMES, _frame_pos{node_id})\n'
            'ret, _static_frame{node_id} = cap{node_id}.read()\n'
            'if not ret:\n'
            '    raise RuntimeError(f"Failed to read frame at position {{_frame_pos{node_id}}}")\n'
            'cap{node_id}.release()'
        ),
        "process": "{output} = _static_frame{node_id}.copy()",
        "cleanup": "",
        "params": {"frame_pos": 0},
        "output_type": "image",
        "uses_args": "video_framepos",
    },
    "IntValue": {
        "init": "",
        "process": "",
        "cleanup": "",
        "params": {"int_value": 0},
        "output_type": "int",
        "skip_codegen": True,
    },
    "FloatValue": {
        "init": "",
        "process": "",
        "cleanup": "",
        "params": {"float_value": 0.0},
        "output_type": "float",
        "skip_codegen": True,
    },
}
