"""
画像処理ノード用テンプレート

Blur, Grayscale, Canny, Threshold等の画像処理ノードを定義
"""

PROCESSING_NODE_TEMPLATES = {
    "Blur": {
        "init": "",
        "process": "{output} = cv2.blur({input}, ({kernel_size}, {kernel_size}))",
        "cleanup": "",
        "params": {"kernel_size": 5},
        "output_type": "image",
    },
    "Grayscale": {
        "init": "",
        "process": (
            "_gray{node_id} = cv2.cvtColor({input}, cv2.COLOR_BGR2GRAY)\n"
            "{output} = cv2.cvtColor(_gray{node_id}, cv2.COLOR_GRAY2BGR)"
        ),
        "cleanup": "",
        "params": {},
        "output_type": "image",
    },
    "Canny": {
        "init": "",
        "process": (
            "_gray{node_id} = cv2.cvtColor({input}, cv2.COLOR_BGR2GRAY)\n"
            "_canny{node_id} = cv2.Canny(_gray{node_id}, {canny_min}, {canny_max})\n"
            "{output} = cv2.cvtColor(_canny{node_id}, cv2.COLOR_GRAY2BGR)"
        ),
        "cleanup": "",
        "params": {"canny_min": 100, "canny_max": 200},
        "output_type": "image",
    },
    "Threshold": {
        "init": "",
        "process": (
            "_gray{node_id} = cv2.cvtColor({input}, cv2.COLOR_BGR2GRAY)\n"
            "_, _thresh{node_id} = cv2.threshold(_gray{node_id}, {binary_threshold}, 255, cv2.{threshold_type})\n"
            "{output} = cv2.cvtColor(_thresh{node_id}, cv2.COLOR_GRAY2BGR)"
        ),
        "cleanup": "",
        "params": {"threshold_type": "THRESH_BINARY", "binary_threshold": 127},
        "output_type": "image",
    },
    "Brightness": {
        "init": "",
        "process": "{output} = cv2.convertScaleAbs({input}, alpha=1.0, beta={brightness})",
        "cleanup": "",
        "params": {"brightness": 0},
        "output_type": "image",
    },
    "Contrast": {
        "init": "",
        "process": "{output} = cv2.convertScaleAbs({input}, alpha={contrast}, beta=0)",
        "cleanup": "",
        "params": {"contrast": 1.0},
        "output_type": "image",
    },
    "Flip": {
        "init": "",
        "process": "{output} = cv2.flip({input}, {flipcode})",
        "cleanup": "",
        "params": {"hflip": False, "vflip": False},
        "output_type": "image",
    },
    "Resize": {
        "init": "",
        "process": "{output} = cv2.resize({input}, ({width}, {height}), interpolation=cv2.{interpolation})",
        "cleanup": "",
        "params": {"width": 640, "height": 480, "interpolation": "INTER_LINEAR"},
        "output_type": "image",
    },
    "Crop": {
        "init": "",
        "process": (
            "_h{node_id}, _w{node_id} = {input}.shape[:2]\n"
            "_x1_{node_id}, _x2_{node_id} = int({min_x} * _w{node_id}), int({max_x} * _w{node_id})\n"
            "_y1_{node_id}, _y2_{node_id} = int({min_y} * _h{node_id}), int({max_y} * _h{node_id})\n"
            "{output} = {input}[_y1_{node_id}:_y2_{node_id}, _x1_{node_id}:_x2_{node_id}]"
        ),
        "cleanup": "",
        "params": {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0},
        "output_type": "image",
    },
    "OmnidirectionalViewer": {
        "init": "_omni_phi_{node_id}, _omni_theta_{node_id} = _create_omni_map({pitch}, {yaw}, {roll}, {imagepoint})",
        "process": "{output} = _remap_omni({input}, _omni_phi_{node_id}, _omni_theta_{node_id})",
        "cleanup": "",
        "params": {"pitch": 0, "yaw": 0, "roll": 0, "imagepoint": 0.0},
        "output_type": "image",
    },
    "GammaCorrection": {
        "init": "",
        "process": (
            "_gamma{node_id} = {gamma}\n"
            "_table{node_id} = (np.arange(256) / 255) ** _gamma{node_id} * 255\n"
            "_table{node_id} = np.clip(_table{node_id}, 0, 255).astype(np.uint8)\n"
            "{output} = cv2.LUT({input}, _table{node_id})"
        ),
        "cleanup": "",
        "params": {"gamma": 1.0},
        "output_type": "image",
    },
    "EqualizeHist": {
        "init": "",
        "process": (
            "_hsv{node_id} = cv2.cvtColor({input}, cv2.COLOR_BGR2HSV)\n"
            "_hsv{node_id}[:, :, 2] = cv2.equalizeHist(_hsv{node_id}[:, :, 2])\n"
            "{output} = cv2.cvtColor(_hsv{node_id}, cv2.COLOR_HSV2BGR)"
        ),
        "cleanup": "",
        "params": {},
        "output_type": "image",
    },
    "Sepia": {
        "init": "_sepia{node_id}_kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])",
        "process": (
            "{output} = cv2.transform({input}, _sepia{node_id}_kernel)\n"
            "{output} = np.clip({output}, 0, 255).astype(np.uint8)"
        ),
        "cleanup": "",
        "params": {},
        "output_type": "image",
    },
    "ApplyColorMap": {
        "init": "",
        "process": "{output} = cv2.applyColorMap({input}, cv2.{colormap})",
        "cleanup": "",
        "params": {"colormap": "COLORMAP_JET"},
        "output_type": "image",
    },
    "HSV": {
        "init": "",
        "process": (
            "_hsv{node_id} = cv2.cvtColor({input}, cv2.COLOR_BGR2HSV)\n"
            "_h{node_id}, _s{node_id}, _v{node_id} = cv2.split(_hsv{node_id})\n"
            "_h{node_id} = ((_h{node_id}.astype(np.int16) + {h_add}) % 180).astype(np.uint8)\n"
            "_s{node_id} = np.clip(_s{node_id}.astype(np.int16) + {s_add}, 0, 255).astype(np.uint8)\n"
            "_v{node_id} = np.clip(_v{node_id}.astype(np.int16) + {v_add}, 0, 255).astype(np.uint8)\n"
            "_hsv{node_id} = cv2.merge([_h{node_id}, _s{node_id}, _v{node_id}])\n"
            "{output} = cv2.cvtColor(_hsv{node_id}, cv2.COLOR_HSV2BGR)"
        ),
        "cleanup": "",
        "params": {"h_add": 0, "s_add": 0, "v_add": 0},
        "output_type": "image",
    },
    "SimpleFilter": {
        "init": "",
        "process": (
            "_kernel{node_id} = np.array([[{x0y0}, {x1y0}, {x2y0}], [{x0y1}, {x1y1}, {x2y1}], [{x0y2}, {x1y2}, {x2y2}]]) * {filter_k}\n"
            "{output} = cv2.filter2D({input}, -1, _kernel{node_id})"
        ),
        "cleanup": "",
        "params": {
            "x0y0": 0,
            "x1y0": 0,
            "x2y0": 0,
            "x0y1": 0,
            "x1y1": 1,
            "x2y1": 0,
            "x0y2": 0,
            "x1y2": 0,
            "x2y2": 0,
            "filter_k": 1.0,
        },
        "output_type": "image",
    },
}
