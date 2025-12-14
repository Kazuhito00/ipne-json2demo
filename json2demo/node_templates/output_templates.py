"""
出力ノード・描画ノード・その他ノード用テンプレート

ResultImage, ImageConcat, VideoWriter, MOT等のノードを定義
"""

OUTPUT_NODE_TEMPLATES = {
    # 描画ノード
    "ImageConcat": {
        "init": "",
        "process": "",  # 動的生成
        "cleanup": "",
        "params": {},
        "output_type": "image",
        "multi_input": True,
        "dynamic_codegen": True,
    },
    "ImageAlphaBlend": {
        "init": "",
        "process": (
            "_img{node_id}_resized = cv2.resize({input2}, ({input1}.shape[1], {input1}.shape[0]))\n"
            "{output} = cv2.addWeighted({input1}, {alpha}, _img{node_id}_resized, {beta}, {gamma_blend})"
        ),
        "cleanup": "",
        "params": {"alpha": 0.5, "beta": 0.5, "gamma_blend": 0},
        "output_type": "image",
        "multi_input": True,
    },
    "PutText": {
        "init": "",
        "process": (
            '{output} = {input}.copy()\n'
            'cv2.putText({output}, "{text}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)'
        ),
        "cleanup": "",
        "params": {"text": "Text"},
        "output_type": "image",
    },
    "DrawInformation": {
        "init": "",
        "process": "{output} = {input}",
        "cleanup": "",
        "params": {},
        "output_type": "image",
        "skip_codegen": True,
    },
    # 出力ノード
    "ResultImage": {
        "init": "",
        "process": 'cv2.imshow("Result_{node_id}", {input})',
        "cleanup": "",
        "params": {},
        "output_type": "display",
    },
    "ResultImageLarge": {
        "init": "",
        "process": 'cv2.imshow("ResultLarge_{node_id}", {input})',
        "cleanup": "",
        "params": {},
        "output_type": "display",
    },
    # その他ノード
    "ExecPythonCode": {
        "init": "",
        "process": (
            "input_image = {input}\n"
            "{code}\n"
            "{output} = output_image"
        ),
        "cleanup": "",
        "params": {"code": "output_image = input_image"},
        "output_type": "image",
    },
    "MOT": {
        "init": "_mot{node_id}_tracker = MOTTracker(dt=1/30, max_staleness={max_staleness})",
        "process": (
            "_mot{node_id}_detections = [Detection(box=box) for box in {src_boxes}]\n"
            "_mot{node_id}_tracker.step(detections=_mot{node_id}_detections)\n"
            "_mot{node_id}_tracks = _mot{node_id}_tracker.active_tracks()"
        ),
        "draw": "{output} = _mot_draw_tracks({src_input}, _mot{node_id}_tracks)",
        "cleanup": "",
        "params": {"max_staleness": 5},
        "output_type": "image",
        "requires_detection": True,
        "skip_src_draw": True,
    },
    "VideoWriter": {
        "init": '_fourcc{node_id} = cv2.VideoWriter_fourcc(*"mp4v")\n_video_writer{node_id} = None',
        "process": (
            'if _video_writer{node_id} is None:\n'
            '    _h{node_id}, _w{node_id} = {input}.shape[:2]\n'
            '    _video_writer{node_id} = cv2.VideoWriter("output_{node_id}.mp4", _fourcc{node_id}, _cap_fps, (_w{node_id}, _h{node_id}))\n'
            '_video_writer{node_id}.write({input})\n'
            'cv2.imshow("VideoWriter_{node_id}", {input})'
        ),
        "cleanup": "if _video_writer{node_id} is not None:\n    _video_writer{node_id}.release()",
        "params": {},
        "output_type": "display",
    },
    # 分析ノード
    "BRISQUE": {
        "init": '_brisque{node_id}_model_path = "brisque_model_live.yml"\n_brisque{node_id}_range_path = "brisque_range_live.yml"',
        "process": (
            '_brisque{node_id}_score = cv2.quality.QualityBRISQUE_compute({input}, _brisque{node_id}_model_path, _brisque{node_id}_range_path)[0]\n'
            '{output} = {input}.copy()\n'
            '_brisque{node_id}_h, _brisque{node_id}_w = {output}.shape[:2]\n'
            '_brisque{node_id}_text = f"BRISQUE: {{_brisque{node_id}_score:.2f}}"\n'
            '(_brisque{node_id}_tw, _brisque{node_id}_th), _ = cv2.getTextSize(_brisque{node_id}_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)\n'
            'cv2.putText({output}, _brisque{node_id}_text, (_brisque{node_id}_w - _brisque{node_id}_tw - 10, _brisque{node_id}_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)\n'
            'cv2.putText({output}, _brisque{node_id}_text, (_brisque{node_id}_w - _brisque{node_id}_tw - 10, _brisque{node_id}_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)'
        ),
        "cleanup": "",
        "params": {},
        "output_type": "image",
    },
    "RGBHistgram": {
        "init": "",
        "process": (
            '# RGB Histogram visualization\n'
            '_hist{node_id}_img = np.zeros((200, 256, 3), dtype=np.uint8)\n'
            'for i, col in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):\n'
            '    _hist{node_id} = cv2.calcHist([{input}], [i], None, [256], [0, 256])\n'
            '    cv2.normalize(_hist{node_id}, _hist{node_id}, 0, 200, cv2.NORM_MINMAX)\n'
            '    for j in range(256):\n'
            '        cv2.line(_hist{node_id}_img, (j, 200), (j, 200 - int(_hist{node_id}[j][0])), col)\n'
            'cv2.imshow("RGBHistgram_{node_id}_Input", {input})\n'
            'cv2.imshow("RGBHistgram_{node_id}_Histogram", _hist{node_id}_img)'
        ),
        "cleanup": "",
        "params": {},
        "output_type": "none",
    },
}
