"""
初期化コード生成モジュール

ノードの初期化コードとクリーンアップコードを生成
"""

from json2demo.node_templates import NODE_TEMPLATES
from json2demo.model_mappings import ONNX_MODEL_FILE_MAPPING
from json2demo.json_parsing import extract_setting_value, get_linked_param_value
from json2demo.code_generation.dependency_analyzer import get_actual_node_type


def generate_initialization_code(sorted_nodes, nodes, links, var_mapping,
                                  video_arg_suffix, webcam_arg_suffix,
                                  image_arg_suffix, framepos_arg_suffix):
    """
    ノードの初期化コードを生成

    Returns:
        tuple: (init_codes, cleanup_codes)
    """
    init_codes = []
    cleanup_codes = []

    for node_key in sorted_nodes:
        node_info = nodes[node_key]
        node_type = node_info["type"]
        setting = node_info["setting"]

        actual_type = get_actual_node_type(node_type, setting, node_info, nodes)

        if actual_type not in NODE_TEMPLATES:
            print(f"Warning: Unknown node type '{actual_type}', skipping...")
            continue

        template = NODE_TEMPLATES[actual_type]
        if template["init"]:
            init_code = template["init"]

            for param, default in template["params"].items():
                linked_value = get_linked_param_value(node_key, param, links, nodes)
                if linked_value is not None:
                    value = linked_value
                else:
                    value = extract_setting_value(
                        setting, node_info["id"], node_info["type"], param, default
                    )

                if isinstance(value, str):
                    init_code = init_code.replace(f"{{{param}}}", value)
                    if param == "model":
                        model_file = ONNX_MODEL_FILE_MAPPING.get(value, "unknown.onnx")
                        init_code = init_code.replace("{model_path}", model_file)
                else:
                    init_code = init_code.replace(f"{{{param}}}", str(value))

            init_code = init_code.replace("{output}", var_mapping[node_key])
            init_code = init_code.replace("{node_id}", node_info["id"])

            if node_key in video_arg_suffix:
                suffix = video_arg_suffix[node_key]
                init_code = init_code.replace("args.video", f"args.video{suffix}")
            if node_key in webcam_arg_suffix:
                suffix = webcam_arg_suffix[node_key]
                init_code = init_code.replace("args.camera", f"args.camera{suffix}")
                init_code = init_code.replace("args.width", f"args.width{suffix}")
                init_code = init_code.replace("args.height", f"args.height{suffix}")
            if node_key in image_arg_suffix:
                suffix = image_arg_suffix[node_key]
                init_code = init_code.replace("args.image", f"args.image{suffix}")
            if node_key in framepos_arg_suffix:
                suffix = framepos_arg_suffix[node_key]
                init_code = init_code.replace("args.frame_pos", f"args.frame_pos{suffix}")
                init_code = init_code.replace("args.video", f"args.video{suffix}")

            init_codes.append(init_code)

        if template["cleanup"]:
            cleanup_code = template["cleanup"].replace("{node_id}", node_info["id"])
            cleanup_codes.append(cleanup_code)

    return init_codes, cleanup_codes


def format_init_code_lines(init_codes, indent="    "):
    """
    初期化コードをフォーマット（インデント付き）

    Args:
        init_codes: 初期化コードリスト
        indent: インデント文字列

    Returns:
        list: フォーマットされたコード行リスト
    """
    code_lines = []
    for index, init_code in enumerate(init_codes):
        for line in init_code.split("\n"):
            code_lines.append(f"{indent}{line}")
        if index < len(init_codes) - 1:
            code_lines.append("")
    return code_lines


def format_cleanup_code_lines(cleanup_codes, indent="    "):
    """
    クリーンアップコードをフォーマット

    Args:
        cleanup_codes: クリーンアップコードリスト
        indent: インデント文字列

    Returns:
        list: フォーマットされたコード行リスト
    """
    code_lines = []
    for cleanup_code in cleanup_codes:
        for line in cleanup_code.split("\n"):
            code_lines.append(f"{indent}{line}")
    return code_lines


def find_first_input_source(sorted_nodes, nodes):
    """
    最初の入力ソースノードを見つける

    Returns:
        str or None: 最初の入力ソースノードのID
    """
    input_source_types = {"Video", "WebCam", "RTSPInput", "VideoSetFramePos", "ScreenCapture"}
    for node_key in sorted_nodes:
        node_info = nodes[node_key]
        if node_info["type"] in input_source_types:
            return node_info["id"]
    return None
