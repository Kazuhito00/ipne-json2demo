"""
Argparse生成モジュール

生成コードのCLI引数処理を生成
"""

from json2demo.node_templates import NODE_TEMPLATES
from json2demo.json_parsing import extract_setting_value, get_linked_param_value


def collect_input_nodes(sorted_nodes, nodes, links):
    """
    入力ノード情報を収集

    Returns:
        tuple: (video_nodes, image_nodes, webcam_nodes, framepos_nodes)
    """
    video_nodes = []
    image_nodes = []
    webcam_nodes = []
    framepos_nodes = []

    for node_key in sorted_nodes:
        node_info = nodes[node_key]
        node_type = node_info["type"]
        node_id = node_info["id"]

        if node_type == "WebCam":
            webcam_nodes.append(node_key)

        if node_type in NODE_TEMPLATES:
            template = NODE_TEMPLATES[node_type]
            if template.get("uses_args") == "video":
                video_nodes.append(node_key)
            elif template.get("uses_args") == "video_framepos":
                linked_value = get_linked_param_value(node_key, "frame_pos", links, nodes)
                if linked_value is not None:
                    fp_default = linked_value
                else:
                    fp_default = extract_setting_value(
                        node_info["setting"], node_id, node_type, "frame_pos", 0
                    )
                framepos_nodes.append((node_key, fp_default))
            elif template.get("uses_args") == "image":
                image_nodes.append(node_key)

    return video_nodes, image_nodes, webcam_nodes, framepos_nodes


def build_arg_suffix_mappings(video_nodes, image_nodes, webcam_nodes, framepos_nodes):
    """
    引数サフィックスマッピングを構築

    Returns:
        tuple: (video_arg_suffix, image_arg_suffix, webcam_arg_suffix, framepos_arg_suffix)
    """
    video_arg_suffix = {}
    image_arg_suffix = {}
    webcam_arg_suffix = {}
    framepos_arg_suffix = {}

    for index, node_key in enumerate(video_nodes):
        video_arg_suffix[node_key] = "" if len(video_nodes) == 1 else str(index + 1)

    for index, node_key in enumerate(image_nodes):
        image_arg_suffix[node_key] = "" if len(image_nodes) == 1 else str(index + 1)

    for index, node_key in enumerate(webcam_nodes):
        webcam_arg_suffix[node_key] = "" if len(webcam_nodes) == 1 else str(index + 1)

    for index, (node_key, _) in enumerate(framepos_nodes):
        framepos_arg_suffix[node_key] = "" if len(framepos_nodes) == 1 else str(index + 1)

    return video_arg_suffix, image_arg_suffix, webcam_arg_suffix, framepos_arg_suffix


def generate_argparse_code(video_nodes, image_nodes, webcam_nodes, framepos_nodes,
                           video_arg_suffix, image_arg_suffix, webcam_arg_suffix,
                           framepos_arg_suffix):
    """
    argparse用のコードを生成

    Returns:
        list: argparseコードの行リスト
    """
    code_lines = []
    code_lines.append("")
    code_lines.append("")
    code_lines.append("def parse_args():")
    code_lines.append(
        '    parser = argparse.ArgumentParser(description="OpenCV Demo")'
    )

    for node_key in video_nodes:
        suffix = video_arg_suffix[node_key]
        if suffix == "":
            code_lines.append(
                '    parser.add_argument("--video", type=str, default="sample.mp4", help="Path to video file")'
            )
        else:
            code_lines.append(
                f'    parser.add_argument("--video{suffix}", type=str, default="sample{suffix}.mp4", help="Path to video file {suffix}")'
            )

    for node_key, fp_default in framepos_nodes:
        suffix = framepos_arg_suffix[node_key]
        if suffix == "":
            code_lines.append(
                f'    parser.add_argument("--frame_pos", type=int, default={fp_default}, help="Frame position")'
            )
            code_lines.append(
                '    parser.add_argument("--video", type=str, default="sample.mp4", help="Path to video file")'
            )
        else:
            code_lines.append(
                f'    parser.add_argument("--frame_pos{suffix}", type=int, default={fp_default}, help="Frame position {suffix}")'
            )
            code_lines.append(
                f'    parser.add_argument("--video{suffix}", type=str, default="sample{suffix}.mp4", help="Path to video file {suffix}")'
            )

    for node_key in image_nodes:
        suffix = image_arg_suffix[node_key]
        if suffix == "":
            code_lines.append(
                '    parser.add_argument("--image", type=str, default="sample.jpg", help="Path to image file")'
            )
        else:
            code_lines.append(
                f'    parser.add_argument("--image{suffix}", type=str, default="sample{suffix}.jpg", help="Path to image file {suffix}")'
            )

    for index, node_key in enumerate(webcam_nodes):
        suffix = webcam_arg_suffix[node_key]
        if suffix == "":
            code_lines.append(
                '    parser.add_argument("--camera", type=int, default=0, help="Camera ID")'
            )
            code_lines.append(
                '    parser.add_argument("--width", type=int, default=960, help="Camera width")'
            )
            code_lines.append(
                '    parser.add_argument("--height", type=int, default=540, help="Camera height")'
            )
        else:
            code_lines.append(
                f'    parser.add_argument("--camera{suffix}", type=int, default={index}, help="Camera ID {suffix}")'
            )
            code_lines.append(
                f'    parser.add_argument("--width{suffix}", type=int, default=960, help="Camera width {suffix}")'
            )
            code_lines.append(
                f'    parser.add_argument("--height{suffix}", type=int, default=540, help="Camera height {suffix}")'
            )

    code_lines.append("    return parser.parse_args()")

    return code_lines
