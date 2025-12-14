"""
コード生成オーケストレーター

このモジュールは、Node Editor JSONからPythonコードを生成するメイン関数を提供します。
各サブモジュールを組み合わせてコードを生成します。
"""

from json2demo.json_parsing import (
    extract_node_info,
    extract_links,
    topological_sort,
)
from json2demo.code_generation.dependency_analyzer import (
    check_requires_numpy,
    check_requires_dl,
    analyze_node_types,
    identify_skipped_nodes,
    build_variable_mapping,
)
from json2demo.code_generation.import_generator import (
    generate_imports,
    generate_dl_imports,
    generate_helper_functions,
    generate_perf_helper,
)
from json2demo.code_generation.argparse_generator import (
    collect_input_nodes,
    build_arg_suffix_mappings,
    generate_argparse_code,
)
from json2demo.code_generation.init_generator import (
    generate_initialization_code,
    format_init_code_lines,
    format_cleanup_code_lines,
    find_first_input_source,
)
from json2demo.code_generation.pass_generator import (
    classify_nodes,
    build_detect_input_mapping,
    generate_pass1_transformations,
    generate_pass2_detection,
    generate_pass3_drawing,
    generate_pass4_deferred,
)
from json2demo.code_generation.output_generator import (
    generate_output_phase,
    generate_terminal_nodes_display,
    generate_key_handling,
    generate_main_footer,
)


def generate_code(json_data, output_path=None, enable_perf=False, window_prefix=""):
    """
    OpenCVデモコードを生成

    Args:
        json_data: 解析済みJSONデータ
        output_path: 出力ファイルパス (Noneの場合は標準出力)
        enable_perf: パフォーマンス計測を有効にするか
        window_prefix: imshowウィンドウ名のプレフィックス

    Returns:
        str: 生成されたコード文字列
    """
    nodes = extract_node_info(json_data)
    links = extract_links(json_data)
    sorted_nodes = topological_sort(nodes, links)

    requires_dl = check_requires_dl(nodes)
    requires_numpy = check_requires_numpy(nodes)

    # ヘルパー関数が必要なノードタイプをチェック
    helper_node_types = {"OmnidirectionalViewer", "MOT"}
    requires_helpers = requires_dl or any(
        nodes[nk]["type"] in helper_node_types for nk in nodes
    )
    skipped_nodes = identify_skipped_nodes(sorted_nodes, nodes)
    var_mapping = build_variable_mapping(sorted_nodes, nodes, links, skipped_nodes)

    (
        used_node_types,
        needs_mediapipe,
        needs_onnx,
        face_uses_yunet,
        face_uses_mediapipe,
        face_uses_facemesh,
        od_model_types,
    ) = analyze_node_types(sorted_nodes, nodes, links)

    video_nodes, image_nodes, webcam_nodes, framepos_nodes = collect_input_nodes(
        sorted_nodes, nodes, links
    )
    (
        video_arg_suffix,
        image_arg_suffix,
        webcam_arg_suffix,
        framepos_arg_suffix,
    ) = build_arg_suffix_mappings(video_nodes, image_nodes, webcam_nodes, framepos_nodes)
    needs_args = video_nodes or image_nodes or webcam_nodes or framepos_nodes

    code_lines = []

    # 1. Import文を生成
    code_lines.extend(
        generate_imports(needs_onnx, requires_numpy, enable_perf, used_node_types, needs_args)
    )

    if requires_dl:
        code_lines.extend(generate_dl_imports(needs_mediapipe, needs_onnx))

    # 2. parse_args関数を生成
    if needs_args:
        code_lines.extend(
            generate_argparse_code(
                video_nodes, image_nodes, webcam_nodes, framepos_nodes,
                video_arg_suffix, image_arg_suffix, webcam_arg_suffix, framepos_arg_suffix,
            )
        )

    # 3. main関数を生成
    code_lines.append("")
    code_lines.append("")
    code_lines.append("def main():")

    if needs_args:
        code_lines.append("    args = parse_args()")
        code_lines.append("")

    init_codes, cleanup_codes = generate_initialization_code(
        sorted_nodes, nodes, links, var_mapping,
        video_arg_suffix, webcam_arg_suffix, image_arg_suffix, framepos_arg_suffix,
    )
    code_lines.extend(format_init_code_lines(init_codes))

    # VideoWriterがある場合のみ_cap_fpsエイリアスを生成
    if "VideoWriter" in used_node_types:
        first_input_id = find_first_input_source(sorted_nodes, nodes)
        if first_input_id is not None:
            code_lines.append("")
            code_lines.append(f"    _cap_fps = _cap_fps{first_input_id}")

    code_lines.append("")
    code_lines.append("    while True:")

    if enable_perf:
        code_lines.append("        # Performance measurement")
        code_lines.append("        _perf_times = {}")
        code_lines.append("        _perf_total_start = time.perf_counter()")
        code_lines.append("")

    process_nodes, output_nodes = classify_nodes(sorted_nodes, nodes, links, var_mapping)
    detect_input_mapping = build_detect_input_mapping(process_nodes, links, var_mapping)

    pass1_lines, deferred_nodes = generate_pass1_transformations(
        process_nodes, links, nodes, var_mapping, enable_perf, window_prefix
    )
    code_lines.extend(pass1_lines)

    code_lines.extend(
        generate_pass2_detection(
            process_nodes, links, nodes, var_mapping, detect_input_mapping, enable_perf, window_prefix
        )
    )

    code_lines.extend(
        generate_pass3_drawing(process_nodes, links, nodes, var_mapping, enable_perf, window_prefix)
    )

    code_lines.extend(
        generate_pass4_deferred(deferred_nodes, links, nodes, var_mapping, enable_perf, window_prefix)
    )

    code_lines.extend(
        generate_output_phase(output_nodes, process_nodes, links, nodes, var_mapping, enable_perf, window_prefix)
    )

    code_lines.extend(
        generate_terminal_nodes_display(
            process_nodes, output_nodes, links, nodes, var_mapping, enable_perf, window_prefix
        )
    )

    code_lines.extend(generate_key_handling())

    code_lines.extend(format_cleanup_code_lines(cleanup_codes))
    code_lines.append("    cv2.destroyAllWindows()")

    # 4. ヘルパー関数・描画関数を生成（main関数の後）
    if requires_helpers:
        code_lines.append("")
        code_lines.extend(
            generate_helper_functions(
                used_node_types,
                face_uses_mediapipe,
                face_uses_facemesh,
                face_uses_yunet,
                od_model_types,
                enable_perf,
            )
        )
    elif enable_perf:
        code_lines.append("")
        code_lines.extend(generate_perf_helper())
    else:
        code_lines.append("")

    # 5. if __name__ == "__main__": main()
    code_lines.extend(generate_main_footer())

    generated_code = "\n".join(code_lines)
    generated_code = generated_code.replace("{{", "{").replace("}}", "}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(generated_code)
        print(f"Generated: {output_path}")
    else:
        print(generated_code)

    return generated_code


__all__ = ["generate_code"]
