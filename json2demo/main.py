#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
json2demo CLIエントリーポイント

Node Editor JSONファイルからOpenCV HighGUIデモコードを生成

Usage:
    python -m json2demo <json_file> [output_file] [--perf]
"""

import argparse
import sys
from pathlib import Path

from json2demo import parse_json, generate_code


def main():
    """CLIメイン関数"""
    parser = argparse.ArgumentParser(
        description="Node Editor JSONからOpenCV HighGUIデモコードを生成"
    )
    parser.add_argument("json_file", help="入力JSONファイル")
    parser.add_argument(
        "output_file", nargs="?", help="出力Pythonファイル（省略時は標準出力）"
    )
    parser.add_argument("--perf", action="store_true", help="処理時間計測コードを生成")

    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    json_data = parse_json(json_path)
    output_path = args.output_file

    # 出力ファイル名からウィンドウプレフィックスを生成
    window_prefix = ""
    if output_path:
        output_name = Path(output_path).stem
        # _perf サフィックスを除去
        if output_name.endswith("_perf"):
            output_name = output_name[:-5]
        window_prefix = output_name

    generate_code(json_data, output_path, enable_perf=args.perf, window_prefix=window_prefix)


if __name__ == "__main__":
    main()
