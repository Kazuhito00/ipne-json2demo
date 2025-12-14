"""
Node processing templates for code generation.

全ノードテンプレートを統合してNODE_TEMPLATESとしてエクスポート
"""

from json2demo.node_templates.input_templates import INPUT_NODE_TEMPLATES
from json2demo.node_templates.processing_templates import PROCESSING_NODE_TEMPLATES
from json2demo.node_templates.output_templates import OUTPUT_NODE_TEMPLATES
from json2demo.node_templates.dl_templates import DL_NODE_TEMPLATES

# 全テンプレートを統合
NODE_TEMPLATES = {}
NODE_TEMPLATES.update(INPUT_NODE_TEMPLATES)
NODE_TEMPLATES.update(PROCESSING_NODE_TEMPLATES)
NODE_TEMPLATES.update(OUTPUT_NODE_TEMPLATES)
NODE_TEMPLATES.update(DL_NODE_TEMPLATES)

__all__ = [
    "NODE_TEMPLATES",
    "INPUT_NODE_TEMPLATES",
    "PROCESSING_NODE_TEMPLATES",
    "OUTPUT_NODE_TEMPLATES",
    "DL_NODE_TEMPLATES",
]
