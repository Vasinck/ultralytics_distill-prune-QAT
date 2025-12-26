# QAT（量化感知训练）模块
# 自研通用目标检测框架 v1.2 的一部分

from .module_fuser import fuse_yolo_modules, replace_silu_with_relu
from .qconfig_presets import get_qconfig_preset, SUPPORTED_BACKENDS
from .onnx_exporter import export_qat_to_onnx, verify_onnx_model

__all__ = [
    'fuse_yolo_modules',
    'replace_silu_with_relu',
    'get_qconfig_preset',
    'SUPPORTED_BACKENDS',
    'export_qat_to_onnx',
    'verify_onnx_model',
]
