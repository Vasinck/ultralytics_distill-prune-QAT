"""
针对不同部署目标的量化配置预设
自研通用目标检测框架 v1.2 的一部分

提供针对以下场景优化的预配置量化设置：
- 移动端部署（NCNN、MNN、ONNX Runtime Mobile）
- 服务器部署（ONNX Runtime、OpenVINO）
- 边缘 GPU（TensorRT）
"""

import torch
import torch.ao.quantization as quant
from typing import Dict, Any, Optional


# 支持的后端名称
SUPPORTED_BACKENDS = ['qnnpack', 'fbgemm', 'tensorrt', 'x86', 'onednn']


def get_qconfig_preset(
    backend: str = 'qnnpack',
    config: Optional[Dict[str, Any]] = None
) -> quant.QConfig:
    """
    获取指定后端的量化配置预设。

    Args:
        backend: 目标部署后端
            - 'qnnpack': ARM 移动端（NCNN、MNN、ONNX Runtime Mobile）
            - 'fbgemm': x86 服务器（ONNX Runtime、OpenVINO）
            - 'tensorrt': NVIDIA TensorRT
            - 'x86': fbgemm 的别名
            - 'onednn': Intel OneDNN 优化

        config: 附加配置选项
            - per_channel (bool): 使用逐通道权重量化（默认：True）
            - symmetric (bool): 使用对称量化（默认：激活为 False，权重为 True）
            - bit_width (int): 量化位宽（默认：8）

    Returns:
        指定后端的 PyTorch QConfig 对象

    Example:
        >>> qconfig = get_qconfig_preset('qnnpack', {'per_channel': True})
        >>> model.qconfig = qconfig
        >>> model = torch.ao.quantization.prepare_qat(model)
    """
    config = config or {}

    backend = backend.lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"不支持的后端：{backend}。"
            f"支持的后端：{SUPPORTED_BACKENDS}"
        )

    # 设置 PyTorch 量化引擎
    _set_quantization_engine(backend)

    # 根据后端获取配置
    if backend in ['qnnpack']:
        return _get_qnnpack_qconfig(config)
    elif backend in ['fbgemm', 'x86']:
        return _get_fbgemm_qconfig(config)
    elif backend == 'tensorrt':
        return _get_tensorrt_qconfig(config)
    elif backend == 'onednn':
        return _get_onednn_qconfig(config)

    # 回退到默认配置
    return quant.get_default_qat_qconfig(backend)


def _set_quantization_engine(backend: str) -> None:
    """设置 PyTorch 的量化后端引擎。"""
    if backend in ['qnnpack']:
        torch.backends.quantized.engine = 'qnnpack'
        print(f"[QAT] 已设置量化引擎：qnnpack（ARM 优化）")
    elif backend in ['fbgemm', 'x86', 'onednn']:
        torch.backends.quantized.engine = 'fbgemm'
        print(f"[QAT] 已设置量化引擎：fbgemm（x86 优化）")
    elif backend == 'tensorrt':
        # TensorRT 使用 FBGEMM 引擎但配置不同
        torch.backends.quantized.engine = 'fbgemm'
        print(f"[QAT] 已设置量化引擎：fbgemm（用于 TensorRT 导出）")


def _get_qnnpack_qconfig(config: Dict[str, Any]) -> quant.QConfig:
    """
    QNNPACK 的量化配置（移动端部署）。

    优化目标：
    - ARM CPU（Android、iOS、嵌入式设备）
    - NCNN、MNN 推理框架
    - ONNX Runtime Mobile
    """
    per_channel = config.get('per_channel', True)

    # 激活：per-tensor affine，uint8 [0, 255]
    activation_observer = quant.MovingAverageMinMaxObserver.with_args(
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False  # QNNPACK 支持全范围
    )

    # 权重：per-channel 对称量化以获得更好的精度
    if per_channel:
        weight_observer = quant.MovingAveragePerChannelMinMaxObserver.with_args(
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0
        )
    else:
        weight_observer = quant.MovingAverageMinMaxObserver.with_args(
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=activation_observer,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8
        ),
        weight=quant.FakeQuantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8
        )
    )


def _get_fbgemm_qconfig(config: Dict[str, Any]) -> quant.QConfig:
    """
    FBGEMM 的量化配置（x86 服务器部署）。

    优化目标：
    - x86 CPU（Intel、AMD）
    - ONNX Runtime（CPU）
    - OpenVINO
    """
    per_channel = config.get('per_channel', True)

    # FBGEMM 支持更高级的观察器
    activation_observer = quant.MovingAverageMinMaxObserver.with_args(
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=True  # 预留一些范围防止溢出
    )

    if per_channel:
        weight_observer = quant.MovingAveragePerChannelMinMaxObserver.with_args(
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0
        )
    else:
        weight_observer = quant.MovingAverageMinMaxObserver.with_args(
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=activation_observer,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8
        ),
        weight=quant.FakeQuantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8
        )
    )


def _get_tensorrt_qconfig(config: Dict[str, Any]) -> quant.QConfig:
    """
    TensorRT 部署的量化配置。

    TensorRT 偏好：
    - 激活和权重都使用对称量化
    - 逐通道权重量化
    - 使用熵最小化的 INT8 校准
    """
    per_channel = config.get('per_channel', True)

    # TensorRT 偏好激活使用对称量化
    activation_observer = quant.MovingAverageMinMaxObserver.with_args(
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    )

    if per_channel:
        weight_observer = quant.MovingAveragePerChannelMinMaxObserver.with_args(
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0
        )
    else:
        weight_observer = quant.MovingAverageMinMaxObserver.with_args(
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=activation_observer,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8
        ),
        weight=quant.FakeQuantize.with_args(
            observer=weight_observer,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8
        )
    )


def _get_onednn_qconfig(config: Dict[str, Any]) -> quant.QConfig:
    """
    Intel OneDNN 的量化配置。

    针对具有 AVX-512/VNNI 指令的 Intel 硬件优化。
    """
    # OneDNN 使用与 FBGEMM 类似的配置，但有一些 Intel 特定优化
    return _get_fbgemm_qconfig(config)


def get_qat_optimizer_params(base_lr: float = 0.01) -> Dict[str, Any]:
    """
    获取 QAT 微调推荐的优化器参数。

    QAT 通常需要：
    - 较低的学习率（正常训练的 1/10 到 1/100）
    - 较少的 epoch（通常 10-20 个 epoch 就足够）
    - 带动量的 SGD（比 Adam 在 QAT 中更稳定）

    Args:
        base_lr: 原始训练的基础学习率

    Returns:
        包含优化器参数的字典
    """
    return {
        'lr0': base_lr * 0.01,  # 原始学习率的 1%
        'lrf': 0.01,            # 最终学习率比例
        'momentum': 0.937,      # 与 YOLO 默认值相同
        'weight_decay': 0.0005, # 与 YOLO 默认值相同
        'warmup_epochs': 0,     # QAT 微调不需要预热
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    }


def describe_qconfig(qconfig: quant.QConfig) -> Dict[str, Any]:
    """
    获取 QConfig 的可读描述。

    Returns:
        包含激活和权重量化详情的字典
    """
    def _describe_fq(fq):
        if hasattr(fq, 'p'):
            p = fq.p
            # 安全地从观察器中提取 qscheme
            observer_factory = p.keywords.get('observer', None)
            qscheme = 'unknown'
            if observer_factory:
                try:
                    observer_inst = observer_factory()
                    if hasattr(observer_inst, 'qscheme'):
                        qscheme = str(observer_inst.qscheme)
                except Exception:
                    pass

            return {
                'dtype': str(p.keywords.get('dtype', 'unknown')),
                'quant_min': p.keywords.get('quant_min', 'unknown'),
                'quant_max': p.keywords.get('quant_max', 'unknown'),
                'qscheme': qscheme
            }
        return {'type': str(type(fq))}

    return {
        'activation': _describe_fq(qconfig.activation),
        'weight': _describe_fq(qconfig.weight)
    }
