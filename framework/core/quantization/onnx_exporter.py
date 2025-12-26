"""
量化模型的 ONNX 导出工具
自研通用目标检测框架 v1.3 的一部分

处理 QAT 模型到 ONNX 格式的导出，
针对不同部署目标正确处理量化算子。
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union


class ONNXExportWrapper(nn.Module):
    """
    用于 ONNX 导出的简单包装器，处理 YOLO 复杂的 forward 签名。

    YOLO 模型的 forward(self, x, *args, **kwargs) 与 torch.jit.trace/script 不兼容。
    此包装器提供一个简洁的接口。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """仅接受输入张量的简单前向传播。"""
        # 调用模型的 forward 或 predict 方法
        # 对于 YOLO，我们需要推理输出（而不是训练输出）
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            # 直接调用 - 模型应处理推理模式
            output = self.model(x)
            # 处理不同的输出格式
            if isinstance(output, (list, tuple)):
                return output[0] if len(output) == 1 else output
            return output


def export_qat_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    dynamic_axes: bool = False,
) -> Path:
    """
    将 QAT 模型导出为带有 QDQ 算子的 ONNX 格式。

    此函数在转换为 INT8 之前导出 QAT 模型（带有 FakeQuantize 节点）。
    FakeQuantize 节点将被转换为 ONNX 的 QuantizeLinear/DequantizeLinear (QDQ) 算子。

    Args:
        model: QAT PyTorch 模型（带有 FakeQuantize，转换之前的状态）
        output_path: 保存 ONNX 模型的路径
        config: 导出配置
            - backend (str): 目标后端（'qnnpack'、'fbgemm'、'tensorrt'）
            - opset_version (int): ONNX opset 版本（默认：13）
            - simplify (bool): 运行 onnxsim 简化（默认：True）
        input_shape: 输入张量形状 (N, C, H, W)
        dynamic_axes: 启用动态 batch size

    Returns:
        导出的 ONNX 文件路径

    Note:
        对于 TensorRT/ONNX Runtime 部署，导出的模型将包含
        用于 INT8 推理的 QDQ（Quantize-DeQuantize）节点。
    """
    import copy

    config = config or {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 复制模型以避免修改原始模型
    model = copy.deepcopy(model)
    model.eval()
    model.cpu()

    # 包装模型以实现干净的 ONNX 导出（处理 YOLO 的 *args, **kwargs）
    export_model = ONNXExportWrapper(model)
    export_model.eval()

    # 创建示例输入
    dummy_input = torch.randn(*input_shape)

    # 配置导出选项
    backend = config.get('backend', 'qnnpack')
    opset_version = config.get('opset_version', 13)  # QDQ 需要 opset >= 13
    do_simplify = config.get('simplify', True)

    print(f"[QAT] 正在为 {backend} 后端导出 QAT 模型到 ONNX...")
    print(f"[QAT] FakeQuantize 节点将被转换为 QDQ 算子")

    # 输入/输出名称
    input_names = ['images']
    output_names = ['output0']

    # 动态轴配置
    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            'images': {0: 'batch_size'},
            'output0': {0: 'batch_size'},
        }

    # 使用追踪方式导出（对于带包装器的 QAT 模型效果更好）
    try:
        # 第一次尝试：追踪包装后的模型并导出
        print("[QAT] 正在追踪包装后的模型用于 ONNX 导出...")
        with torch.no_grad():
            # 运行一次以确保模型处于推理模式
            _ = export_model(dummy_input)
            traced_model = torch.jit.trace(export_model, dummy_input)

        torch.onnx.export(
            traced_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_config,
        )
        print(f"[QAT] ONNX 导出成功（追踪方式）：{output_path}")

    except Exception as trace_error:
        print(f"[QAT] 追踪导出失败：{trace_error}")
        print("[QAT] 尝试使用包装器直接导出...")

        try:
            # 第二次尝试：不使用追踪直接导出包装器
            torch.onnx.export(
                export_model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_config,
            )
            print(f"[QAT] ONNX 导出成功（直接方式）：{output_path}")

        except Exception as direct_error:
            print(f"[QAT] 直接导出失败：{direct_error}")
            print("[QAT] 尝试使用内部模型导出...")

            # 第三次尝试：尝试获取内部模型（backbone/head）
            try:
                inner_model = model
                if hasattr(model, 'model'):
                    inner_model = model.model

                # 包装内部模型
                inner_wrapper = ONNXExportWrapper(inner_model)
                inner_wrapper.eval()

                with torch.no_grad():
                    traced_inner = torch.jit.trace(inner_wrapper, dummy_input)

                torch.onnx.export(
                    traced_inner,
                    dummy_input,
                    str(output_path),
                    export_params=True,
                    opset_version=opset_version,
                    input_names=input_names,
                    output_names=output_names,
                )
                print(f"[QAT] ONNX 导出成功（内部模型）：{output_path}")
            except Exception as inner_error:
                print(f"[QAT] 所有导出方法均失败：{inner_error}")
                raise inner_error

    # 验证导出的模型
    verify_onnx_model(output_path)

    # 可选地简化模型
    if do_simplify:
        try:
            simplified_path = simplify_onnx(output_path)
            if simplified_path:
                output_path = simplified_path
        except Exception as e:
            print(f"[QAT] 警告：ONNX 简化失败：{e}")

    return output_path


def verify_onnx_model(onnx_path: Union[str, Path]) -> bool:
    """
    验证 ONNX 模型结构和有效性。

    Args:
        onnx_path: ONNX 模型路径

    Returns:
        如果模型有效则返回 True
    """
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"[QAT] ONNX 模型验证通过")

        # 打印模型信息
        print(f"[QAT] ONNX 模型信息：")
        print(f"      - IR 版本：{model.ir_version}")
        print(f"      - Opset 版本：{model.opset_import[0].version}")
        print(f"      - 输入：{[i.name for i in model.graph.input]}")
        print(f"      - 输出：{[o.name for o in model.graph.output]}")

        return True

    except ImportError:
        print(f"[QAT] 警告：未安装 onnx 包，跳过验证")
        return True
    except Exception as e:
        print(f"[QAT] 警告：ONNX 验证失败：{e}")
        return False


def simplify_onnx(onnx_path: Union[str, Path]) -> Optional[Path]:
    """
    使用 onnxsim 简化 ONNX 模型。

    Args:
        onnx_path: ONNX 模型路径

    Returns:
        简化后模型的路径，如果简化失败则返回 None
    """
    try:
        import onnxsim
        import onnx

        onnx_path = Path(onnx_path)
        simplified_path = onnx_path.with_suffix('.sim.onnx')

        print(f"[QAT] 正在简化 ONNX 模型...")
        model = onnx.load(str(onnx_path))
        model_simp, check = onnxsim.simplify(model)

        if check:
            onnx.save(model_simp, str(simplified_path))
            print(f"[QAT] 简化后的 ONNX 已保存到：{simplified_path}")

            # 比较大小
            original_size = onnx_path.stat().st_size / (1024 * 1024)
            simplified_size = simplified_path.stat().st_size / (1024 * 1024)
            print(f"[QAT] 大小：{original_size:.2f} MB -> {simplified_size:.2f} MB")

            return simplified_path
        else:
            print(f"[QAT] 警告：ONNX 简化检查失败")
            return None

    except ImportError:
        print(f"[QAT] 警告：未安装 onnxsim，跳过简化")
        print(f"[QAT] 安装命令：pip install onnxsim")
        return None
    except Exception as e:
        print(f"[QAT] 警告：ONNX 简化失败：{e}")
        return None


def export_for_tensorrt(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
) -> Path:
    """
    导出专门为 TensorRT 部署优化的模型。

    TensorRT 可以直接解析 ONNX 模型，并使用显式 QDQ 节点或
    隐式校准来应用 INT8 量化。

    Args:
        model: PyTorch 模型（可以是 QAT 或 FP32）
        output_path: 输出 ONNX 路径
        input_shape: 输入张量形状

    Returns:
        导出的 ONNX 模型路径
    """
    config = {
        'backend': 'tensorrt',
        'opset_version': 13,
        'simplify': True
    }

    return export_qat_to_onnx(
        model,
        output_path,
        config=config,
        input_shape=input_shape
    )


def export_for_ncnn(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)
) -> Path:
    """
    导出用于 NCNN 移动端部署的模型。

    NCNN 需要特定的 ONNX 结构以实现最佳转换。

    Args:
        model: 量化后的 PyTorch 模型
        output_path: 输出 ONNX 路径
        input_shape: 输入张量形状

    Returns:
        导出的 ONNX 模型路径
    """
    config = {
        'backend': 'qnnpack',
        'opset_version': 11,  # NCNN 在 opset 11 下效果最佳
        'simplify': True
    }

    return export_qat_to_onnx(
        model,
        output_path,
        config=config,
        input_shape=input_shape
    )


def get_onnx_model_stats(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取 ONNX 模型的统计信息。

    Returns:
        包含模型统计信息的字典
    """
    try:
        import onnx
        model = onnx.load(str(onnx_path))

        # 统计算子
        op_counts = {}
        for node in model.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

        # 检查量化算子
        quant_ops = ['QuantizeLinear', 'DequantizeLinear', 'QLinearConv', 'QLinearMatMul']
        has_quantization = any(op in op_counts for op in quant_ops)

        return {
            'file_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024),
            'ir_version': model.ir_version,
            'opset_version': model.opset_import[0].version,
            'num_nodes': len(model.graph.node),
            'num_inputs': len(model.graph.input),
            'num_outputs': len(model.graph.output),
            'has_quantization': has_quantization,
            'op_counts': op_counts,
        }
    except ImportError:
        return {'error': '未安装 onnx 包'}
    except Exception as e:
        return {'error': str(e)}
