"""
YOLO 模型模块融合（用于量化）
自研通用目标检测框架 v1.2 的一部分

此模块提供用于 QAT 的 Conv+BN+Act 模块融合工具，
处理 YOLO 特有的结构（如 SiLU 激活函数）。
"""

import copy
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def replace_silu_with_relu(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    将 SiLU 激活函数替换为 ReLU6 以提高量化兼容性。

    SiLU (Swish) 由于其非单调特性，对量化不友好。
    ReLU6 提供有界输出范围 [0, 6]，更适合 INT8 量化。

    Args:
        model: 要修改的 PyTorch 模型
        inplace: 是否原地修改模型

    Returns:
        SiLU 已被 ReLU6 替换的修改后模型

    Note:
        这可能导致 1-2% 的精度下降，但显著提高量化兼容性。
    """
    if not inplace:
        model = copy.deepcopy(model)

    replaced_count = 0

    for name, module in model.named_modules():
        # 检查 SiLU 激活函数
        if isinstance(module, nn.SiLU):
            # 获取父模块和属性名
            parent = _get_parent_module(model, name)
            attr_name = name.split('.')[-1]

            if parent is not None:
                setattr(parent, attr_name, nn.ReLU6(inplace=True))
                replaced_count += 1

        # 同时检查带有 SiLU act 的 Conv 模块
        if hasattr(module, 'act') and isinstance(module.act, nn.SiLU):
            module.act = nn.ReLU6(inplace=True)
            replaced_count += 1

    if replaced_count > 0:
        print(f"[QAT] 已将 {replaced_count} 个 SiLU 激活函数替换为 ReLU6")

    return model


def _get_parent_module(model: nn.Module, name: str) -> Optional[nn.Module]:
    """通过名称获取父模块，使用直接属性访问（O(depth) 而不是 O(n)）。"""
    parts = name.split('.')
    if len(parts) == 1:
        return model

    parent_name = '.'.join(parts[:-1])
    try:
        parent = model
        for part in parent_name.split('.'):
            parent = getattr(parent, part)
        return parent
    except AttributeError:
        return None


def fuse_yolo_modules(
    model: nn.Module,
    replace_silu: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    融合 YOLO 模型中的 Conv+BN+Act 模块以准备 QAT。

    此函数执行模块融合，这是 QAT 之前必需的步骤：
    1. 可选地将 SiLU 替换为 ReLU6 以获得更好的量化效果
    2. 识别可融合的模块组（Conv+BN+ReLU 模式）
    3. 应用 torch.ao.quantization.fuse_modules

    Args:
        model: 要准备量化的 YOLO 模型
        replace_silu: 是否将 SiLU 替换为 ReLU6（推荐用于 INT8）
        verbose: 打印融合统计信息

    Returns:
        已融合、准备好进行 QAT 准备的模型

    Note:
        YOLO 模型默认使用 SiLU (Swish) 激活函数，PyTorch 量化原生不支持。
        我们将其替换为 ReLU6。
    """
    model = model.cpu()
    model.eval()

    # 步骤 1：如果需要，将 SiLU 替换为 ReLU6
    if replace_silu:
        model = replace_silu_with_relu(model)

    # 步骤 2：查找可融合的模块组
    fuse_list = _find_fusable_modules(model, verbose=verbose)

    if not fuse_list:
        if verbose:
            print("[QAT] 未找到可融合的模块（对于某些模型结构这是正常的）")
        return model

    # 步骤 3：应用模块融合
    try:
        model = torch.ao.quantization.fuse_modules(
            model,
            fuse_list,
            inplace=True
        )
        if verbose:
            print(f"[QAT] 成功融合了 {len(fuse_list)} 个模块组")
    except Exception as e:
        print(f"[QAT] 警告：模块融合失败：{e}")
        print("[QAT] 继续执行但不进行融合（QAT 仍然可以工作，但效率可能较低）")

    return model


def _find_fusable_modules(model: nn.Module, verbose: bool = True) -> List[List[str]]:
    """
    查找模型中所有可融合的 Conv+BN+Act 模式。

    PyTorch 的 fuse_modules 需要精确的子模块路径和特定模式：
    - [conv, bn] 用于无激活函数的 Conv+BN
    - [conv, bn, relu] 用于 Conv+BN+ReLU
    - [conv, bn, relu6] 用于 Conv+BN+ReLU6

    Returns:
        可融合模块名称列表的列表
    """
    fuse_list = []

    for name, module in model.named_modules():
        # 模式 1：YOLO Conv 模块（有 conv、bn、act 作为子模块）
        if _is_yolo_conv_module(module):
            conv_name = f"{name}.conv"
            bn_name = f"{name}.bn"
            act_name = f"{name}.act"

            # 检查激活函数是否可融合
            if _is_fusable_activation(module.act):
                fuse_list.append([conv_name, bn_name, act_name])
            else:
                # 如果 act 不可融合，只融合 conv+bn
                fuse_list.append([conv_name, bn_name])

        # 模式 2：直接的 nn.Conv2d 后跟 nn.BatchNorm2d
        # 这处理自定义模块
        elif isinstance(module, nn.Conv2d):
            parent = _get_parent_module(model, name)
            if parent is not None:
                bn_candidates = ['bn', 'norm', 'bn1']
                for bn_attr in bn_candidates:
                    if hasattr(parent, bn_attr):
                        bn = getattr(parent, bn_attr)
                        if isinstance(bn, nn.BatchNorm2d):
                            # 检查是否有激活函数
                            act_candidates = ['act', 'relu', 'activation']
                            for act_attr in act_candidates:
                                if hasattr(parent, act_attr):
                                    act = getattr(parent, act_attr)
                                    if _is_fusable_activation(act):
                                        parent_name = '.'.join(name.split('.')[:-1])
                                        fuse_list.append([
                                            name,
                                            f"{parent_name}.{bn_attr}" if parent_name else bn_attr,
                                            f"{parent_name}.{act_attr}" if parent_name else act_attr
                                        ])
                                        break
                            break

    if verbose and fuse_list:
        print(f"[QAT] 找到 {len(fuse_list)} 个可融合的模块组")

    return fuse_list


def _is_yolo_conv_module(module: nn.Module) -> bool:
    """检查模块是否是具有 conv+bn+act 结构的 YOLO Conv 模块。"""
    return (
        hasattr(module, 'conv') and isinstance(module.conv, nn.Conv2d) and
        hasattr(module, 'bn') and isinstance(module.bn, nn.BatchNorm2d) and
        hasattr(module, 'act')
    )


def _is_fusable_activation(act: nn.Module) -> bool:
    """检查激活函数是否可被 PyTorch 量化融合。"""
    fusable_types = (nn.ReLU, nn.ReLU6)
    return isinstance(act, fusable_types)


def get_model_layers_info(model: nn.Module) -> List[dict]:
    """
    获取模型中所有层的信息。
    用于调试和理解模型结构。

    Returns:
        包含层名称、类型和量化兼容性的字典列表
    """
    layers = []

    for name, module in model.named_modules():
        layer_info = {
            'name': name,
            'type': type(module).__name__,
            'quantizable': False,
            'fusable': False,
            'has_bn': False,
            'activation': None
        }

        # 检查是否可量化
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_info['quantizable'] = True

        # 检查是否可融合
        if _is_yolo_conv_module(module):
            layer_info['fusable'] = True
            layer_info['has_bn'] = True
            layer_info['activation'] = type(module.act).__name__

        if layer_info['quantizable'] or layer_info['fusable']:
            layers.append(layer_info)

    return layers


def count_quantizable_params(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型中的总参数量和可量化参数量。

    Returns:
        元组 (总参数量, 可量化参数量)
    """
    # 从所有模型参数计算总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 仅从 Conv2d 和 Linear 层计算可量化参数量
    quantizable_params = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            quantizable_params += sum(p.numel() for p in module.parameters())

    return total_params, quantizable_params
