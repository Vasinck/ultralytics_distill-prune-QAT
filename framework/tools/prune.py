import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv, Bottleneck
import torch_pruning as tp
import argparse
import os

# ==========================================
# C2f_v2 实现（用于剪枝支持）
# ==========================================

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

class C2f_v2(nn.Module):
    # 带有 2 个卷积的 CSP Bottleneck
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # 输入通道, 输出通道, 数量, shortcut, 分组, 扩展系数
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 可选的激活函数 act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        # 替换：将 cv1 拆分为 cv0 和 cv1 以支持追踪
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # 将 cv1 权重从 C2f 转移到 C2f_v2 的 cv0 和 cv1
    # C2f.cv1 是一个大卷积，会被 chunk。我们拆分它的权重。
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # 转移 cv1 的 batchnorm 权重和缓冲区
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # 转移剩余的权重和缓冲区
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # 转移所有非方法属性（如用于 YOLO 拓扑的 .i, .f）
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # 将 C2f 替换为 C2f_v2 并保留其参数
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
            print(f"[Pruning] 已将 {name} 替换为 C2f_v2")
        else:
            replace_c2f_with_c2f_v2(child_module)

# ==========================================
# 主剪枝逻辑
# ==========================================

def prune_model(model_path, save_path, pruning_ratio=0.2, example_input_shape=(1, 3, 640, 640)):
    print(f"[Pruning] 正在从 {model_path} 加载模型...")
    yolo = YOLO(model_path)
    model = yolo.model

    model.cpu()
    model.eval()

    # 1. 将 C2f 替换为 C2f_v2
    print("[Pruning] 正在将 C2f 模块替换为支持追踪的 C2f_v2...")
    replace_c2f_with_c2f_v2(model)

    # 2. 构建依赖图
    example_inputs = torch.randn(example_input_shape)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear,)):
            ignored_layers.append(m)
        if m.__class__.__name__ == 'Detect':
            ignored_layers.append(m)

    print(f"[Pruning] 忽略的层：{[type(m).__name__ for m in ignored_layers]}")

    # 调试：检查依赖图（标准方式，不需要手动包装 forward）
    try:
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        print(f"[Debug] 图节点数：{len(DG.module2node)}")

        # 检查全局分组
        all_groups = list(DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=[torch.nn.Conv2d]))
        print(f"[Debug] 找到的可剪枝分组总数：{len(all_groups)}")

    except Exception as e:
        print(f"[Debug] 依赖图构建失败：{e}")
        import traceback
        traceback.print_exc()

    # 3. 初始化剪枝器
    # 根据官方 YOLO 示例推荐使用 GroupNormPruner
    imp = tp.importance.GroupMagnitudeImportance(p=2)

    pruner = tp.pruner.GroupNormPruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    print(f"[Pruning] 剪枝器已初始化。目标稀疏度：{pruning_ratio}")

    # 4. 执行剪枝
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"[Pruning] 剪枝前：MACs={base_macs/1e9:.3f} G, Params={base_nparams/1e6:.3f} M")

    if isinstance(pruner, tp.pruner.BasePruner):
        pruner.step()

    # 5. 验证
    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"[Pruning] 剪枝后：MACs={pruned_macs/1e9:.3f} G, Params={pruned_nparams/1e6:.3f} M")

    if base_nparams > 0:
        print(f"[Pruning] 参数减少：{100 * (base_nparams - pruned_nparams) / base_nparams:.2f}%")
        print(f"[Pruning] MACs 减少：{100 * (base_macs - pruned_macs) / base_macs:.2f}%")

    if base_nparams == pruned_nparams:
        print("\n[Warning] 没有参数被剪枝。")

    # 6. 保存模型
    # 注意：模型现在包含 C2f_v2 类。
    # 如果在标准 ultralytics 中加载，如果未定义 C2f_v2 可能会崩溃。
    # 但是，由于我们正在构建自定义框架，我们可以在框架中包含 C2f_v2 的定义。
    # 或者，我们可以尝试转换回原来的结构？（困难）
    # 最佳实践：按原样保存结构，并确保推理/训练脚本导入 C2f_v2。

    ckpt = {
        'model': model,
        'epoch': -1,
        'best_fitness': None,
        'train_args': {},  # 必须是字典而不是 None，以确保与 ultralytics 兼容
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"[Pruning] 剪枝后的模型已保存到 {save_path}")

    # 7. ONNX 导出
    onnx_path = save_path.replace('.pt', '.onnx')
    try:
        torch.onnx.export(model, example_inputs, onnx_path)
        print(f"[Pruning] ONNX 导出成功：{onnx_path}")
    except Exception as e:
        print(f"[Pruning] 警告：ONNX 导出失败：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自研通用目标检测框架 - 剪枝工具")
    parser.add_argument('--model', type=str, required=True, help='Path to source .pt model')
    parser.add_argument('--save', type=str, required=True, help='Path to save pruned model')
    parser.add_argument('--ratio', type=float, default=0.2, help='Pruning ratio (0.0-1.0)')
    
    args = parser.parse_args()
    
    prune_model(args.model, args.save, args.ratio)