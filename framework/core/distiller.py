import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import select_device


def align_scale(stu_feat, tea_feat):
    """
    特征对齐策略 (来自 CrossKD)
    将学生特征的分布对齐到教师特征的分布

    Args:
        stu_feat (torch.Tensor): 学生特征，shape [N, C, H, W]
        tea_feat (torch.Tensor): 教师特征，shape [N, C, H, W]

    Returns:
        torch.Tensor: 对齐后的学生特征
    """
    N, C = stu_feat.shape[:2]

    # 展平空间维度 [N, C, H, W] -> [N, C, H*W]
    stu_flat = stu_feat.view(N, C, -1)
    tea_flat = tea_feat.view(N, C, -1)

    # 计算每个通道的均值和标准差
    stu_mean = stu_flat.mean(dim=2, keepdim=True)
    stu_std = stu_flat.std(dim=2, keepdim=True)
    tea_mean = tea_flat.mean(dim=2, keepdim=True)
    tea_std = tea_flat.std(dim=2, keepdim=True)

    # 使用 clamp 确保最小标准差，防止除零导致 inf/NaN
    # FP16 友好的 epsilon 值
    eps = 1e-4 if stu_feat.dtype == torch.float16 else 1e-5
    stu_std = stu_std.clamp(min=eps)
    tea_std = tea_std.clamp(min=eps)  # 同时 clamp 教师的标准差

    # 归一化学生特征到零均值单位方差
    normalized = (stu_flat - stu_mean) / stu_std

    # clamp 防止极端值传播，避免 AMP (fp16) 下溢出
    # 使用自适应 clamp 范围
    max_clip = 5.0 if stu_feat.dtype == torch.float16 else 10.0
    normalized = normalized.clamp(-max_clip, max_clip)

    # 对齐到教师特征的分布
    aligned = normalized * tea_std + tea_mean

    return aligned.view_as(stu_feat)


def compute_cwd_loss(stu_feat, tea_feat, temperature=4.0):
    """
    Channel-Wise Distillation Loss (CWD)
    参考论文: "Channel-wise Distillation for Semantic Segmentation" (ICCV 2021)

    核心思想: 在每个空间位置上对通道维度进行 softmax 归一化后计算 KL 散度
    相比 MSE 更稳定，不会因为特征尺度差异导致训练崩溃

    Args:
        stu_feat (torch.Tensor): 学生特征, shape [N, C, H, W]
        tea_feat (torch.Tensor): 教师特征, shape [N, C, H, W]
        temperature (float): 温度系数，软化分布，默认 4.0

    Returns:
        torch.Tensor: CWD 损失值
    """
    N, C, H, W = stu_feat.shape

    # L2 归一化，消除特征幅度差异
    stu_norm = F.normalize(stu_feat, p=2, dim=1)
    tea_norm = F.normalize(tea_feat, p=2, dim=1)

    # [N, C, H, W] -> [N, H*W, C] 转置后在通道维度做 softmax
    stu_flat = stu_norm.view(N, C, -1).permute(0, 2, 1)  # [N, H*W, C]
    tea_flat = tea_norm.view(N, C, -1).permute(0, 2, 1)  # [N, H*W, C]

    # 通道维度 softmax + KL 散度
    stu_soft = F.log_softmax(stu_flat / temperature, dim=-1)
    tea_soft = F.softmax(tea_flat / temperature, dim=-1)

    loss = F.kl_div(stu_soft, tea_soft, reduction='batchmean') * (temperature ** 2)

    return loss


def compute_fgd_loss(stu_feat, tea_feat,
                     alpha_fgd=0.001, beta_fgd=0.0005,
                     gamma_fgd=0.0005, lambda_fgd=0.000005):
    """
    Focal and Global Knowledge Distillation (FGD)
    参考论文: "Focal and Global Knowledge Distillation for Detectors" (CVPR 2022)

    包含四部分损失:
    - Focal Loss: 前景区域特征蒸馏（使用教师注意力作为权重）
    - Background Loss: 背景区域特征蒸馏
    - Mask Loss: 空间注意力蒸馏
    - Relation Loss: 通道间关系蒸馏（Gram 矩阵）

    Args:
        stu_feat (torch.Tensor): 学生特征, shape [N, C, H, W]
        tea_feat (torch.Tensor): 教师特征, shape [N, C, H, W]
        alpha_fgd: 前景损失权重
        beta_fgd: 背景损失权重
        gamma_fgd: 注意力损失权重
        lambda_fgd: 关系损失权重

    Returns:
        torch.Tensor: FGD 总损失
    """
    N, C, H, W = stu_feat.shape

    def _normalize_spatial_attention(feat):
        """计算并归一化空间注意力掩码"""
        spatial = feat.abs().mean(dim=1, keepdim=True)  # [N, 1, H, W]
        spatial_flat = spatial.view(N, -1)
        min_val = spatial_flat.min(dim=1, keepdim=True)[0].view(N, 1, 1, 1)
        max_val = spatial_flat.max(dim=1, keepdim=True)[0].view(N, 1, 1, 1)
        # 使用 clamp 确保分母不为零，1e-6 对 FP16 友好
        denom = torch.clamp(max_val - min_val, min=1e-6)
        return (spatial - min_val) / denom

    # 计算教师和学生的空间注意力掩码
    spatial_mask = _normalize_spatial_attention(tea_feat)
    stu_mask = _normalize_spatial_attention(stu_feat)

    # 2. Focal Loss: 前景区域加权特征差异
    # 使用 smooth L1 损失代替 MSE，更稳定
    feat_diff = F.smooth_l1_loss(stu_feat, tea_feat, reduction='none')
    fg_loss = (spatial_mask * feat_diff).mean()

    # 3. Background Loss: 背景区域特征差异（1 - mask 作为权重）
    bg_loss = ((1 - spatial_mask) * feat_diff).mean()

    # 4. Mask Loss: 空间注意力对齐
    mask_loss = F.smooth_l1_loss(stu_mask, spatial_mask)

    # 5. Relation Loss: 通道间关系矩阵对齐 (Gram 矩阵)
    # 展平空间维度
    stu_flat = stu_feat.view(N, C, -1)  # [N, C, H*W]
    tea_flat = tea_feat.view(N, C, -1)

    # 归一化（避免数值问题）
    stu_flat = F.normalize(stu_flat, p=2, dim=2)
    tea_flat = F.normalize(tea_flat, p=2, dim=2)

    # Gram 矩阵 (通道相关性) [N, C, C]
    stu_gram = torch.bmm(stu_flat, stu_flat.permute(0, 2, 1))
    tea_gram = torch.bmm(tea_flat, tea_flat.permute(0, 2, 1))

    relation_loss = F.smooth_l1_loss(stu_gram, tea_gram)

    # 组合所有损失
    total_loss = (alpha_fgd * fg_loss +
                  beta_fgd * bg_loss +
                  gamma_fgd * mask_loss +
                  lambda_fgd * relation_loss)

    return total_loss


class ChannelAdapter(nn.Module):
    """
    通道适配器
    当学生和教师的特征通道数不匹配时，使用 1x1 卷积进行对齐
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # 使用较小的初始化值，避免初始阶段损失过大
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.conv(x)


def compute_distill_loss(model, batch, preds=None):
    """
    自定义的蒸馏损失计算逻辑（全局函数，支持 Pickle 序列化）

    支持的损失类型:
      - semantic: 语义分离蒸馏（检测头输出，KL散度+对齐）
      - cwd: Channel-Wise Distillation（中间层特征，通道级KL散度）
      - fgd: Focal and Global Distillation（中间层特征，局部+全局蒸馏）
    """
    ctx = model.distiller

    # 0. 如果不需要计算梯度（验证/推理阶段），直接返回 GT Loss，跳过蒸馏
    if not torch.is_grad_enabled():
        if not hasattr(model, 'criterion'):
            model.criterion = model.init_criterion()
        return model.criterion(preds, batch)

    # 1. 确保 Teacher 在正确的设备上
    student_device = next(model.parameters()).device
    teacher_device = next(ctx.teacher_model.parameters()).device

    if teacher_device != student_device:
        ctx.teacher_model.to(student_device)
        # 同步适配器设备
        for adapter in ctx.adapters.values():
            adapter.to(student_device)

    # 2. 获取 Student 的预测 (如果传入为 None)
    if preds is None:
        preds = model(batch["img"])

    # 3. 计算原始真值损失
    if not hasattr(model, 'criterion'):
        model.criterion = model.init_criterion()

    gt_loss, loss_items = model.criterion(preds, batch)

    # 4. 获取教师模型的预测
    teacher_head = ctx.teacher_model.model[-1]
    head_training = teacher_head.training
    teacher_head.training = True

    with torch.no_grad():
        teacher_dtype = next(ctx.teacher_model.parameters()).dtype
        teacher_input = batch["img"].to(dtype=teacher_dtype)
        teacher_preds = ctx.teacher_model(teacher_input)

    teacher_head.training = head_training

    # 5. 计算蒸馏损失
    distill_loss = torch.tensor(0.0, device=student_device)
    alpha = ctx.distill_cfg.get('alpha', 0.5)
    loss_type = ctx.distill_cfg.get('loss_type', 'cwd')  # 默认使用 CWD
    T = ctx.distill_cfg.get('T', 2.0)  # 温度系数

    # ========== 检测头输出蒸馏 (semantic) ==========
    if loss_type == 'semantic':
        if isinstance(preds, (list, tuple)):
            for stu_p, tea_p in zip(preds, teacher_preds):
                # 防御性检查
                if isinstance(tea_p, (list, tuple)):
                    continue
                if stu_p.shape != tea_p.shape:
                    continue

                # 从配置获取 reg_max，默认为 16（YOLO v8/v11 标准值）
                reg_max = ctx.distill_cfg.get('reg_max', 16)
                num_reg_channels = 4 * reg_max

                # 分离回归和分类分支
                stu_reg = stu_p[:, :num_reg_channels, :, :]
                stu_cls = stu_p[:, num_reg_channels:, :, :]
                tea_reg = tea_p[:, :num_reg_channels, :, :]
                tea_cls = tea_p[:, num_reg_channels:, :, :]

                # 回归分支：align_scale + smooth_l1（比 MSE 更稳定）
                stu_reg_aligned = align_scale(stu_reg, tea_reg)
                reg_loss = F.smooth_l1_loss(stu_reg_aligned, tea_reg)

                # 分类分支：KL 散度 + 温度软化
                _, C_cls, _, _ = stu_cls.shape
                stu_cls_flat = stu_cls.permute(0, 2, 3, 1).reshape(-1, C_cls)
                tea_cls_flat = tea_cls.permute(0, 2, 3, 1).reshape(-1, C_cls)

                cls_loss = F.kl_div(
                    F.log_softmax(stu_cls_flat / T, dim=1),
                    F.softmax(tea_cls_flat / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)

                distill_loss = distill_loss + reg_loss + cls_loss

    # ========== 中间层特征蒸馏 (cwd / fgd) ==========
    elif loss_type in ('cwd', 'fgd'):
        # 从 hooks 获取捕获的中间层特征
        feature_layers = ctx.distill_cfg.get('feature_layers', [])
        cwd_temperature = ctx.distill_cfg.get('cwd_temperature', 4.0)

        for layer_idx in feature_layers:
            stu_feat = ctx.student_features.get(layer_idx)
            tea_feat = ctx.teacher_features.get(layer_idx)

            if stu_feat is None or tea_feat is None:
                continue

            # 通道数适配
            if stu_feat.shape[1] != tea_feat.shape[1]:
                adapter = ctx.adapters.get(layer_idx)
                if adapter is not None:
                    stu_feat = adapter(stu_feat)
                else:
                    continue

            # 空间尺寸对齐
            if stu_feat.shape[2:] != tea_feat.shape[2:]:
                stu_feat = F.interpolate(
                    stu_feat, size=tea_feat.shape[2:],
                    mode='bilinear', align_corners=False
                )

            # 计算对应的损失
            if loss_type == 'cwd':
                layer_loss = compute_cwd_loss(stu_feat, tea_feat, cwd_temperature)
            else:  # fgd
                layer_loss = compute_fgd_loss(stu_feat, tea_feat)

            distill_loss = distill_loss + layer_loss

        # 清空特征缓存
        ctx.student_features.clear()
        ctx.teacher_features.clear()

    else:
        raise ValueError(f"不支持的损失类型: {loss_type}，请使用 'semantic', 'cwd' 或 'fgd'")

    # 6. 检查蒸馏损失有效性
    if isinstance(distill_loss, torch.Tensor):
        if torch.isnan(distill_loss) or torch.isinf(distill_loss):
            print(f"[Distiller Warning] distill_loss 异常 ({distill_loss.item():.4f})，跳过本次蒸馏")
            return gt_loss, loss_items

    total_loss = gt_loss + alpha * distill_loss

    return total_loss, loss_items


class DistillationContext:
    """
    蒸馏上下文，封装蒸馏所需的所有状态
    """
    def __init__(self, teacher_model, distill_cfg):
        self.teacher_model = teacher_model
        self.distill_cfg = distill_cfg

        # 中间层特征存储
        self.student_features = {}
        self.teacher_features = {}

        # Hook 句柄，用于清理
        self._hooks = []

        # 通道适配器
        self.adapters = {}

    def _make_hook(self, storage_dict, layer_idx):
        """创建用于捕获特征的 forward hook"""
        def hook(module, input, output):
            # 只在训练模式下捕获特征
            if torch.is_grad_enabled():
                storage_dict[layer_idx] = output
        return hook

    def register_hooks(self, student_model):
        """为学生和教师模型注册特征提取 hooks"""
        feature_layers = self.distill_cfg.get('feature_layers', [])
        if not feature_layers:
            return

        # 清除旧 hooks
        self.clear_hooks()

        for layer_idx in feature_layers:
            # 学生模型 hook
            if hasattr(student_model, 'model') and layer_idx < len(student_model.model):
                h = student_model.model[layer_idx].register_forward_hook(
                    self._make_hook(self.student_features, layer_idx)
                )
                self._hooks.append(h)

            # 教师模型 hook
            if hasattr(self.teacher_model, 'model') and layer_idx < len(self.teacher_model.model):
                h = self.teacher_model.model[layer_idx].register_forward_hook(
                    self._make_hook(self.teacher_features, layer_idx)
                )
                self._hooks.append(h)

    def clear_hooks(self):
        """移除所有 hooks"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.student_features.clear()
        self.teacher_features.clear()


class DistillationTrainer(DetectionTrainer):
    def __init__(self, distill_cfg, overrides=None, _callbacks=None):
        """
        初始化蒸馏训练器
        :param distill_cfg: 蒸馏相关的配置字典
        """
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.distill_cfg = distill_cfg
        self.teacher_model = None

        # 提前加载 Teacher 模型
        self._load_teacher()

    def _load_teacher(self):
        teacher_path = self.distill_cfg.get('teacher')
        if not teacher_path:
            raise ValueError("蒸馏模式已启用但未指定教师模型路径！")

        print(f"[Distiller] 从 {teacher_path} 中载入教师模型...")
        yolo_model = YOLO(teacher_path)
        self.teacher_model = yolo_model.model

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_model.eval()

        if self.args.device:
            device = select_device(self.args.device, verbose=False)
            self.teacher_model.to(device)

        print(f"[Distiller] 教师模型加载成功")

    def _get_layer_out_channels(self, model, layer_idx):
        """获取指定层的输出通道数"""
        try:
            layer = model.model[layer_idx]
            # 尝试常见的通道数属性
            if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
                return layer.cv2.conv.out_channels
            elif hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
                return layer.conv.out_channels
            elif hasattr(layer, 'out_channels'):
                return layer.out_channels
            # C2f 模块
            if hasattr(layer, 'c'):
                return layer.c
            return 0
        except Exception:
            return 0

    def _build_adapters(self, student_model, ctx):
        """构建通道适配器（当学生和教师通道数不匹配时）"""
        feature_layers = self.distill_cfg.get('feature_layers', [])
        device = next(student_model.parameters()).device

        for layer_idx in feature_layers:
            stu_ch = self._get_layer_out_channels(student_model, layer_idx)
            tea_ch = self._get_layer_out_channels(self.teacher_model, layer_idx)

            if stu_ch > 0 and tea_ch > 0 and stu_ch != tea_ch:
                print(f"[Distiller] 层 {layer_idx}: 创建通道适配器 {stu_ch} -> {tea_ch}")
                adapter = ChannelAdapter(stu_ch, tea_ch)
                adapter.to(device)
                ctx.adapters[layer_idx] = adapter

    def save_model(self):
        """
        重写 save_model，在保存前移除 Monkey Patch 和 Hooks，确保模型干净且可被标准库加载
        """
        # 1. 临时移除 Monkey Patch
        custom_loss_method = self.model.loss
        if hasattr(self.model, 'loss'):
            del self.model.loss

        # 移除 distiller 上下文引用
        distiller_ctx = getattr(self.model, "distiller", None)
        if hasattr(self.model, 'distiller'):
            del self.model.distiller

        # 2. 清除所有 hooks（解决 pickle 无法序列化局部函数的问题）
        if distiller_ctx is not None:
            distiller_ctx.clear_hooks()

        # 处理 EMA 模型
        ema_loss_method = None
        ema_distiller_ctx = None
        if hasattr(self, 'ema') and hasattr(self.ema, 'ema'):
            if hasattr(self.ema.ema, 'loss'):
                ema_loss_method = self.ema.ema.loss
                del self.ema.ema.loss
            if hasattr(self.ema.ema, 'distiller'):
                ema_distiller_ctx = self.ema.ema.distiller
                del self.ema.ema.distiller
                # 同样清除 EMA 模型的 hooks
                if ema_distiller_ctx is not None:
                    ema_distiller_ctx.clear_hooks()

        # 3. 调用父类保存逻辑
        super().save_model()

        # 4. 恢复 Monkey Patch 和上下文
        self.model.loss = custom_loss_method
        if distiller_ctx:
            self.model.distiller = distiller_ctx
            # 重新注册 hooks
            loss_type = self.distill_cfg.get('loss_type', 'cwd')
            if loss_type in ('cwd', 'fgd'):
                distiller_ctx.register_hooks(self.model)

        if hasattr(self, 'ema') and hasattr(self.ema, 'ema'):
            if ema_loss_method is not None:
                self.ema.ema.loss = ema_loss_method
            if ema_distiller_ctx is not None:
                self.ema.ema.distiller = ema_distiller_ctx
                # 重新注册 EMA 模型的 hooks
                loss_type = self.distill_cfg.get('loss_type', 'cwd')
                if loss_type in ('cwd', 'fgd'):
                    ema_distiller_ctx.register_hooks(self.ema.ema)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        重写 get_model，拦截模型创建过程，注入自定义损失函数
        """
        # 1. 获取正常的学生模型
        model = super().get_model(cfg, weights, verbose)

        # 2. 创建蒸馏上下文
        ctx = DistillationContext(self.teacher_model, self.distill_cfg)
        model.distiller = ctx

        # 3. 如果启用中间层蒸馏，注册 hooks 和适配器
        loss_type = self.distill_cfg.get('loss_type', 'cwd')
        if loss_type in ('cwd', 'fgd'):
            ctx.register_hooks(model)
            self._build_adapters(model, ctx)

            # 将适配器参数加入优化
            feature_layers = self.distill_cfg.get('feature_layers', [])
            if feature_layers:
                print(f"[Distiller] 中间层蒸馏已启用，目标层: {feature_layers}")

        # 4. 劫持损失函数
        model.loss = types.MethodType(compute_distill_loss, model)

        return model
