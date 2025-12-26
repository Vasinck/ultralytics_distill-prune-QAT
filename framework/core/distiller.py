import torch
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

    # [FIX] 使用 clamp 确保最小标准差，防止除零导致 inf/NaN
    eps = 1e-5
    stu_std = stu_std.clamp(min=eps)

    # 归一化学生特征到零均值单位方差
    normalized = (stu_flat - stu_mean) / stu_std

    # [FIX] clamp 防止极端值传播，避免 AMP (fp16) 下溢出
    normalized = normalized.clamp(-10, 10)

    # 对齐到教师特征的分布
    aligned = normalized * tea_std + tea_mean

    return aligned.view_as(stu_feat)


def compute_distill_loss(model, batch, preds=None):
    """
    自定义的蒸馏损失计算逻辑（全局函数，支持 Pickle 序列化）
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

    # 2. 获取 Student 的预测 (如果传入为 None)
    if preds is None:
        preds = model(batch["img"])

    # 3. 计算原始真值损失
    # 我们需要确保 student_model 已经有了 criterion
    if not hasattr(model, 'criterion'):
        model.criterion = model.init_criterion()
    
    gt_loss, loss_items = model.criterion(preds, batch)

    # 4. 获取教师模型的预测
    # 技巧：临时将教师模型检测头设为训练模式，以获取原始特征而不是解码后的框
    teacher_head = ctx.teacher_model.model[-1]
    head_training = teacher_head.training
    teacher_head.training = True

    with torch.no_grad():
        # 强制转换输入类型以匹配教师模型（解决 AMP/验证时的类型不匹配问题）
        teacher_dtype = next(ctx.teacher_model.parameters()).dtype
        teacher_input = batch["img"].to(dtype=teacher_dtype)
        teacher_preds = ctx.teacher_model(teacher_input)

    teacher_head.training = head_training

    # 5. 计算蒸馏损失（集成 CrossKD 技术）
    distill_loss = 0.0
    alpha = ctx.distill_cfg.get('alpha', 0.5)
    loss_type = ctx.distill_cfg.get('loss_type', 'pkd')  # 'pkd'、'mse' 或 'semantic'
    T = ctx.distill_cfg.get('T', 2.0)  # 温度系数，用于分类分支软化

    # 使用 CrossKD 改进的特征蒸馏
    if isinstance(preds, (list, tuple)):
        for _, (stu_p, tea_p) in enumerate(zip(preds, teacher_preds)):
            # 防御性检查：如果 tea_p 仍然是列表（复杂结构），则跳过
            if isinstance(tea_p, (list, tuple)):
                continue

            # 如果形状不匹配，跳过此层
            if stu_p.shape != tea_p.shape:
                continue

            # **语义分离蒸馏 (v1.2 新增)**
            if loss_type == 'semantic':
                # 检测头输出结构：[N, 4*reg_max + nc, H, W]
                # 前 64 通道（4*16）：DFL 回归分布
                # 后 nc 通道：分类 logits
                reg_max = 16
                num_reg_channels = 4 * reg_max  # 64

                # 分离回归和分类分支
                stu_reg = stu_p[:, :num_reg_channels, :, :]
                stu_cls = stu_p[:, num_reg_channels:, :, :]
                tea_reg = tea_p[:, :num_reg_channels, :, :]
                tea_cls = tea_p[:, num_reg_channels:, :, :]

                # 回归分支：align_scale + MSE
                stu_reg_aligned = align_scale(stu_reg, tea_reg)
                reg_loss = F.mse_loss(stu_reg_aligned, tea_reg)

                # 分类分支：KL 散度 + 温度软化
                # 对空间维度展平后在通道维度做 softmax
                _, C_cls, _, _ = stu_cls.shape
                stu_cls_flat = stu_cls.permute(0, 2, 3, 1).reshape(-1, C_cls)  # [N*H*W, nc]
                tea_cls_flat = tea_cls.permute(0, 2, 3, 1).reshape(-1, C_cls)  # [N*H*W, nc]

                cls_loss = F.kl_div(
                    F.log_softmax(stu_cls_flat / T, dim=1),
                    F.softmax(tea_cls_flat / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)

                distill_loss += reg_loss + cls_loss

            # **CrossKD 改进：特征对齐 + MSE**
            elif loss_type == 'pkd':
                stu_aligned = align_scale(stu_p, tea_p)
                distill_loss += F.mse_loss(stu_aligned, tea_p)

            # **简单 MSE 损失（向后兼容）**
            else:
                distill_loss += F.mse_loss(stu_p, tea_p)

    # [修复] 安全检查：如果蒸馏损失异常（NaN/Inf），跳过本次蒸馏
    if isinstance(distill_loss, torch.Tensor):
        if torch.isnan(distill_loss) or torch.isinf(distill_loss):
            print(f"[Distiller Warning] distill_loss is {distill_loss.item():.4f}, skipping distillation this step")
            return gt_loss, loss_items

    # 将蒸馏损失加权
    total_loss = gt_loss + alpha * distill_loss

    return total_loss, loss_items

class DistillationContext:
    def __init__(self, teacher_model, distill_cfg):
        self.teacher_model = teacher_model
        self.distill_cfg = distill_cfg

class DistillationTrainer(DetectionTrainer):
    def __init__(self, distill_cfg, overrides=None, _callbacks=None):
        """
        初始化蒸馏训练器
        :param distill_cfg: 蒸馏相关的配置字典（teacher, alpha, T）
        """
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.distill_cfg = distill_cfg
        self.teacher_model = None
        self.original_criterion = None  # 保存学生模型原始的损失函数

        # 提前加载 Teacher 模型
        self._load_teacher()

    def _load_teacher(self):
        teacher_path = self.distill_cfg.get('teacher')
        if not teacher_path:
            raise ValueError("Distillation enabled but no teacher model specified!")
        
        print(f"[Distiller] Loading teacher model from {teacher_path}...")
        # 使用 YOLO 类加载模型，这是更稳健的公开 API
        # YOLO(path) 会自动下载并加载权重
        yolo_model = YOLO(teacher_path)
        self.teacher_model = yolo_model.model

        # 冻结教师模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # 设置为评估模式
        self.teacher_model.eval()

        # 确保教师模型和当前设备一致（在 get_model 后会被再次检查，这里先预处理）
        if self.args.device:
            # 使用 select_device 正确解析设备字符串（如 '2' -> 'cuda:2'）
            device = select_device(self.args.device, verbose=False)
            self.teacher_model.to(device)

        print(f"[Distiller] 教师模型加载成功")

    def save_model(self):
        """
        重写 save_model，在保存前移除 Monkey Patch，确保模型干净且可被标准库加载
        同时处理 EMA 模型
        """
        # 1. 临时移除 Monkey Patch（删除实例属性，回退到类方法）
        custom_loss_method = self.model.loss
        if hasattr(self.model, 'loss'):
             del self.model.loss

        # 移除 distiller 上下文引用
        distiller_ctx = getattr(self.model, "distiller", None)
        if hasattr(self.model, 'distiller'):
            del self.model.distiller

        # 处理 EMA 模型（它也可能被 Monkey Patch 污染）
        ema_loss_method = None
        ema_distiller_ctx = None
        if hasattr(self, 'ema') and hasattr(self.ema, 'ema'):
            if hasattr(self.ema.ema, 'loss'):
                ema_loss_method = self.ema.ema.loss
                del self.ema.ema.loss
            if hasattr(self.ema.ema, 'distiller'):
                ema_distiller_ctx = self.ema.ema.distiller
                del self.ema.ema.distiller

        # 2. 调用父类保存逻辑
        super().save_model()

        # 3. 恢复 Monkey Patch 和上下文
        self.model.loss = custom_loss_method
        if distiller_ctx:
            self.model.distiller = distiller_ctx
            
        if hasattr(self, 'ema') and hasattr(self.ema, 'ema'):
            if ema_loss_method is not None:
                self.ema.ema.loss = ema_loss_method
            if ema_distiller_ctx is not None:
                self.ema.ema.distiller = ema_distiller_ctx

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        重写 get_model，拦截模型创建过程，注入自定义损失函数
        """
        # 1. 获取正常的学生模型
        model = super().get_model(cfg, weights, verbose)

        # 2. 将训练器挂载到模型上，以便在全局函数中访问
        # 使用 DistillationContext 避免引用整个训练器（会导致 pickle 失败）
        model.distiller = DistillationContext(self.teacher_model, self.distill_cfg)

        # 3. 劫持损失函数
        # 使用 types.MethodType 将全局函数绑定为实例方法
        # 这样 Pickle 可以序列化它（因为它引用的是模块级函数）
        model.loss = types.MethodType(compute_distill_loss, model)

        return model

