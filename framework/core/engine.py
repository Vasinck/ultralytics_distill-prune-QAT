import yaml
import os
import sys
import torch
from ultralytics import YOLO
from framework.core.distiller import DistillationTrainer

# 导入 C2f_v2 以支持剪枝后的模型
from framework.tools.prune import C2f_v2

# 将 C2f_v2 注册到 __main__ 模块以修复 pickle 加载问题
# 这样 torch.load 在加载剪枝模型时可以找到 C2f_v2
sys.modules['__main__'].C2f_v2 = C2f_v2

# 将 C2f_v2 添加到安全全局变量，支持 PyTorch 2.6+ 的 weights_only 加载
try:
    torch.serialization.add_safe_globals([C2f_v2])
except AttributeError:
    # PyTorch < 2.6 没有这个方法
    pass

class DetectionEngine:
    def __init__(self, config_path):
        """
        初始化检测引擎
        :param config_path: 训练任务配置文件路径 (yaml)
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.model = None

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件未找到: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def init_model(self):
        """
        加载模型 (Ultralytics Core Layer)
        支持加载剪枝后的模型（包含 C2f_v2 结构）
        """
        model_type = self.config.get('model_type', 'yolov8n.pt')
        print(f"[Framework] 正在初始化模型：{model_type}...")

        # 检测是否是剪枝后的模型（通过检查文件中是否包含 C2f_v2）
        if os.path.exists(model_type) and model_type.endswith('.pt'):
            try:
                # 对于自定义类（如 C2f_v2）需要使用 weights_only=False
                ckpt = torch.load(model_type, map_location='cpu', weights_only=False)
                # 检查模型结构中是否包含 C2f_v2
                if 'model' in ckpt:
                    model_state = str(ckpt['model'])
                    if 'C2f_v2' in model_state:
                        print(f"[Framework] 检测到包含 C2f_v2 结构的剪枝模型")
            except Exception as e:
                print(f"[Framework] 警告：无法检查模型文件：{e}")

        self.model = YOLO(model_type)

    def train(self):
        """
        执行训练流程 (支持普通训练、蒸馏训练和微调)
        """
        if self.model is None:
            self.init_model()

        # 准备训练参数
        train_cfg = self.config.get('training', {})
        data_cfg = self.config.get('data', {})

        # 确保数据配置路径是绝对路径
        data_path = data_cfg.get('config_path')
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)

        task_name = self.config.get('task_name', 'default_experiment')

        # 构建基础参数字典 (overrides)
        args = {
            'data': data_path,
            'project': self.config.get('project', 'runs/train'),
            'name': task_name,
            'epochs': train_cfg.get('epochs', 10),
            'batch': train_cfg.get('batch', 16),
            'imgsz': train_cfg.get('imgsz', 640),
            'device': train_cfg.get('device', ''),
            'workers': train_cfg.get('workers', 2),
            'optimizer': train_cfg.get('optimizer', 'auto'),
            'exist_ok': True,
            'model': self.config.get('model_type', 'yolov8n.pt')  # Trainer 需要知道模型文件
        }

        # 微调模式配置
        finetune_cfg = train_cfg.get('finetune', {})
        if finetune_cfg.get('enabled', False):
            print(f"[Framework] 微调模式已启用")
            # 微调通常需要较小的学习率
            if 'lr0' in finetune_cfg:
                args['lr0'] = finetune_cfg['lr0']
            if 'lrf' in finetune_cfg:
                args['lrf'] = finetune_cfg['lrf']
            # 微调时通常需要重置优化器状态
            if finetune_cfg.get('reset_optimizer', True):
                print(f"[Framework] 优化器状态将被重置")

        print(f"[Framework] 正在启动任务：{task_name}")

        # 检查是否启用蒸馏
        if train_cfg.get('distill', False):
            print(f"[Framework] 蒸馏模式已启用")
            distill_cfg = train_cfg.get('distillation', {})

            # 实例化自定义训练器
            # 注意：DistillationTrainer 继承自 DetectionTrainer
            # 我们需要手动传入 overrides 字典
            trainer = DistillationTrainer(
                distill_cfg=distill_cfg,
                overrides=args
            )
            trainer.train()

        else:
            # 标准训练流程
            print(f"[Framework] 标准训练模式")
            self.model.train(**args)

        project_dir = self.config.get('project', 'runs/train')
        print(f"[Framework] 训练完成。结果已保存至 {project_dir}/{task_name}")

if __name__ == "__main__":
    pass
