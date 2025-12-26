"""
自动化搜索脚本 (Automated Search)
给定目标约束（如 Params, FLOPs, Latency），自动遍历剪枝率，
寻找满足约束且精度损失最小的模型结构。
"""

import torch
import torch_pruning as tp
import argparse
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# 将项目根目录添加到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from framework.tools.prune import prune_model, C2f_v2
from framework.tools.benchmark import get_latency
from framework.core.engine import DetectionEngine
import yaml

# 注册 C2f_v2
sys.modules['__main__'].C2f_v2 = C2f_v2
try:
    torch.serialization.add_safe_globals([C2f_v2])
except AttributeError:
    pass


class PruningCandidate:
    """剪枝候选者，记录剪枝率和评估结果"""
    def __init__(self, ratio: float, params_M: float, flops_G: float,
                 latency_ms: Optional[float] = None, score: float = 0.0,
                 mAP50: Optional[float] = None, mAP50_95: Optional[float] = None):
        self.ratio = ratio
        self.params_M = params_M
        self.flops_G = flops_G
        self.latency_ms = latency_ms
        self.score = score
        self.model_path = None
        self.mAP50 = mAP50
        self.mAP50_95 = mAP50_95

    def __repr__(self):
        map_str = f", mAP50={self.mAP50:.4f}" if self.mAP50 is not None else ""
        return (f"候选者(ratio={self.ratio:.2f}, params={self.params_M:.2f}M, "
                f"flops={self.flops_G:.2f}G{map_str}, score={self.score:.4f})")


class AutoSearch:
    """自动化搜索器"""

    def __init__(self, base_model: str, constraints: Dict, search_space: Dict,
                 save_dir: str = "runs/search"):
        """
        初始化搜索器

        Args:
            base_model: 基础模型路径
            constraints: 约束条件，例如 {'params': 2.0, 'flops': 4.0, 'latency': 10.0}
            search_space: 搜索空间，例如 {'min': 0.1, 'max': 0.5, 'step': 0.05}
            save_dir: 保存目录
        """
        self.base_model_path = base_model
        self.constraints = constraints
        self.search_space = search_space
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 加载基础模型
        print(f"\n[Search] Loading base model from {base_model}...")
        self.base_model = YOLO(base_model).model
        self.base_model.eval()

        # 测量基础模型指标
        self._calibrate_base_model()

        # 候选者列表
        self.candidates: List[PruningCandidate] = []

    def _calibrate_base_model(self):
        """校准：测量基础模型的 FLOPs/Params"""
        print("\n[Search] Calibrating base model...")

        example_input = torch.randn(1, 3, 640, 640)
        # count_ops_and_params 返回 (MACs, params)
        self.base_flops, self.base_params = tp.utils.count_ops_and_params(self.base_model, example_input)

        print(f"[Search] Base model - Params: {self.base_params/1e6:.2f}M, FLOPs: {self.base_flops/1e9:.2f}G")

        # 如果有 latency 约束，测量基础模型延迟
        if 'latency' in self.constraints:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            latency_stats = get_latency(self.base_model, device=device,
                                       warmup_runs=5, measure_runs=20)
            self.base_latency = latency_stats['mean_ms']
            print(f"[Search] Base model - Latency: {self.base_latency:.2f}ms")
        else:
            self.base_latency = None

    def generate_search_space(self) -> List[float]:
        """生成搜索空间（剪枝率列表）"""
        min_ratio = self.search_space.get('min', 0.1)
        max_ratio = self.search_space.get('max', 0.5)
        step = self.search_space.get('step', 0.05)

        # 生成剪枝率列表
        ratios = np.arange(min_ratio, max_ratio + step/2, step)
        ratios = [float(r) for r in ratios]

        print(f"\n[Search] Search space: {len(ratios)} candidates with ratios {ratios}")
        return ratios

    def prune_and_evaluate_tier1(self, ratio: float) -> Optional[PruningCandidate]:
        """
        Tier 1 评估：剪枝并评估静态指标（Params, FLOPs）

        Args:
            ratio: 剪枝率

        Returns:
            PruningCandidate 或 None（如果剪枝失败）
        """
        print(f"\n[Search] Evaluating ratio={ratio:.2f}...")

        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            temp_model_path = tmp_file.name

        try:
            # 剪枝
            print(f"[Search]   Pruning with ratio={ratio:.2f}...")
            prune_model(
                model_path=self.base_model_path,
                save_path=temp_model_path,
                pruning_ratio=ratio,
                example_input_shape=(1, 3, 640, 640)
            )

            # 加载剪枝后的模型
            ckpt = torch.load(temp_model_path, map_location='cpu', weights_only=False)
            pruned_model = ckpt['model']
            pruned_model.eval()

            # 计算指标
            example_input = torch.randn(1, 3, 640, 640)
            # count_ops_and_params 返回 (MACs, params)
            pruned_flops, pruned_params = tp.utils.count_ops_and_params(pruned_model, example_input)

            params_M = pruned_params / 1e6
            flops_G = pruned_flops / 1e9

            print(f"[Search]   Pruned - Params: {params_M:.2f}M ({100*(self.base_params-pruned_params)/self.base_params:.1f}% reduction)")
            print(f"[Search]   Pruned - FLOPs: {flops_G:.2f}G ({100*(self.base_flops-pruned_flops)/self.base_flops:.1f}% reduction)")

            # 检查是否满足约束
            meets_constraints = True
            if 'params' in self.constraints:
                if params_M > self.constraints['params']:
                    print(f"[Search]   ❌ Params constraint not met: {params_M:.2f}M > {self.constraints['params']:.2f}M")
                    meets_constraints = False

            if 'flops' in self.constraints:
                if flops_G > self.constraints['flops']:
                    print(f"[Search]   ❌ FLOPs constraint not met: {flops_G:.2f}G > {self.constraints['flops']:.2f}G")
                    meets_constraints = False

            if not meets_constraints:
                os.unlink(temp_model_path)
                return None

            print(f"[Search]   ✅ Meets static constraints")

            # 创建候选者
            candidate = PruningCandidate(
                ratio=ratio,
                params_M=params_M,
                flops_G=flops_G
            )

            # 保存模型到永久位置
            candidate_name = f"candidate_ratio_{ratio:.2f}.pt"
            candidate_path = self.save_dir / candidate_name
            shutil.move(temp_model_path, str(candidate_path))
            candidate.model_path = str(candidate_path)

            return candidate

        except Exception as e:
            print(f"[Search]   ❌ Error during pruning: {e}")
            if os.path.exists(temp_model_path):
                os.unlink(temp_model_path)
            return None

    def score_candidate(self, candidate: PruningCandidate) -> float:
        """
        为候选者打分（Tier 1：基于静态指标）

        评分策略：在满足约束的前提下，剪枝率越小越好（保留更多参数，精度损失更小）

        Args:
            candidate: 候选者

        Returns:
            分数（越高越好）
        """
        # 简单策略：剪枝率越小，分数越高
        # 分数范围 [0, 1]，ratio=0 得分 1.0，ratio=1 得分 0
        score = 1.0 - candidate.ratio
        candidate.score = score
        return score

    def run_tier1_search(self) -> List[PruningCandidate]:
        """
        执行 Tier 1 搜索（静态评估）

        Returns:
            满足约束的候选者列表
        """
        print("\n" + "="*80)
        print("开始 Tier 1 搜索（静态评估）")
        print("="*80)

        # 生成搜索空间
        ratios = self.generate_search_space()

        # 遍历剪枝率
        valid_candidates = []
        for ratio in ratios:
            candidate = self.prune_and_evaluate_tier1(ratio)
            if candidate is not None:
                # 打分
                score = self.score_candidate(candidate)
                valid_candidates.append(candidate)
                print(f"[Search]   Score: {score:.4f}")

        # 按分数排序
        valid_candidates.sort(key=lambda x: x.score, reverse=True)

        print(f"\n[Search] Tier 1 complete: {len(valid_candidates)}/{len(ratios)} candidates meet constraints")

        return valid_candidates

    def quick_finetune_and_evaluate(self, candidate: PruningCandidate,
                                    data_config: str, epochs: int = 2) -> Optional[Dict]:
        """
        Tier 2 评估：快速微调候选模型并评估精度

        Args:
            candidate: 候选者
            data_config: 数据集配置文件路径
            epochs: 快速训练的 epoch 数

        Returns:
            评估结果字典，包含 mAP 等指标
        """
        print(f"\n[Search] Tier 2 - Quick finetuning candidate ratio={candidate.ratio:.2f}...")

        # 创建临时训练配置
        temp_config_path = self.save_dir / f"temp_train_config_{candidate.ratio:.2f}.yaml"

        train_config = {
            'task_name': f'search_finetune_{candidate.ratio:.2f}',
            'model_type': candidate.model_path,
            'data': {
                'config_path': data_config
            },
            'training': {
                'epochs': epochs,
                'batch': 4,  # 小 batch 加快速度
                'imgsz': 640,
                'device': '',
                'workers': 2,
                'optimizer': 'SGD',
                'finetune': {
                    'enabled': True,
                    'lr0': 0.001,
                    'lrf': 0.1,
                    'reset_optimizer': True
                },
                'distill': True,
                'distillation': {
                    'teacher': self.base_model_path,  # 使用原始模型作为 Teacher
                    'loss_type': 'logits',
                    'alpha': 0.5,
                    'T': 2.0
                }
            }
        }

        # 保存临时配置
        with open(temp_config_path, 'w') as f:
            yaml.dump(train_config, f)

        try:
            # 运行快速训练
            print(f"[Search]   Training for {epochs} epochs...")
            engine = DetectionEngine(str(temp_config_path))
            engine.train()

            # 读取训练结果
            results_path = Path('runs/train') / train_config['task_name'] / 'results.csv'

            if results_path.exists():
                # 手动解析 CSV 文件（不依赖 pandas）
                with open(results_path, 'r') as f:
                    lines = f.readlines()

                if len(lines) < 2:
                    print(f"[Search]   Warning: Results file is empty")
                    return None

                # 解析表头
                header = lines[0].strip().split(',')
                header = [h.strip() for h in header]

                # 解析最后一行数据
                last_line = lines[-1].strip().split(',')
                last_line = [v.strip() for v in last_line]

                # 创建字典
                data_dict = dict(zip(header, last_line))

                # 提取指标
                metrics = {
                    'mAP50': float(data_dict.get('metrics/mAP50(B)', 0.0)),
                    'mAP50_95': float(data_dict.get('metrics/mAP50-95(B)', 0.0)),
                    'box_loss': float(data_dict.get('train/box_loss', 0.0)),
                    'cls_loss': float(data_dict.get('train/cls_loss', 0.0))
                }

                print(f"[Search]   Results - mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}")

                # 清理训练输出目录（可选，节省空间）
                # shutil.rmtree(Path('runs/train') / train_config['task_name'])

                return metrics
            else:
                print(f"[Search]   Warning: Results file not found")
                return None

        except Exception as e:
            print(f"[Search]   Error during quick finetuning: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 清理临时配置文件
            if temp_config_path.exists():
                temp_config_path.unlink()

    def run_tier2_search(self, candidates: List[PruningCandidate],
                        data_config: str, epochs: int = 2,
                        top_k: int = None) -> List[PruningCandidate]:
        """
        执行 Tier 2 搜索（快速微调评估）

        Args:
            candidates: Tier 1 筛选出的候选者列表
            data_config: 数据集配置文件路径
            epochs: 快速训练的 epoch 数
            top_k: 只评估 Top K 个候选者（None 表示全部）

        Returns:
            经过 Tier 2 评估的候选者列表
        """
        print("\n" + "="*80)
        print("开始 Tier 2 搜索（快速微调评估）")
        print("="*80)

        if not candidates:
            print("[Search] No candidates to evaluate in Tier 2")
            return []

        # 限制评估数量
        eval_candidates = candidates[:top_k] if top_k else candidates
        print(f"[Search] Evaluating {len(eval_candidates)}/{len(candidates)} candidates")

        # 快速微调并评估
        for candidate in eval_candidates:
            metrics = self.quick_finetune_and_evaluate(candidate, data_config, epochs)

            if metrics:
                candidate.mAP50 = metrics['mAP50']
                candidate.mAP50_95 = metrics['mAP50_95']

                # 重新打分：基于 mAP50（权重 0.7）和剪枝率（权重 0.3）
                # 剪枝率越小越好，mAP 越高越好
                map_score = candidate.mAP50  # 0-1 范围
                ratio_score = 1.0 - candidate.ratio  # 剪枝率越小分数越高
                candidate.score = 0.7 * map_score + 0.3 * ratio_score

                print(f"[Search]   Updated score: {candidate.score:.4f}")

        # 按新分数重新排序
        eval_candidates.sort(key=lambda x: x.score, reverse=True)

        print(f"\n[Search] Tier 2 complete")
        return eval_candidates

    def save_results(self, candidates: List[PruningCandidate]):
        """保存搜索结果"""
        results_path = self.save_dir / "search_results.json"

        results = {
            'base_model': self.base_model_path,
            'constraints': self.constraints,
            'search_space': self.search_space,
            'base_metrics': {
                'params_M': self.base_params / 1e6,
                'flops_G': self.base_flops / 1e9,
                'latency_ms': self.base_latency if self.base_latency else None
            },
            'candidates': [
                {
                    'ratio': c.ratio,
                    'params_M': c.params_M,
                    'flops_G': c.flops_G,
                    'latency_ms': c.latency_ms,
                    'mAP50': c.mAP50,
                    'mAP50_95': c.mAP50_95,
                    'score': c.score,
                    'model_path': c.model_path
                }
                for c in candidates
            ]
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[Search] Results saved to {results_path}")

    def print_summary(self, candidates: List[PruningCandidate]):
        """打印搜索总结"""
        print("\n" + "="*80)
        print("搜索总结")
        print("="*80)

        if not candidates:
            print("没有候选者满足约束条件！")
            return

        print(f"\n找到 {len(candidates)} 个满足约束条件的候选者：")

        # 检查是否有 mAP 数据
        has_map = any(c.mAP50 is not None for c in candidates)

        if has_map:
            print(f"\n{'排名':<6} {'剪枝率':<8} {'参数量(M)':<12} {'FLOPs(G)':<12} {'mAP50':<10} {'分数':<10} {'路径'}")
            print("-"*90)
            for i, c in enumerate(candidates[:10], 1):  # 显示前 10 个
                map_str = f"{c.mAP50:.4f}" if c.mAP50 is not None else "N/A"
                print(f"{i:<6} {c.ratio:<8.2f} {c.params_M:<12.2f} {c.flops_G:<12.2f} {map_str:<10} {c.score:<10.4f} {Path(c.model_path).name}")
        else:
            print(f"\n{'排名':<6} {'剪枝率':<8} {'参数量(M)':<12} {'FLOPs(G)':<12} {'分数':<10} {'路径'}")
            print("-"*80)
            for i, c in enumerate(candidates[:10], 1):  # 显示前 10 个
                print(f"{i:<6} {c.ratio:<8.2f} {c.params_M:<12.2f} {c.flops_G:<12.2f} {c.score:<10.4f} {Path(c.model_path).name}")

        print("\n" + "="*80)
        print(f"最佳候选者：ratio={candidates[0].ratio:.2f}")
        print(f"  参数量：{candidates[0].params_M:.2f}M（减少 {100*(self.base_params/1e6-candidates[0].params_M)/(self.base_params/1e6):.1f}%）")
        print(f"  FLOPs：{candidates[0].flops_G:.2f}G（减少 {100*(self.base_flops/1e9-candidates[0].flops_G)/(self.base_flops/1e9):.1f}%）")
        print(f"  模型保存位置：{candidates[0].model_path}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="自研通用目标检测框架 - 自动化搜索工具")
    parser.add_argument('--model', type=str, required=True, help='Path to base model (.pt)')
    parser.add_argument('--target-params', type=float, default=None, help='Target params constraint (M)')
    parser.add_argument('--target-flops', type=float, default=None, help='Target FLOPs constraint (G)')
    parser.add_argument('--target-latency', type=float, default=None, help='Target latency constraint (ms)')
    parser.add_argument('--min-ratio', type=float, default=0.1, help='Minimum pruning ratio')
    parser.add_argument('--max-ratio', type=float, default=0.5, help='Maximum pruning ratio')
    parser.add_argument('--step', type=float, default=0.05, help='Pruning ratio step size')
    parser.add_argument('--save-dir', type=str, default='runs/search', help='Directory to save results')

    # Tier 2 评估参数
    parser.add_argument('--enable-tier2', action='store_true', help='Enable Tier 2 evaluation (quick finetuning)')
    parser.add_argument('--tier2-data', type=str, default='framework/configs/unified_dataset.yaml',
                       help='Data config for Tier 2 evaluation')
    parser.add_argument('--tier2-epochs', type=int, default=2, help='Number of epochs for quick finetuning')
    parser.add_argument('--tier2-topk', type=int, default=None, help='Only evaluate top K candidates in Tier 2')

    args = parser.parse_args()

    # 构建约束条件
    constraints = {}
    if args.target_params is not None:
        constraints['params'] = args.target_params
    if args.target_flops is not None:
        constraints['flops'] = args.target_flops
    if args.target_latency is not None:
        constraints['latency'] = args.target_latency

    if not constraints:
        print("Error: At least one constraint must be specified (--target-params, --target-flops, or --target-latency)")
        return

    # 构建搜索空间
    search_space = {
        'min': args.min_ratio,
        'max': args.max_ratio,
        'step': args.step
    }

    # 运行搜索
    searcher = AutoSearch(
        base_model=args.model,
        constraints=constraints,
        search_space=search_space,
        save_dir=args.save_dir
    )

    # Tier 1 搜索
    candidates = searcher.run_tier1_search()

    # Tier 2 搜索（可选）
    if args.enable_tier2 and candidates:
        candidates = searcher.run_tier2_search(
            candidates=candidates,
            data_config=args.tier2_data,
            epochs=args.tier2_epochs,
            top_k=args.tier2_topk
        )

    # 保存结果
    searcher.save_results(candidates)

    # 打印总结
    searcher.print_summary(candidates)


if __name__ == "__main__":
    main()
