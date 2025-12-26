"""
延迟基准测试工具 (Latency Benchmark)
用于评估模型在当前设备上的实际推理速度
"""

import torch
import time
import csv
import os
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import argparse

# 将项目根目录添加到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入 C2f_v2 以支持剪枝后的模型
try:
    from framework.tools.prune import C2f_v2
    sys.modules['__main__'].C2f_v2 = C2f_v2
    torch.serialization.add_safe_globals([C2f_v2])
except (ImportError, AttributeError):
    pass


def get_latency(model, input_shape=(1, 3, 640, 640), warmup_runs=10, measure_runs=50, device='cuda'):
    """
    测量模型的推理延迟

    Args:
        model: PyTorch 模型或 YOLO 模型
        input_shape: 输入张量形状 (batch, channels, height, width)
        warmup_runs: 预热次数
        measure_runs: 测量次数
        device: 设备 ('cuda' or 'cpu')

    Returns:
        dict: 包含延迟统计信息的字典
    """
    # 如果是 YOLO 对象，提取内部的 PyTorch 模型
    if isinstance(model, YOLO):
        model = model.model

    # 确保模型在正确的设备上
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        device_obj = torch.device('cuda')
    else:
        model = model.cpu()
        device_obj = torch.device('cpu')
        device = 'cpu'

    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(input_shape).to(device_obj)

    print(f"[Benchmark] Warming up model for {warmup_runs} iterations...")
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # 如果使用 CUDA，同步以确保预热完成
    if device == 'cuda':
        torch.cuda.synchronize()

    print(f"[Benchmark] Measuring latency over {measure_runs} iterations...")
    latencies = []

    with torch.no_grad():
        for _ in range(measure_runs):
            if device == 'cuda':
                # 使用 CUDA Event 进行精确计时
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = model(dummy_input)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
                latencies.append(elapsed_time)
            else:
                # CPU 使用 time.perf_counter
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
                latencies.append(elapsed_time)

    # 计算统计数据
    latencies = np.array(latencies)
    stats = {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'fps': float(1000.0 / np.mean(latencies)),
        'device': device,
        'input_shape': str(input_shape),
        'warmup_runs': warmup_runs,
        'measure_runs': measure_runs
    }

    return stats


def benchmark_model(model_path, input_shape=(1, 3, 640, 640), warmup_runs=10, measure_runs=50, device=''):
    """
    对给定的模型文件进行基准测试

    Args:
        model_path: 模型文件路径 (.pt)
        input_shape: 输入形状
        warmup_runs: 预热次数
        measure_runs: 测量次数
        device: 设备 ('' for auto, 'cuda', 'cpu')

    Returns:
        dict: 包含模型信息和延迟统计的字典
    """
    print(f"\n[Benchmark] Loading model from {model_path}...")

    # 加载模型
    model = YOLO(model_path)

    # 自动选择设备
    if device == '':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[Benchmark] Device: {device}")
    print(f"[Benchmark] Input shape: {input_shape}")

    # 获取模型参数量和 FLOPs
    try:
        from ultralytics.utils.torch_utils import model_info
        info = model_info(model.model, verbose=False)
        params = info['params']
        flops = info['GFLOPs']
    except:
        # 备用方法
        params = sum(p.numel() for p in model.model.parameters())
        flops = None

    print(f"[Benchmark] Parameters: {params / 1e6:.2f}M")
    if flops:
        print(f"[Benchmark] FLOPs: {flops:.2f}G")

    # 测量延迟
    stats = get_latency(model, input_shape, warmup_runs, measure_runs, device)

    # 添加模型信息
    result = {
        'model_path': model_path,
        'model_name': Path(model_path).stem,
        'params_M': params / 1e6,
        'flops_G': flops if flops else 'N/A',
        **stats
    }

    return result


def print_benchmark_results(results):
    """
    打印基准测试结果
    """
    print("\n" + "="*80)
    print("Benchmark Results")
    print("="*80)
    print(f"Model: {results['model_name']}")
    print(f"Parameters: {results['params_M']:.2f}M")
    print(f"FLOPs: {results['flops_G']}G" if isinstance(results['flops_G'], float) else f"FLOPs: {results['flops_G']}")
    print(f"Device: {results['device']}")
    print(f"Input Shape: {results['input_shape']}")
    print("-"*80)
    print(f"Mean Latency:   {results['mean_ms']:.2f} ms")
    print(f"Std Latency:    {results['std_ms']:.2f} ms")
    print(f"Median Latency: {results['median_ms']:.2f} ms")
    print(f"Min Latency:    {results['min_ms']:.2f} ms")
    print(f"Max Latency:    {results['max_ms']:.2f} ms")
    print(f"P95 Latency:    {results['p95_ms']:.2f} ms")
    print(f"P99 Latency:    {results['p99_ms']:.2f} ms")
    print("-"*80)
    print(f"Throughput:     {results['fps']:.2f} FPS")
    print("="*80)


def save_to_csv(results, csv_path='benchmark_results.csv'):
    """
    将结果保存到 CSV 文件
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

    # 检查文件是否存在
    file_exists = os.path.exists(csv_path)

    # 定义字段顺序
    fieldnames = [
        'model_name', 'model_path', 'params_M', 'flops_G', 'device', 'input_shape',
        'mean_ms', 'std_ms', 'median_ms', 'min_ms', 'max_ms', 'p95_ms', 'p99_ms', 'fps',
        'warmup_runs', 'measure_runs'
    ]

    # 写入 CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writeheader()

        # 写入数据
        writer.writerow(results)

    print(f"\n[Benchmark] Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="自研通用目标检测框架 - 延迟基准测试工具")
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pt)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--runs', type=int, default=50, help='Number of measurement runs')
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu, empty for auto)')
    parser.add_argument('--save', type=str, default='', help='Path to save CSV results (optional)')

    args = parser.parse_args()

    # 构建输入形状
    input_shape = (args.batch, 3, args.imgsz, args.imgsz)

    # 运行基准测试
    results = benchmark_model(
        model_path=args.model,
        input_shape=input_shape,
        warmup_runs=args.warmup,
        measure_runs=args.runs,
        device=args.device
    )

    # 打印结果
    print_benchmark_results(results)

    # 保存到 CSV（如果指定）
    if args.save:
        save_to_csv(results, args.save)


if __name__ == "__main__":
    main()
