import argparse
import sys
import os

# 将当前目录加入 python path，确保能导入 framework 模块
sys.path.append(os.getcwd())

from framework.core.engine import DetectionEngine

def main():
    parser = argparse.ArgumentParser(description="自研通用目标检测框架入口脚本")
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        default='framework/configs/train_plan.yaml',
        help='Path to the training configuration YAML file.'
    )
    
    args = parser.parse_args()
    
    print(r"""
    ========================================
       自研通用目标检测框架 v1.2
    ========================================
    """)

    try:
        engine = DetectionEngine(args.config)
        engine.train()
    except Exception as e:
        print(f"[错误] 执行失败 {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
