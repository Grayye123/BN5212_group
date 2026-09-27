"""按本机 GPU 实测 ResNet18 训练的显存占用，用于确定阶段 4 的分辨率与 batch size。

沿用 PROJECT.md 2026-09-08 那次测试的协议，便于对照：随机初始化 ResNet18（不下载预训练
权重）、随机三通道输入、AdamW、AMP float16、全部参数更新，每个设置跑 3 个更新步，
取张量峰值与框架保留峰值。这是资源测试，不使用任何真实胸片，不是模型性能结果。

用法：
    python code/audit/vram_probe.py                 # 默认若干组设置
    python code/audit/vram_probe.py --sizes 224,512 --batches 8,16,32
"""
import argparse
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

GIB = 1 << 30


def probe(size, batch, steps=3):
    import torch
    from torch import nn, optim
    from torchvision.models import resnet18

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dev = torch.device('cuda')

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(dev)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        x = torch.randn(batch, 3, size, size, device=dev)
        y = torch.randint(0, 2, (batch, 1), device=dev).float()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated() / GIB
    peak_res = torch.cuda.max_memory_reserved() / GIB
    del model, opt, scaler, x, y, loss
    torch.cuda.empty_cache()
    return peak_alloc, peak_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='224,512')
    ap.add_argument('--batches', default='8,16,32')
    ap.add_argument('--steps', type=int, default=3)
    a = ap.parse_args()

    import torch
    print('torch', torch.__version__, '| 编译 CUDA', torch.version.cuda, flush=True)
    if not torch.cuda.is_available():
        print('CUDA 不可用，无法实测。')
        return 1
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    print('GPU: %s | 总显存 %.2f GiB | 架构 %s' % (name, total, torch.cuda.get_arch_list()), flush=True)
    print()
    print('%-16s %14s %16s %s' % ('输入 / batch', '张量峰值', '框架保留峰值', '建议进程预算'))
    for size in [int(s) for s in a.sizes.split(',')]:
        for batch in [int(b) for b in a.batches.split(',')]:
            try:
                alloc, res = probe(size, batch, a.steps)
                budget = res * 1.5 + 0.5
                print('%-16s %11.2f GiB %12.2f GiB   约 %.1f GiB'
                      % ('%d / %d' % (size, batch), alloc, res, budget), flush=True)
            except torch.cuda.OutOfMemoryError:
                print('%-16s %s' % ('%d / %d' % (size, batch), '显存不足 (OOM)'), flush=True)
                torch.cuda.empty_cache()
    print()
    print('GiB = 2^30 字节。峰值不含整卡其他占用；真实数据读取与增强会再增加开销。')
    print('这是资源测试，不是模型性能结果。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
