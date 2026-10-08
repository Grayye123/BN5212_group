"""Verify real checkpoints and export an allowlisted aggregate-only bundle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import time

from final_study import METRICS, SEEDS, digest, read_rows, report, verify, verify_result, write_json


def verify_checkpoints(dest, plan):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18
    from pilot_train import Images, predict, seed_all
    seed_all()
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    checks = []
    for doctor in plan['doctors']:
        for seed in SEEDS:
            run = dest / doctor['name'] / f'seed_{seed}'
            verify_result(run / 'results.json', plan, doctor, seed)
            rows = read_rows(run / 'test.csv')
            predictions = read_rows(run / 'predictions.csv')
            loader = DataLoader(Images(rows, dest / 'pixels224'), batch_size=32, shuffle=False, num_workers=0)
            for name in ('seen', 'unseen'):
                path = run / f'{name}.pt'
                model.load_state_dict(torch.load(path, weights_only=True, map_location=device), strict=True)
                regenerated = predict(model, loader, device)
                expected = np.asarray([float(r[name]) for r in predictions])
                error = float(np.max(np.abs(regenerated-expected)))
                if error > 1e-6:
                    raise ValueError('Checkpoint re-inference does not match saved predictions')
                checks.append(dict(provider=doctor['name'], seed=seed, model=name, sha256=digest(path),
                                   bytes=path.stat().st_size, predictions=len(rows), max_abs_score_error=error))
                print(json.dumps(dict(stage='checkpoint_verified', provider=doctor['name'], seed=seed, model=name, max_abs_score_error=error)), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return checks


def plot_intervals(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#2266AA', '#B87716', '#536F45')
    markers = ('o', 's', '^')
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 6.0), sharey=True)
    for ax, metric, title in zip(axes, METRICS, ('AUROC difference', 'Average precision difference', 'Brier difference')):
        for i, run in enumerate(data['runs']):
            lo, hi = run['paired_bootstrap']['intervals'][metric]
            key = SEEDS.index(run['seed'])
            ax.hlines(i, lo, hi, color=colors[key], linewidth=1.8)
            ax.scatter(run['difference'][metric], i, color=colors[key], marker=markers[key], s=42, zorder=3)
        ax.axvline(0, color='#555555', linewidth=1, linestyle='--')
        ax.set_title(title, fontsize=13, loc='left', pad=12)
        ax.grid(axis='x', color='#DDDDDD', linewidth=.6)
        ax.set_xlabel('Seen minus unseen', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(data['runs'])), [f"{r['provider']}  seed {SEEDS.index(r['seed'])+1}" for r in data['runs']])
    axes[0].invert_yaxis()
    fig.suptitle('Fixed provider groups and repeated training seeds', fontsize=17, x=.19, ha='left')
    fig.text(.19, .04, 'Bars: 95% paired patient bootstrap intervals conditional on each trained model pair.\nAUROC / AP: positive favors seen. Brier: negative favors seen. Provider test patients overlap.', fontsize=10, color='#444444')
    fig.subplots_adjust(left=.19, right=.98, top=.85, bottom=.19, wspace=.28)
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)


def publish(source, private, destination):
    if destination.exists():
        raise FileExistsError('Use a fresh public bundle; do not overwrite frozen results')
    plan = verify(source, private)
    report(private, plan)
    data = json.loads((private / 'summary_results.json').read_text(encoding='utf-8'))
    audit = json.loads((private / 'audit_summary.json').read_text(encoding='utf-8'))
    if audit['source_cohort_sha256'] != plan['source_cohort_sha256'] or audit['source_labels_sha256'] != plan['source_labels_sha256']:
        raise ValueError('Audit does not match frozen experiment')
    checkpoint_checks = verify_checkpoints(private, plan)
    verification = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                        actual_checks=['frozen source and manifest hashes', 'patient isolation and label/view matching',
                                       'prediction keys/labels against each test manifest', 'independent AUROC/AP/Brier recomputation',
                                       'strict checkpoint loading and full test re-inference for all 18 models'],
                        model_checks=checkpoint_checks, paired_runs=9, model_fits=18,
                        image_cache_count=len(list((private / 'pixels224').glob('*.npy'))),
                        training_code_sha256=plan['training_code_sha256'])
    build = private / f'public_build_{time.time_ns()}'
    build.mkdir()
    # Explicit aggregate objects only: never copy plan.json, CSV cases, paths, reports or model bytes.
    write_json(build / 'results.json', data)
    write_json(build / 'audit.json', {k: audit[k] for k in ('status', 'frozen_labels', 'frozen_cohort', 'label_counts', 'source_counts',
              'reconciliation_counts', 'human_rows', 'changed_from_human', 'additional_independent_human_reviews',
              'additional_expert_image_reviews', 'case_mix', 'between_provider_test_patient_overlap', 'limitations')})
    write_json(build / 'verification.json', verification)
    write_json(build / 'protocol.json', dict(model='ImageNet ResNet18', resolution=224, seeds=list(SEEDS), pair_seed=plan['pair_seed'],
               epochs=5, optimizer='AdamW', lr=1e-4, weight_decay=1e-4, batch_size=32, selection='shared validation BCE',
               eligibility=plan['selection'], source_cohort_sha256=plan['source_cohort_sha256'], source_labels_sha256=plan['source_labels_sha256'],
               providers=[{k: d[k] for k in ('name', 'sizes', 'seen_target_studies')} for d in plan['doctors']],
               bootstrap='1000 paired patient samples per model pair; seed spread descriptive only'))
    plot_intervals(data, build / 'paired_differences.png')
    text = ['# 期末扩展 · 真实实验结果', '',
            '3 个预先按可用规模选定的医生关联组，3 个固定训练种子，seen/unseen 各 5 轮；18 个 ImageNet ResNet18 模型真实训练完成。复用已有标签和病人划分，属于扩展复测。', '',
            '## 结果入口', '', 'results.json：9 次模型对的完整指标/条件病人区间及跨种子描述；protocol.json：冻结设置和规模；audit.json：结构与可观测病例构成；verification.json：此次实际核查及权重校验值。权重和逐病例预测仅留在授权本机。', '',
            '![配对差值与条件病人区间](paired_differences.png)', '',
            '## AUROC 和种子变化', '', '| 组 | seen AUROC 均值 | unseen AUROC 均值 | 差值均值 | 差值样本标准差 | 差值范围 |',
            '| --- | ---: | ---: | ---: | ---: | --- |']
    for group in data['provider_seed_summary']:
        runs = [r for r in data['runs'] if r['provider'] == group['provider']]
        d = group['differences']['AUROC']
        text.append(f"| {group['provider']} | {statistics.mean(r['seen']['AUROC'] for r in runs):.3f} | {statistics.mean(r['unseen']['AUROC'] for r in runs):.3f} | {d['mean']:+.3f} | {d['sample_sd']:.3f} | [{d['minimum']:+.3f}, {d['maximum']:+.3f}] |")
    crossing = sum(r['paired_bootstrap']['intervals']['AUROC'][0] <= 0 <= r['paired_bootstrap']['intervals']['AUROC'][1] for r in data['runs'])
    text += ['', f'{crossing}/9 个 AUROC 差值的条件病人 95% 区间包含 0。逐次区间见结果 JSON 和图。该计数没有多重比较校正，不作为总体确认性推断。', '',
             '差值定义为 seen−unseen。AUROC/AP 越高越好，Brier 越低越好。AP 字段沿用 AUPRC，但算法为 average precision。均值/标准差/范围来自 3 次训练，仅作描述，不是包含训练随机性的置信区间。', '',
             '## 解释边界', '',
             '- 标签来自报告；此次新增独立人工/专家影像验证均为 0。结构核查不等于标签准确率验证，未提及不代表医学上无病。',
             '- 每个医生内固定测试病例，跨医生测试病人有重叠；不能把重复种子或重叠病人作为新增独立样本。',
             '- 两训练组平衡标签和体位，其他临床病例构成没有完全匹配。结果不证明医生因果作用或医生水平。',
             '- 固定小规模试验已评估过部分测试病例，本次不是新的外部验证。',
             '- 常数分数基线来自训练阳性率，不读取图像；没有额外架构对照或新算法声明。', '',
             '## 复现和交付', '',
             '代码见 [运行说明](../../code/README.md)，研究协议见 [PROJECT.md](../../docs/PROJECT.md)。克隆能获取公开材料，完整重跑仍需个人数据授权及私有输入。视频要求见 [录制说明](../../video/README.md)。', '']
    (build / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    sums = '\n'.join(f'{digest(path)}  {path.name}' for path in sorted(build.iterdir()) if path.is_file()) + '\n'
    (build / 'SHA256SUMS.txt').write_text(sums, encoding='utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(build, destination)
    print(json.dumps(dict(stage='public_bundle_created', models_verified=len(checkpoint_checks), paired_runs=9, auroc_intervals_containing_zero=crossing)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('private/midterm_v1'))
    parser.add_argument('--private', type=Path, default=Path('private/final_v1'))
    parser.add_argument('--dest', type=Path, default=Path('results/final_v1'))
    args = parser.parse_args()
    publish(args.source, args.private, args.dest)
