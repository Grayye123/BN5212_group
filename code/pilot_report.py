"""Export aggregate-only midterm notes after a completed real experiment."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, default=Path('private/midterm_v1'))
    p.add_argument('--dest', type=Path, default=Path('slides/MIDTERM_V1.md'))
    a = p.parse_args()
    result = json.loads((a.work/'results.json').read_text(encoding='utf-8'))
    sample = json.loads((a.work/'summary.json').read_text(encoding='utf-8'))
    assert result['status'] == 'completed_real_image_experiment'
    ci = result['paired_bootstrap']['intervals']
    table = ['| 指标 | 包含目标医生病例 | 排除目标医生病例 | 差值（包含−排除） | 差值的病人配对 95% 区间 |',
             '| --- | ---: | ---: | ---: | --- |']
    for k in ('AUROC', 'AUPRC', 'Brier'):
        interval = f'[{ci[k][0]:.3f}, {ci[k][1]:.3f}]' if ci[k] else '无法估计'
        table.append(f"| {k} | {result['seen'][k]:.3f} | {result['unseen'][k]:.3f} | {result['difference'][k]:+.3f} | {interval} |")
    auc_ci = ci['AUROC']
    if auc_ci and auc_ci[0] <= 0 <= auc_ci[1]:
        finding = 'AUROC 差值区间包含 0，这一小规模实验未提供稳定的组间差异证据。'
    elif auc_ci:
        finding = '本次 AUROC 差值区间未包含 0，但结果仅覆盖一个医生关联组和一个训练种子，仍需重复实验。'
    else:
        finding = '无法稳定估计 AUROC 差值区间，不作组间优劣判断。'
    best = {name: min([h for h in result['history'] if h['model'] == name], key=lambda h:h['valid_loss'])['epoch'] for name in ('seen','unseen')}
    text = f'''# 期中第一版实验：训练是否见过目标医生关联病例

本页只记录真实运行结果和可直接用于汇报的要点。7 分钟汇报、四人均发言；成员与具体页码分工由小组自行确认。

## 1. 问题、动机与相关工作（约 70 秒）

我们比较同一种胸片模型：训练中包含或排除目标医生关联病例，是否影响它在同一批目标病例上的肺不张报告标签预测？医生编号只用于研究分组，模型不读取医生编号。

这是一项小规模稳健性探索，不用于评价医生水平，也不证明医生造成了性能变化。

方法基础：[He 等的残差网络工作](https://arxiv.org/abs/1512.03385)提供图像模型架构；[Smit 等的 CheXbert 工作](https://arxiv.org/abs/2004.09167)研究从放射报告抽取标签。我们的初版重点是医生关联病例的配对比较，不是发明新的网络或将报告标签当作影像金标准。

## 2. 数据与标签（约 50 秒）

课程子集为 5,534 张影像、3,351 次检查、1,000 位病人。CheXbert 先生成报告标签，四人完成 222 条分歧与 50 条一致样本判定；助手依照小组规则做保守自动协调，保留原始判定。

likely 记不确定，most likely 记阳性；非常轻微 volume loss 不计入，其他表述按规则处理；难以可靠解析的冲突排除。四态最终为阳性 731、明确阴性 11、不确定 333、未提及 2,276。不确定排除，未提及按未明确报出记 0，不等于医学上确定无病。

选片与标签过滤后保留 {sample['cohort']['studies']} 次检查、{sample['cohort']['patients']} 位病人。报告标签不是独立影像金标准；未新增人工检查机械未提及部分。

## 3. 实验控制（约 80 秒）

按病人以 60/20/20 分为训练、验证和测试，种子 20260927，病人不跨集合。目标医生按可用独立病人数最多选择，不按模型结果选择。

| 样本 | 检查数 | 独立病人数 | 阳性检查数 |
| --- | ---: | ---: | ---: |
| 包含目标医生的训练组 | {sample['seen']['studies']} | {sample['seen']['patients']} | {sample['seen']['positive_studies']} |
| 排除目标医生的训练组 | {sample['unseen']['studies']} | {sample['unseen']['patients']} | {sample['unseen']['positive_studies']} |
| 共享验证集 | {sample['valid']['studies']} | {sample['valid']['patients']} | {sample['valid']['positive_studies']} |
| 同一目标医生测试集 | {sample['test']['studies']} | {sample['test']['patients']} | {sample['test']['positive_studies']} |

两组训练集按标签和体位精确匹配，各有 PA {sample['seen']['views'].get('PA',0)} / AP {sample['seen']['views'].get('AP',0)}。包含组保留目标医生 {sample['seen_target_studies']} 次检查，排除组为 0。验证集排除目标医生病例。训练独立病人数并未强制相等。

为缩短首次网络数据准备时间，训练前将两组训练预算收敛到每组 256 次、验证预算 128 次，按固定种子分层抽样。尚未查看测试结果时已确定该调整，完整规模清单保留供后续扩展。

## 4. 模型与运行（约 50 秒）

ImageNet 预训练 ResNet18，224×224 输入，全部参数微调；AdamW，学习率 1e-4，batch 32，AMP，各训练 5 轮。按共同验证集 BCE 选择模型：包含组第 {best['seen']} 轮，排除组第 {best['unseen']} 轮。

## 5. 真实结果（约 100 秒）

{chr(10).join(table)}

AUROC / AUPRC 越高越好，Brier 越低越好。两模型在完全相同的测试检查上比较，按病人配对 bootstrap 1,000 次；有效重复 {result['paired_bootstrap']['valid_replicates']} 次。该区间不覆盖重新训练的随机性。

{finding}

参考：使用训练阳性率作为固定概率的基线，在该测试集上 AUROC 为 {result['constant_baseline']['AUROC']:.3f}，AUPRC 为 {result['constant_baseline']['AUPRC']:.3f}，Brier 为 {result['constant_baseline']['Brier']:.3f}。

## 6. 限制与期末方向（约 70 秒）

当前已完成一次真实图像配对实验，不声称完成全项目。仅一个目标医生、一个随机种子；自动规则和报告抽取仍可能带来标签偏差；病情、体位和患者构成等因素可能影响差异。

期末再扩展多医生和多种子，改进标签复核，检查其他分辨率与潜在混杂。无差异也是可报告的初版结果，不为获得更好结果改选医生或测试集。

## 依据

本机 `{a.work.as_posix()}/results.json`、`summary.json`、`run_config.json` 与对应预测/权重；原始明细不入库。图表在同目录 `results.png`。本页是汇报内容与讲述顺序，不是已完成的 PPT 文件。
'''
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    a.dest.write_text(text, encoding='utf-8')
    print('Aggregate midterm notes saved:', a.dest)


if __name__ == '__main__':
    main()
