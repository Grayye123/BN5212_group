"""比对两个独立标注器（CheXbert 与第二标注器）在肺不张四态上的结果，导出分歧清单。

第二标注器的结果由 claude_labels.csv 提供（仅覆盖判断集 review_judge.jsonl 中的报告）；
判断集之外的报告，其描述性段落不含 atelecta* / collaps* / volume loss，机械判为「未提及」。

输出：
    <out>/label_comparison.csv    逐检查两方标签
    <out>/disagreements.jsonl     分歧条目（含上下文），供人工裁决
    <out>/agreement_sample.jsonl  从一致条目中随机抽样，供人工复核「两边都错」的情况

用法：
    python code/audit/compare_labels.py --out out/audit --sample 50 --seed 20260920
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

NAME = {'1': '阳性', '0': '明确阴性', '-1': '不确定', '': '未提及'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/audit')
    ap.add_argument('--sample', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20260920)
    a = ap.parse_args()

    judged = {}
    with open(os.path.join(a.out, 'claude_labels.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            judged[r['study_id']] = r['claude_atelectasis']

    ctx = {}
    with open(os.path.join(a.out, 'review_judge.jsonl'), encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            ctx[r['study_id']] = r['context']

    rows = []
    with open(os.path.join(a.out, 'chexbert_labels.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            sid = r['study_id']
            rows.append({'study_id': sid, 'subject_id': r['subject_id'],
                         'chexbert': r['Atelectasis'],
                         'second': judged.get(sid, ''),
                         'judged': sid in judged})

    agree = [r for r in rows if r['chexbert'] == r['second']]
    disagree = [r for r in rows if r['chexbert'] != r['second']]

    print('检查总数 %d · 一致 %d (%.1f%%) · 分歧 %d (%.1f%%)'
          % (len(rows), len(agree), 100 * len(agree) / len(rows),
             len(disagree), 100 * len(disagree) / len(rows)))
    print()
    print('分布对照：')
    cb, sc = Counter(r['chexbert'] for r in rows), Counter(r['second'] for r in rows)
    print('%-10s %10s %10s' % ('', 'CheXbert', '第二标注器'))
    for k in ('1', '0', '-1', ''):
        print('%-10s %10d %10d' % (NAME[k], cb[k], sc[k]))
    print()
    print('混淆矩阵（行=CheXbert，列=第二标注器）：')
    print('%-10s %8s %8s %8s %8s' % ('', '阳性', '明确阴性', '不确定', '未提及'))
    m = Counter((r['chexbert'], r['second']) for r in rows)
    for k in ('1', '0', '-1', ''):
        print('%-10s %8d %8d %8d %8d' % (NAME[k], m[(k, '1')], m[(k, '0')], m[(k, '-1')], m[(k, '')]))

    dest = os.path.join(a.out, 'label_comparison.csv')
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['study_id', 'subject_id', 'chexbert', 'second', 'judged'])
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(a.out, 'disagreements.jsonl'), 'w', encoding='utf-8') as fh:
        for r in disagree:
            fh.write(json.dumps({'study_id': r['study_id'],
                                 'chexbert': NAME[r['chexbert']],
                                 'second': NAME[r['second']],
                                 'context': ctx.get(r['study_id'], '(判断集之外：描述性段落未出现相关词)'),
                                 'human': ''}, ensure_ascii=False) + '\n')

    rnd = random.Random(a.seed)
    pool = [r for r in agree if r['judged']]
    pick = rnd.sample(pool, min(a.sample, len(pool)))
    with open(os.path.join(a.out, 'agreement_sample.jsonl'), 'w', encoding='utf-8') as fh:
        for r in pick:
            fh.write(json.dumps({'study_id': r['study_id'], 'both': NAME[r['chexbert']],
                                 'context': ctx.get(r['study_id'], ''), 'human': ''},
                                ensure_ascii=False) + '\n')
    print()
    print('分歧 %d 条 -> disagreements.jsonl（待人工裁决）' % len(disagree))
    print('一致抽样 %d 条 -> agreement_sample.jsonl（待人工复核，抓「两边都错」）' % len(pick))
    return 0


if __name__ == '__main__':
    sys.exit(main())
