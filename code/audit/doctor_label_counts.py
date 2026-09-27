"""合并子集清单、CheXbert 标签与体位信息，统计每位主治医生的独立阳性 / 阴性病人数。

输入（均在本机 out/audit/，不入库）：
    subset_manifest.csv   逐图：subject_id, study_id, dicom_id, 三个医生编号
    chexbert_labels.csv   逐检查：14 个观察项的四态标签
    dicom_headers.csv     逐图：ViewPosition 等

标签规则（PROJECT.md 草案，阶段 2 尚未冻结）：
    阳性 1 -> 阳性；明确阴性 0 与未提及 '' -> 阴性；不确定 -1 -> 排除该检查。
病人归类：只要有一次阳性检查即记为阳性病人，否则记为阴性病人。

可选 --frontal-only：只保留至少有一张正面片（PA / AP）的检查。

用法：
    python code/audit/doctor_label_counts.py --out out/audit
输出 <out>/doctor_label_counts.csv
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


def load(path):
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/audit')
    ap.add_argument('--frontal-only', action='store_true')
    a = ap.parse_args()

    manifest = load(os.path.join(a.out, 'subset_manifest.csv'))
    labels = {r['study_id']: r['Atelectasis'] for r in load(os.path.join(a.out, 'chexbert_labels.csv'))}
    heads = load(os.path.join(a.out, 'dicom_headers.csv'))

    frontal = defaultdict(bool)
    for r in heads:
        if r['ViewPosition'] in ('PA', 'AP'):
            frontal[r['study_id']] = True

    # 每次检查 -> 病人、主治医生
    study = {}
    for r in manifest:
        study[r['study_id']] = (r['subject_id'], r['attending_provider_id'])

    kept, dropped_unc, dropped_view, missing = 0, 0, 0, 0
    doc = defaultdict(lambda: {'pos_pat': set(), 'neg_pat': set(), 'pos_std': 0, 'neg_std': 0})
    pat_state = defaultdict(dict)  # (doctor, patient) -> 是否见过阳性

    for sid, (pid, att) in study.items():
        if a.frontal_only and not frontal[sid]:
            dropped_view += 1
            continue
        lab = labels.get(sid)
        if lab is None:
            missing += 1
            continue
        if lab == '-1':
            dropped_unc += 1
            continue
        kept += 1
        positive = (lab == '1')
        d = doc[att or '(缺失)']
        if positive:
            d['pos_std'] += 1
        else:
            d['neg_std'] += 1
        prev = pat_state[att or '(缺失)'].get(pid, False)
        pat_state[att or '(缺失)'][pid] = prev or positive

    for att, pats in pat_state.items():
        for pid, positive in pats.items():
            doc[att]['pos_pat' if positive else 'neg_pat'].add(pid)

    print('纳入检查 %d · 因「不确定」排除 %d · 因无正面片排除 %d · 无标签 %d'
          % (kept, dropped_unc, dropped_view, missing))

    rows = []
    for att, v in doc.items():
        rows.append({
            'attending_provider_id': att,
            'studies': v['pos_std'] + v['neg_std'],
            'pos_studies': v['pos_std'],
            'neg_studies': v['neg_std'],
            'patients': len(v['pos_pat']) + len(v['neg_pat']),
            'pos_patients': len(v['pos_pat']),
            'neg_patients': len(v['neg_pat']),
        })
    rows.sort(key=lambda r: -r['patients'])
    for i, r in enumerate(rows, 1):
        r['rank'] = 'D%02d' % i

    dest = os.path.join(a.out, 'doctor_label_counts.csv')
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['rank', 'attending_provider_id', 'studies',
                                           'pos_studies', 'neg_studies', 'patients',
                                           'pos_patients', 'neg_patients'])
        w.writeheader()
        w.writerows(rows)

    print()
    print('%-5s %8s %8s %8s %9s %9s %8s' % ('排名', '检查', '阳性检查', '阴性检查', '病人', '阳性病人', '阴性病人'))
    for r in rows[:15]:
        print('%-5s %8d %8d %8d %9d %9d %8d' % (r['rank'], r['studies'], r['pos_studies'],
                                                r['neg_studies'], r['patients'],
                                                r['pos_patients'], r['neg_patients']))
    print()
    for th in (30, 20, 15, 10, 5):
        ok = [r for r in rows if r['pos_patients'] >= th and r['neg_patients'] >= th]
        print('阳性病人 >= %2d 且 阴性病人 >= %2d : %2d 位医生' % (th, th, len(ok)))
    print('->', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
