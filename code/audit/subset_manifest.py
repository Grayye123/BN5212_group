"""从 ZIP 成员清单构建课程子集清单，并与官方索引表、医生表关联（只读）。

输入：out/audit/zip_index_BN5212_MIMIC-CXR.json（含完整 MIMIC 路径）
      <base>/MIMIC-CXR/cxr-record-list.csv.gz、cxr-study-list.csv.gz、cxr-provider-list.csv.gz
输出：<out>/subset_manifest.csv   逐图：subject_id, study_id, dicom_id, 三个医生编号
      <out>/provider_counts.csv   每位主治医生的检查数、病人数、影像数
不读取像素、不解压、不写入服务器。
"""
import argparse
import csv
import gzip
import json
import os
import re
from collections import defaultdict

DCM = re.compile(r'files/p\d\d/p(\d+)/s(\d+)/([0-9a-f-]+)\.dcm$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='Z:/')
    ap.add_argument('--index', default='out/audit/zip_index_BN5212_MIMIC-CXR.json')
    ap.add_argument('--out', default='out/audit')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    members = json.load(open(a.index, encoding='utf-8'))
    rows = []
    for m in members:
        hit = DCM.search(m['name'])
        if hit:
            rows.append({'subject_id': hit.group(1), 'study_id': hit.group(2),
                         'dicom_id': hit.group(3), 'csize': m['csize'], 'size': m['size']})
    print('子集影像 dcm=%d 检查=%d 病人=%d' % (
        len(rows), len({r['study_id'] for r in rows}), len({r['subject_id'] for r in rows})))

    def gz(name):
        return gzip.open(os.path.join(a.base, 'MIMIC-CXR', name), 'rt', encoding='utf-8')

    # 官方全量索引核对
    with gz('cxr-record-list.csv.gz') as fh:
        official = {(r['subject_id'], r['study_id'], r['dicom_id']) for r in csv.DictReader(fh)}
    miss = [r for r in rows if (r['subject_id'], r['study_id'], r['dicom_id']) not in official]
    print('不在官方 cxr-record-list 中的子集影像:', len(miss))

    # 医生表
    with gz('cxr-provider-list.csv.gz') as fh:
        prov = {r['study_id']: r for r in csv.DictReader(fh)}
    hit = sum(1 for r in rows if r['study_id'] in prov)
    print('能关联到医生表的影像: %d / %d' % (hit, len(rows)))

    for r in rows:
        p = prov.get(r['study_id'], {})
        r['ordering_provider_id'] = p.get('ordering_provider_id', '')
        r['attending_provider_id'] = p.get('attending_provider_id', '')
        r['resident_provider_id'] = p.get('resident_provider_id', '')

    dest = os.path.join(a.out, 'subset_manifest.csv')
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('->', dest)

    # 每位主治医生
    agg = defaultdict(lambda: {'studies': set(), 'patients': set(), 'images': 0})
    for r in rows:
        k = r['attending_provider_id'] or '(缺失)'
        agg[k]['studies'].add(r['study_id'])
        agg[k]['patients'].add(r['subject_id'])
        agg[k]['images'] += 1
    dest = os.path.join(a.out, 'provider_counts.csv')
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['attending_provider_id', 'studies', 'patients', 'images'])
        for k, v in sorted(agg.items(), key=lambda kv: -len(kv[1]['studies'])):
            w.writerow([k, len(v['studies']), len(v['patients']), v['images']])
    print('主治医生数 =', len(agg), '->', dest)
    for k, v in sorted(agg.items(), key=lambda kv: -len(kv[1]['studies']))[:40]:
        print('   %-8s 检查%4d 病人%4d 影像%4d' % (k, len(v['studies']), len(v['patients']), v['images']))
    # 住院医
    res = {r['resident_provider_id'] for r in rows if r['resident_provider_id']}
    nres = len({r['study_id'] for r in rows if r['resident_provider_id']})
    print('住院医: %d 位, 覆盖 %d 次检查' % (len(res), nres))


if __name__ == '__main__':
    main()
