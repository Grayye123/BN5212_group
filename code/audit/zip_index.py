"""只读盘点 MIMIC 目录下各 ZIP 的成员清单（不解压、不修改任何原始文件）。

两种用法：
    # 在服务器上直接跑
    python3 zip_index.py --base /path/to/authorized/MIMIC

    # 在已挂载服务器目录的本机上跑（只读 ZIP 目录页，输出写本地）
    python zip_index.py --base Z:\ --out out/audit

输出：
    zip_index_<name>.json  每个 ZIP 的成员清单（文件名、原始大小、压缩大小、CRC、时间）
    zip_summary.json       各 ZIP 的汇总（成员数、按扩展名计数、总字节数）
成员数超过 FULL_LIST_LIMIT 的 ZIP 只写汇总和前 200 条样例。
"""
import argparse
import json
import os
import sys
import time
import zipfile
from collections import Counter

ZIPS = [
    'MIMIC-CXR/BN5212_MIMIC-CXR.zip',
    'MIMIC-CXR/dataset.zip',
    'MIMIC-CXR/data_subset1.zip',
    'MIMIC-CXR/data_subset2.zip',
    'MIMIC-CXR/data_subset3.zip',
    'MIMIC-CXR/data_subset4.zip',
    'MIMIC-CXR/mimic-cxr-reports.zip',
    'MIMIC-IV/MIMIC_IV.zip',
    'MIMIC-IV-NOTE/physionet.org.zip',
]
FULL_LIST_LIMIT = 50000


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help='MIMIC 数据根目录（服务器路径或本机挂载点）')
    ap.add_argument('--out', default=None, help='输出目录，默认 <base>/audit')
    args = ap.parse_args()
    base = args.base
    out = args.out or os.path.join(base, 'audit')
    os.makedirs(out, exist_ok=True)
    log('base=%s out=%s' % (base, out))

    summary = {}
    for rel in ZIPS:
        path = os.path.join(base, rel.replace('/', os.sep))
        name = os.path.basename(rel)
        if not os.path.exists(path):
            log('缺失: ' + rel)
            summary[name] = {'missing': True}
            continue
        log('读取目录: ' + rel)
        t0 = time.time()
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
        ext = Counter()
        dirs = 0
        total = 0
        members = []
        for i in infos:
            if i.is_dir():
                dirs += 1
                continue
            total += i.file_size
            e = os.path.splitext(i.filename)[1].lower() or '(none)'
            ext[e] += 1
            members.append({
                'name': i.filename,
                'size': i.file_size,
                'csize': i.compress_size,
                'crc': i.CRC,
                'date': '%04d-%02d-%02d %02d:%02d:%02d' % i.date_time,
            })
        top = Counter(m['name'].split('/')[0] for m in members)
        summary[name] = {
            'path': rel,
            'zip_bytes': os.path.getsize(path),
            'members_total': len(infos),
            'files': len(members),
            'dirs': dirs,
            'uncompressed_bytes': total,
            'by_ext': dict(ext.most_common()),
            'top_level': dict(top.most_common(20)),
            'seconds': round(time.time() - t0, 1),
        }
        dest = os.path.join(out, 'zip_index_' + name.replace('.zip', '') + '.json')
        if len(members) > FULL_LIST_LIMIT:
            payload = {'note': 'members exceed limit; sample only', 'sample': members[:200]}
        else:
            payload = members
        with open(dest, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False)
        log('  %d 个成员, %.1f s -> %s' % (len(infos), time.time() - t0, dest))
    with open(os.path.join(out, 'zip_summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log('完成: ' + os.path.join(out, 'zip_summary.json'))


if __name__ == '__main__':
    sys.exit(main())
