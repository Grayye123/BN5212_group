"""从 ZIP 内只读取 DICOM 文件头，补全体位等元数据（不解压、不读像素）。

用法：
    python code/audit/dicom_headers.py --zip Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip --out out/audit

输出 <out>/dicom_headers.csv：每张影像一行，含 subject_id / study_id / dicom_id 及
ViewPosition、PatientOrientation、Rows、Columns、PhotometricInterpretation 等。
只 read() 每个成员的前若干字节，stop_before_pixels 保证不解码像素。
"""
import argparse
import csv
import io
import os
import re
import sys
import time
import zipfile

import pydicom

DCM = re.compile(r'files/p\d\d/p(\d+)/s(\d+)/([0-9a-f-]+)\.dcm$')
FIELDS = ['ViewPosition', 'ViewCodeSequence', 'PatientOrientation', 'SeriesDescription',
          'StudyDate', 'ProcedureCodeSequence', 'PerformedProcedureStepDescription',
          'Rows', 'Columns', 'PhotometricInterpretation', 'BitsStored',
          'Modality', 'Manufacturer', 'BodyPartExamined']
HEAD_BYTES = 16384


def val(ds, name):
    if name not in ds:
        return ''
    v = ds[name].value
    if name == 'ViewCodeSequence':
        try:
            return str(v[0].CodeMeaning)
        except Exception:
            return ''
    if name == 'ProcedureCodeSequence':
        try:
            return str(v[0].CodeMeaning)
        except Exception:
            return ''
    if isinstance(v, (list, tuple)) or v.__class__.__name__ == 'MultiValue':
        return '|'.join(str(x) for x in v)
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip')
    ap.add_argument('--out', default='out/audit')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 张，0 表示全部')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    zf = zipfile.ZipFile(a.zip)
    names = [n for n in zf.namelist() if DCM.search(n)]
    if a.limit:
        names = names[:a.limit]
    print('待读取 %d 张' % len(names), flush=True)

    dest = os.path.join(a.out, 'dicom_headers.csv')
    t0 = time.time()
    failed = 0
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'study_id', 'dicom_id'] + FIELDS + ['head_bytes', 'error'])
        for i, n in enumerate(names, 1):
            m = DCM.search(n)
            row = [m.group(1), m.group(2), m.group(3)]
            err = ''
            nb = HEAD_BYTES
            ds = None
            while nb <= 1 << 21:
                try:
                    with zf.open(n) as src:
                        buf = src.read(nb)
                    ds = pydicom.dcmread(io.BytesIO(buf), stop_before_pixels=True, force=True)
                    if 'ViewPosition' in ds or 'Rows' in ds:
                        break
                except Exception as exc:
                    err = type(exc).__name__
                    ds = None
                nb *= 4
            if ds is None:
                failed += 1
                w.writerow(row + [''] * len(FIELDS) + [nb, err or 'unreadable'])
            else:
                w.writerow(row + [val(ds, f) for f in FIELDS] + [nb, ''])
            if i % 500 == 0:
                el = time.time() - t0
                print('  %d/%d  %.0f s  预计剩余 %.0f s' % (i, len(names), el, el / i * (len(names) - i)),
                      flush=True)
    print('完成 %d 张, 失败 %d, 用时 %.0f s -> %s' % (len(names), failed, time.time() - t0, dest))


if __name__ == '__main__':
    sys.exit(main())
