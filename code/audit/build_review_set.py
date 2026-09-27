"""抽出需要人工/第二标注器判断的报告片段，用于与 CheXbert 交叉比对（只读 ZIP）。

分两类：
  auto  描述性段落中完全没有 atelecta* 的报告 —— 机械判为「未提及」，无需判断
  judge 有提及的报告 —— 抽出含提及的句子及前后各一句作为上下文，供独立标注

**输出不含 CheXbert 的标签**，以保证第二次标注是盲标。比对在 compare_labels 阶段再做。

用法：
    python code/audit/build_review_set.py --out out/audit
输出：
    <out>/review_judge.jsonl   需判断的条目：study_id, context
    <out>/review_auto.csv      机械判为未提及的 study_id
"""
import argparse
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_labels import TXT, findings_text, sentences  # noqa: E402

# 判断集的入选词比 report_labels.TERM 宽：肺不张在报告里也常写成 collapse 或 volume loss，
# 只搜 atelecta* 会漏掉这些表述，交叉比对就不完整。该词表覆盖 CheXbert 非空标签的 99.8%。
TERM = re.compile(r'atelecta\w*|collaps\w*|volume loss', re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip')
    ap.add_argument('--out', default='out/audit')
    ap.add_argument('--context', type=int, default=1, help='提及句前后各取几句作为上下文')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    zf = zipfile.ZipFile(a.zip)
    names = [n for n in zf.namelist() if TXT.search(n)]

    judge, auto = [], []
    total_chars = 0
    for n in names:
        m = TXT.search(n)
        sid = m.group(2)
        body = findings_text(zf.read(n).decode('utf-8', 'replace'))
        if not TERM.search(body):
            auto.append(sid)
            continue
        sents = sentences(body)
        keep = set()
        for i, s in enumerate(sents):
            if TERM.search(s):
                for j in range(max(0, i - a.context), min(len(sents), i + a.context + 1)):
                    keep.add(j)
        ctx = ' '.join(sents[i].strip() for i in sorted(keep))
        ctx = re.sub(r'\s+', ' ', ctx).strip()
        total_chars += len(ctx)
        judge.append({'study_id': sid, 'context': ctx})

    with open(os.path.join(a.out, 'review_judge.jsonl'), 'w', encoding='utf-8') as fh:
        for r in judge:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(a.out, 'review_auto.csv'), 'w', encoding='utf-8') as fh:
        fh.write('study_id\n')
        for sid in auto:
            fh.write(sid + '\n')

    print('机械判「未提及」 %d 份' % len(auto))
    print('需判断           %d 份' % len(judge))
    print('待读文本合计     %d 字符，平均 %.0f 字符/份' % (total_chars, total_chars / max(1, len(judge))))
    print('->', os.path.join(a.out, 'review_judge.jsonl'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
