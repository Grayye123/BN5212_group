"""从报告文本重新抽取肺不张（Atelectasis）四态标签（只读 ZIP，不解压、不修改原始数据）。

规则参照 CheXpert/NegBio 的思路，但为本项目独立实现，逐条可查：
  1. 只在 FINDINGS / IMPRESSION 等描述性段落中找提及，忽略 INDICATION / HISTORY /
     COMPARISON / TECHNIQUE / EXAMINATION 等非发现段落。
  2. 提及 atelecta*（atelectasis / atelectatic / atelectases）才进入判定，否则「未提及」。
  3. 句中命中否定模式 -> 明确阴性(0)；命中不确定模式 -> 不确定(-1)；否则阳性(1)。
  4. 一份报告有多处提及时取最强证据：阳性 > 不确定 > 阴性。

用法：
    python code/audit/report_labels.py --zip Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip --out out/audit
输出 <out>/report_labels.csv：subject_id, study_id, atelectasis, evidence
"""
import argparse
import csv
import os
import re
import sys
import zipfile

TXT = re.compile(r'files/p\d\d/p(\d+)/s(\d+)\.txt$')
TERM = re.compile(r'atelecta\w*', re.I)

# 非发现段落：这些标题之后到下一个标题之间的内容不参与判定
SKIP_HEADS = re.compile(
    r'^\s*(INDICATION|HISTORY|CLINICAL HISTORY|COMPARISON|COMPARISONS|TECHNIQUE|'
    r'EXAMINATION|REASON FOR EXAM|WET READ|NOTIFICATION|RECOMMENDATION\(S\))\s*:', re.I)
ANY_HEAD = re.compile(r'^\s*[A-Z][A-Z \'/()_-]{2,40}:', re.M)

NEG = re.compile(
    r'(no|without|absent|free of|resolution of|resolved|clear of|negative for)', re.I)
NEG_POST = re.compile(
    r'(is|are|has|have|was|were)?\s*(not|no longer)\s+\w{0,12}\s*'
    r'(seen|identified|present|evident|visualized|noted)', re.I)
UNC = re.compile(
    r'(may|might|could|possible|possibly|probable|probably|likely|suspect\w*|'
    r'question\w*|concerning for|worrisome for|cannot (be )?exclude[d]?|not excluded|'
    r'difficult to exclude|equivocal|indeterminate)', re.I)
# 「atelectasis or pneumonia」「consolidation versus atelectasis」这类并列鉴别也算不确定
UNC_ALT = re.compile(r'atelecta\w*\s+(or|versus|vs\.?)\s|\s(or|versus|vs\.?)\s+atelecta', re.I)
# 断开否定作用域的连接词：否定词与 atelecta* 之间出现这些词，则否定不覆盖该提及
BREAK = re.compile(r'(with|but|however|although|though|except|aside from|other than)', re.I)


def findings_text(report):
    """去掉非发现段落，返回参与判定的文本。"""
    lines = report.splitlines()
    keep, skipping = [], False
    for ln in lines:
        if ANY_HEAD.match(ln):
            skipping = bool(SKIP_HEADS.match(ln))
        if not skipping:
            keep.append(ln)
    return '\n'.join(keep)


def sentences(text):
    flat = re.sub(r'\s+', ' ', text)
    return re.split(r'(?<=[.;:])\s+', flat)


def label_one(report):
    text = findings_text(report)
    if not TERM.search(text):
        return '', ''
    best, evid = None, ''
    order = {1: 3, -1: 2, 0: 1}
    for s in sentences(text):
        for m in TERM.finditer(s):
            before, after = s[:m.start()], s[m.end():]
            # 否定只在「否定词 -> 提及」之间没有连接词时成立
            neg = False
            for nm in NEG.finditer(before):
                if not BREAK.search(before[nm.end():]):
                    neg = True
                    break
            if not neg and NEG_POST.match(after.lstrip()[:40]):
                neg = True
            if neg:
                v = 0
            elif UNC.search(s) or UNC_ALT.search(s):
                v = -1
            else:
                v = 1
            if best is None or order[v] > order[best]:
                best, evid = v, s.strip()[:300]
    return ('' if best is None else str(best)), evid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip')
    ap.add_argument('--out', default='out/audit')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    zf = zipfile.ZipFile(a.zip)
    names = [n for n in zf.namelist() if TXT.search(n)]
    print('报告数', len(names))

    counts = {'1': 0, '0': 0, '-1': 0, '': 0}
    dest = os.path.join(a.out, 'report_labels.csv')
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'study_id', 'atelectasis', 'evidence'])
        for n in names:
            m = TXT.search(n)
            rep = zf.read(n).decode('utf-8', 'replace')
            lab, ev = label_one(rep)
            counts[lab] += 1
            w.writerow([m.group(1), m.group(2), lab, ev])
    print('阳性(1)      %5d' % counts['1'])
    print('明确阴性(0)  %5d' % counts['0'])
    print('不确定(-1)   %5d' % counts['-1'])
    print('未提及       %5d' % counts[''])
    print('提及合计     %5d' % (counts['1'] + counts['0'] + counts['-1']))
    print('->', dest)


if __name__ == '__main__':
    sys.exit(main())
