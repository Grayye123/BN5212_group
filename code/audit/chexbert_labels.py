import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""用 CheXbert 对课程子集的 3,351 份报告重新标注（只读 ZIP，不解压、不修改原始数据）。

标注器：f1chexbert 封装的 CheXbert（Smit et al., 2020），权重来自 Stanford AIMI 的
HuggingFace 仓库 StanfordAIMI/RRG_scorers。mode="classification" 输出 CheXpert 四态：
    ''  未提及 · 1 阳性 · 0 明确阴性 · -1 不确定

送入模型的文本沿用 report_labels.py 的段落过滤：只保留 FINDINGS / IMPRESSION 等描述性
段落，去掉 INDICATION / HISTORY / COMPARISON / TECHNIQUE 等非发现段落。

用法：
    python code/audit/chexbert_labels.py --zip Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip --out out/audit
输出 <out>/chexbert_labels.csv：subject_id, study_id, 14 个观察项的四态标签
"""
import argparse
import csv
import os
import re
import sys
import time
import zipfile

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from report_labels import TXT, findings_text  # noqa: E402  复用段落过滤，保证与自写规则同口径




def patch_tokenizer():
    """f1chexbert 0.0.2 调用 tokenizer.encode_plus(<token 列表>)，该接口在 transformers 5.x
    已移除。这里用等价的现代接口重写其 tokenize：token -> id，再补 [CLS]/[SEP]，
    超过 512 时按原实现截断为 511 + [SEP]。行为与原版一致。"""
    import f1chexbert.f1chexbert as fx

    def tokenize(impressions, tokenizer):
        imp = impressions.str.replace(r'\s+', ' ', regex=True).str.strip()
        out = []
        for i in range(imp.shape[0]):
            toks = tokenizer.tokenize(imp.iloc[i])
            if toks:
                ids = ([tokenizer.cls_token_id]
                       + tokenizer.convert_tokens_to_ids(toks)
                       + [tokenizer.sep_token_id])
                if len(ids) > 512:
                    ids = ids[:511] + [tokenizer.sep_token_id]
                out.append(ids)
            else:
                out.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
        return out

    fx.tokenize = tokenize


def ensure_checkpoint():
    """f1chexbert 依赖 hf_hub_download(force_filename=...)，该参数在新版 huggingface_hub
    中已失效，权重会落到 HF 缓存的 snapshots/ 下而不是它期望的 <cache>/chexbert.pth。
    这里把已下载的文件移到位；没下载过则交给 f1chexbert 自己下。"""
    from platformdirs import user_cache_dir
    cache = user_cache_dir('chexbert')
    want = os.path.join(cache, 'chexbert.pth')
    if os.path.exists(want):
        return want
    for root, _dirs, files in os.walk(cache):
        for f in files:
            if f == 'chexbert.pth':
                src = os.path.join(root, f)
                if os.path.abspath(src) != os.path.abspath(want):
                    print('搬运权重 %s -> %s' % (src, want), flush=True)
                    os.replace(src, want)
                return want
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip')
    ap.add_argument('--out', default='out/audit')
    ap.add_argument('--device', default=None, help='cuda 或 cpu，默认自动')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    from f1chexbert import F1CheXbert
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('device =', dev, '| torch', torch.__version__, flush=True)
    print('加载 CheXbert（首次运行会下载约 1.3 GB 权重）...', flush=True)
    ensure_checkpoint()
    t0 = time.time()
    clf = F1CheXbert(device=dev)
    patch_tokenizer()
    print('加载完成 %.0f s' % (time.time() - t0), flush=True)

    zf = zipfile.ZipFile(a.zip)
    names = [n for n in zf.namelist() if TXT.search(n)]
    if a.limit:
        names = names[:a.limit]
    print('报告数', len(names), flush=True)

    cols = clf.target_names
    dest = os.path.join(a.out, 'chexbert_labels.csv')
    t0 = time.time()
    counts = {'': 0, '1': 0, '0': 0, '-1': 0}
    ate = cols.index('Atelectasis')
    with open(dest, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['subject_id', 'study_id'] + cols)
        for i, n in enumerate(names, 1):
            m = TXT.search(n)
            text = findings_text(zf.read(n).decode('utf-8', 'replace'))
            text = re.sub(r'\s+', ' ', text).strip()
            lab = clf.get_label(text, mode='classification')
            counts[str(lab[ate])] += 1
            w.writerow([m.group(1), m.group(2)] + [str(x) for x in lab])
            if i % 200 == 0:
                el = time.time() - t0
                print('  %d/%d  %.0f s  预计剩余 %.0f s' % (i, len(names), el, el / i * (len(names) - i)),
                      flush=True)
    print('\n=== Atelectasis 四态 ===')
    print('阳性(1)      %5d' % counts['1'])
    print('明确阴性(0)  %5d' % counts['0'])
    print('不确定(-1)   %5d' % counts['-1'])
    print('未提及       %5d' % counts[''])
    print('提及合计     %5d' % (counts['1'] + counts['0'] + counts['-1']))
    print('旧记录参考: 832 / 13 / 213 / 2293（提及合计 1058）')
    print('->', dest)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
