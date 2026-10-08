"""One paired real-image midterm experiment. Private files stay under --work."""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import time
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import pydicom
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

SEED = 20260927


def load(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def image_from_dicom(data):
    ds = pydicom.dcmread(io.BytesIO(data))
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise ValueError('Unsupported or invalid pixel array')
    pixels = pixels * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        pixels = -pixels
    elif ds.PhotometricInterpretation != 'MONOCHROME2':
        raise ValueError('Unsupported photometric interpretation')
    low, high = np.percentile(pixels, [0.5, 99.5])
    if high <= low:
        raise ValueError('Constant image')
    pixels = (np.clip((pixels-low)/(high-low), 0, 1)*255).astype(np.uint8)
    im = Image.fromarray(pixels)
    im.thumbnail((224, 224), Image.Resampling.BILINEAR)
    canvas = Image.new('L', (224, 224), 0)
    canvas.paste(im, ((224-im.width)//2, (224-im.height)//2))
    return np.asarray(canvas)


def prepare_cache(work, archive, rows, cache_path=None):
    cache = cache_path or work / 'pixels224'
    cache.mkdir(exist_ok=True)
    unique = {r['dicom_id']: r for r in rows}
    pending = [k for k in sorted(unique) if not (cache / (k+'.npy')).exists()]
    if not pending:
        return cache
    if not archive.is_file():
        raise FileNotFoundError('DICOM archive unavailable. Restore data mount or pass --zip.')
    start = time.time()
    with zipfile.ZipFile(archive) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.lower().endswith('.dcm')}
        if not set(pending) <= set(members):
            raise ValueError('Selected images missing from archive')
        # Preserve ZIP storage locality on a remote spinning disk/SSHFS mount.
        # Dataset/loader ordering is unchanged; this only orders cache creation.
        pending.sort(key=lambda key: z.getinfo(members[key]).header_offset)
    local = threading.local()
    handles = []
    def initialize_reader():
        local.archive = zipfile.ZipFile(archive)
        handles.append(local.archive)
    def convert(key):
        # Read original member into memory; persist only a derived thumbnail.
        arr = image_from_dicom(local.archive.read(members[key]))
        dest = cache / (key+'.npy')
        with open(dest.with_suffix('.partial'), 'wb') as f:
            np.save(f, arr, allow_pickle=False)
        dest.with_suffix('.partial').replace(dest)
        return key
    try:
        with ThreadPoolExecutor(max_workers=1, initializer=initialize_reader) as executor:
            for i, _ in enumerate(executor.map(convert, pending), 1):
                if i % 25 == 0 or i == len(pending):
                    print(json.dumps(dict(stage='pixels', done=i, total=len(pending), seconds=round(time.time()-start))), flush=True)
    finally:
        for handle in handles:
            handle.close()
    return cache


class Images(Dataset):
    def __init__(self, rows, cache):
        self.rows = rows
        self.cache = cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        a = np.load(self.cache/(r['dicom_id']+'.npy'), allow_pickle=False)
        if a.shape != (224, 224) or a.dtype != np.uint8:
            raise ValueError('Invalid cached image')
        x = torch.from_numpy(a.copy()).float().unsqueeze(0).repeat(3, 1, 1)/255
        x = (x-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
        return x, torch.tensor(float(r['y']))


def predict(model, loader, device):
    model.eval()
    output = []
    with torch.no_grad():
        for x, y in loader:
            output.extend(torch.sigmoid(model(x.to(device)).flatten()).cpu().numpy().tolist())
    return np.asarray(output)


def metrics(y, p):
    return dict(AUROC=float(roc_auc_score(y, p)), AUPRC=float(average_precision_score(y, p)),
                Brier=float(brier_score_loss(y, p)))


def bootstrap(rows, y, a, b, count=1000):
    rng = np.random.default_rng(SEED)
    ids = sorted({r['subject_id'] for r in rows})
    groups = {pid: np.asarray([i for i, r in enumerate(rows) if r['subject_id'] == pid]) for pid in ids}
    values = {k: [] for k in ('AUROC', 'AUPRC', 'Brier')}
    for _ in range(count):
        idx = np.concatenate([groups[pid] for pid in rng.choice(ids, len(ids), replace=True)])
        if len(set(y[idx])) < 2:
            continue
        ma, mb = metrics(y[idx], a[idx]), metrics(y[idx], b[idx])
        for k in values:
            values[k].append(ma[k]-mb[k])
    return dict(valid_replicates=len(values['AUROC']), requested_replicates=count,
                intervals={k: np.percentile(v, [2.5, 97.5]).tolist() if v else None for k, v in values.items()})


def train_one(name, rows, valid, test, cache, work, epochs, weights):
    seed_all()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    loaders = {key: DataLoader(Images(group, cache), batch_size=32, shuffle=key == 'train', num_workers=0)
               for key, group in [('train', rows), ('valid', valid), ('test', test)]}
    best = float('inf')
    history = []
    yvalid = np.array([int(r['y']) for r in valid])
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(x).flatten(), y)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()) * len(y))
        pv = predict(model, loaders['valid'], device)
        vloss = float(-(yvalid*np.log(np.clip(pv, 1e-7, 1-1e-7))+(1-yvalid)*np.log(np.clip(1-pv, 1e-7, 1-1e-7))).mean())
        row = dict(model=name, epoch=epoch, train_loss=sum(losses)/len(rows), valid_loss=vloss, **metrics(yvalid, pv))
        history.append(row)
        print(json.dumps(row), flush=True)
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), work/(name+'.pt'))
    model.load_state_dict(torch.load(work/(name+'.pt'), weights_only=True, map_location=device))
    predictions = predict(model, loaders['test'], device)
    del model, optimizer, scaler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return predictions, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=Path('private/midterm_v1'))
    ap.add_argument('--zip', type=Path, default=Path('Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip'))
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=SEED, help='Training and bootstrap seed; does not change input manifests')
    ap.add_argument('--random-init', action='store_true', help='Explicit alternative; never silently substitute for ImageNet weights')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--cache', type=Path, help='Reuse an existing derived-image cache with the same preprocessing')
    ap.add_argument('--weights-cache', type=Path, help='Reuse an existing torchvision hub cache')
    a = ap.parse_args()
    globals()['SEED'] = a.seed
    if a.epochs < 1:
        raise ValueError('epochs must be positive')
    torch.set_num_threads(8)
    seen, unseen, valid, test = [load(a.work/(n+'.csv')) for n in ('seen', 'unseen', 'valid', 'test')]
    from collections import Counter
    assert len(seen) == len(unseen) > 0
    assert Counter((r['view'], r['y']) for r in seen) == Counter((r['view'], r['y']) for r in unseen)
    patient_sets = [{r['subject_id'] for r in group} for group in (seen+unseen, valid, test)]
    assert not (patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2])
    for group in (seen, unseen, valid, test):
        assert {int(r['y']) for r in group} == {0, 1}
        assert len({r['dicom_id'] for r in group}) == len(group)
    config = dict(seed=SEED, epochs=a.epochs, model='ResNet18', init='random' if a.random_init else 'ImageNet1K_V1',
                  resolution=224, batch_size=32, optimizer='AdamW', lr=1e-4, weight_decay=1e-4,
                  selection='minimum shared validation BCE', torch=torch.__version__,
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                  primary_metric='AUROC seen minus unseen', bootstrap='1000 paired patient resamples',
                  manifests={n: hashlib.sha256((a.work/(n+'.csv')).read_bytes()).hexdigest() for n in ('seen','unseen','valid','test')})
    (a.work/'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    cache = prepare_cache(a.work, a.zip, seen+unseen+valid+test, a.cache)
    if a.prepare_only:
        return
    # Keep downloaded model cache private to this experiment.
    torch.hub.set_dir(str(a.weights_cache or a.work/'torchhub'))
    weights = None if a.random_init else ResNet18_Weights.IMAGENET1K_V1
    pa, ha = train_one('seen', seen, valid, test, cache, a.work, a.epochs, weights)
    pb, hb = train_one('unseen', unseen, valid, test, cache, a.work, a.epochs, weights)
    y = np.array([int(r['y']) for r in test])
    result = dict(status='completed_real_image_experiment', config=config, seen=metrics(y, pa), unseen=metrics(y, pb),
                  constant_baseline=metrics(y, np.full(len(y), np.mean([int(r['y']) for r in seen]))),
                  paired_bootstrap=bootstrap(test, y, pa, pb), history=ha+hb)
    result['difference'] = {k: result['seen'][k]-result['unseen'][k] for k in result['seen']}
    (a.work/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with open(a.work/'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['subject_id','study_id','y','seen','unseen'])
        writer.writerows((r['subject_id'],r['study_id'],int(r['y']),float(x),float(z)) for r,x,z in zip(test,pa,pb))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for name, pred, color, style in [('Seen', pa, '#2266AA', '-'), ('Unseen', pb, '#B87716', '--')]:
        fpr, tpr, _ = roc_curve(y, pred)
        axs[0].plot(fpr, tpr, color=color, linestyle=style, label=f'{name}: AUC={roc_auc_score(y,pred):.3f}')
    axs[0].plot([0,1],[0,1], '--', color='gray'); axs[0].legend(); axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Same held-out target cohort')
    for name, hist, color, style in [('Seen',ha,'#2266AA','-'),('Unseen',hb,'#B87716','--')]:
        axs[1].plot([r['epoch'] for r in hist],[r['valid_loss'] for r in hist], marker='o',color=color,linestyle=style,label=name)
    axs[1].legend(); axs[1].set(xlabel='Epoch',ylabel='Validation BCE',title='Shared validation set')
    fig.tight_layout(); fig.savefig(a.work/'results.png', dpi=160); plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','history')}, indent=2),flush=True)


if __name__ == '__main__':
    main()
