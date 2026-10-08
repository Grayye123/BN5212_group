"""Frozen multi-provider extension. Private inputs and images never leave disk."""
import argparse
from contextlib import contextmanager
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

SEEDS = (20260927, 20260928, 20260929)
PAIR_SEED = 20261008
GROUPS = ('seen', 'unseen', 'valid', 'test')
METRICS = ('AUROC', 'AUPRC', 'Brier')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    if not rows:
        raise ValueError('Empty cohort')
    with open(path, 'x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def summary(rows):
    return dict(studies=len(rows), patients=len({r['subject_id'] for r in rows}),
                positive_studies=sum(int(r['y']) for r in rows),
                positive_patients=len({r['subject_id'] for r in rows if int(r['y'])}),
                negative_patients=len({r['subject_id'] for r in rows if not int(r['y'])}),
                views=dict(Counter(r['view'] for r in rows)))


def validate_cohort(rows):
    if len({r['study_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate study')
    if len({r['dicom_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image')
    patients = defaultdict(set)
    for row in rows:
        if row['split'] not in ('train', 'valid', 'test') or row['view'] not in ('PA', 'AP') or row['y'] not in ('0', '1'):
            raise ValueError('Invalid frozen cohort row')
        patients[row['subject_id']].add(row['split'])
    if any(len(splits) != 1 for splits in patients.values()):
        raise ValueError('Patient leakage in source cohort')


def select_targets(rows, count=3, minimum=20):
    validate_cohort(rows)
    doctors = {r['doctor'] for r in rows}
    eligible = []
    for doctor in doctors:
        test = summary([r for r in rows if r['doctor'] == doctor and r['split'] == 'test'])
        train = [r for r in rows if r['doctor'] == doctor and r['split'] == 'train']
        if min(test['positive_patients'], test['negative_patients']) >= minimum and len(train) >= minimum:
            patients = len({r['subject_id'] for r in rows if r['doctor'] == doctor})
            eligible.append((doctor, patients))
    ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))
    if len(ranked) < count:
        raise ValueError('Insufficient eligible provider groups; do not relax criteria after seeing results')
    return [doctor for doctor, _ in ranked[:count]]


def make_pair(rows, doctor, seed=PAIR_SEED):
    rng = random.Random(seed)
    train = [r for r in rows if r['split'] == 'train']
    groups = dict(seen=[], unseen=[],
                  valid=[r for r in rows if r['split'] == 'valid' and r['doctor'] != doctor],
                  test=[r for r in rows if r['split'] == 'test' and r['doctor'] == doctor])
    for view in ('PA', 'AP'):
        for y in ('0', '1'):
            target = sorted([r for r in train if r['doctor'] == doctor and r['view'] == view and r['y'] == y], key=lambda r: r['study_id'])
            other = sorted([r for r in train if r['doctor'] != doctor and r['view'] == view and r['y'] == y], key=lambda r: r['study_id'])
            rng.shuffle(target)
            rng.shuffle(other)
            retained = min(len(target), len(other))
            groups['seen'].extend(target[:retained] + other[:len(other)-retained])
            groups['unseen'].extend(other)
    check_pair(groups, doctor)
    return groups


def check_pair(groups, doctor):
    for name in GROUPS:
        if {r['y'] for r in groups[name]} != {'0', '1'}:
            raise ValueError('A split has insufficient classes')
    if Counter((r['view'], r['y']) for r in groups['seen']) != Counter((r['view'], r['y']) for r in groups['unseen']):
        raise ValueError('Training strata mismatch')
    if sum(r['doctor'] == doctor for r in groups['seen']) < 20:
        raise ValueError('Insufficient target exposure')
    if any(r['doctor'] == doctor for r in groups['unseen'] + groups['valid']):
        raise ValueError('Target exposure in unseen or validation')
    if any(r['doctor'] != doctor for r in groups['test']):
        raise ValueError('Non-target test case')
    sets = [{r['subject_id'] for r in group} for group in
            (groups['seen'] + groups['unseen'], groups['valid'], groups['test'])]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Patient leakage')


def prepare(source, dest):
    if dest.exists():
        raise FileExistsError('Choose a new destination; frozen experiments are never overwritten')
    rows = read_rows(source / 'cohort.csv')
    targets = select_targets(rows)
    plan = dict(version='final-v1', status='frozen_before_training',
                seeds=list(SEEDS), pair_seed=PAIR_SEED, epochs=5,
                source_cohort_sha256=digest(source / 'cohort.csv'),
                source_labels_sha256=digest(source / 'labels.csv'),
                training_code_sha256=digest(Path(__file__).with_name('pilot_train.py')),
                selection='top 3 by cohort patient count, test positive/negative patients >=20, training exposure >=20',
                doctors=[])
    dest.mkdir(parents=True)
    for i, doctor in enumerate(targets, 1):
        name = f'provider_{i:02d}'
        root = dest / name
        root.mkdir()
        groups = make_pair(rows, doctor)
        hashes = {}
        for split, group in groups.items():
            write_rows(root / f'{split}.csv', group)
            hashes[split] = digest(root / f'{split}.csv')
        for seed in SEEDS:
            run = root / f'seed_{seed}'
            run.mkdir()
            for split in GROUPS:
                shutil.copyfile(root / f'{split}.csv', run / f'{split}.csv')
        plan['doctors'].append(dict(name=name, target=doctor, manifests=hashes,
                                   sizes={split: summary(group) for split, group in groups.items()},
                                   seen_target_studies=sum(r['doctor'] == doctor for r in groups['seen'])))
    write_json(dest / 'plan.json', plan)
    print(json.dumps(dict(stage='prepared', seeds=plan['seeds'], doctors=[{k: d[k] for k in ('name', 'sizes', 'seen_target_studies')} for d in plan['doctors']]), indent=2))
    return plan


def verify(source, dest):
    plan = json.loads((dest / 'plan.json').read_text(encoding='utf-8'))
    if plan['seeds'] != list(SEEDS) or plan['epochs'] != 5:
        raise ValueError('Frozen budget changed')
    if digest(source / 'cohort.csv') != plan['source_cohort_sha256'] or digest(source / 'labels.csv') != plan['source_labels_sha256']:
        raise ValueError('Frozen source changed')
    if digest(Path(__file__).with_name('pilot_train.py')) != plan['training_code_sha256']:
        raise ValueError('Training implementation changed after plan freeze')
    for doctor in plan['doctors']:
        groups = {}
        for split in GROUPS:
            path = dest / doctor['name'] / f'{split}.csv'
            if digest(path) != doctor['manifests'][split]:
                raise ValueError('Frozen provider manifest changed')
            groups[split] = read_rows(path)
            for seed in SEEDS:
                if digest(dest / doctor['name'] / f'seed_{seed}' / f'{split}.csv') != doctor['manifests'][split]:
                    raise ValueError('Training seed manifests differ')
        check_pair(groups, doctor['target'])
    return plan


def all_rows(dest, plan):
    return [row for doctor in plan['doctors'] for split in GROUPS
            for row in read_rows(dest / doctor['name'] / f'{split}.csv')]


def cache_images(dest, plan, reuse_cache, archive):
    import numpy as np
    from pilot_train import prepare_cache
    cache = dest / 'pixels224'
    cache.mkdir(exist_ok=True)
    rows = all_rows(dest, plan)
    unique = {r['dicom_id'] for r in rows}
    copied = 0
    for key in sorted(unique):
        target = cache / f'{key}.npy'
        source = reuse_cache / target.name
        if not target.exists() and source.is_file():
            shutil.copyfile(source, target)
            copied += 1
    print(json.dumps(dict(stage='cache', reused=copied, required=len(unique), existing=sum((cache / f'{k}.npy').is_file() for k in unique))), flush=True)
    prepare_cache(dest, archive, rows, cache)
    for key in unique:
        image = np.load(cache / f'{key}.npy', allow_pickle=False)
        if image.shape != (224, 224) or image.dtype != np.uint8 or image.max() <= image.min():
            raise ValueError('Image cache validation failed')
    print(json.dumps(dict(stage='cache_validated', images=len(unique))), flush=True)
    return cache


def independent_metrics(y, p):
    positives = [score for label, score in zip(y, p) if label]
    negatives = [score for label, score in zip(y, p) if not label]
    if not positives or not negatives:
        raise ValueError('Test metrics need both classes')
    auroc = sum((a > b) + .5*(a == b) for a in positives for b in negatives) / (len(positives)*len(negatives))
    buckets = defaultdict(list)
    for label, score in zip(y, p):
        buckets[score].append(label)
    total = correct = 0
    ap = 0.
    for score in sorted(buckets, reverse=True):
        labels = buckets[score]
        gained = sum(labels)
        total += len(labels)
        correct += gained
        ap += (gained/len(positives))*(correct/total)
    return dict(AUROC=auroc, AUPRC=ap, Brier=statistics.mean((score-label)**2 for label, score in zip(y, p)))


def verify_result(path, plan, doctor, seed):
    result = json.loads(path.read_text(encoding='utf-8'))
    config = result['config']
    if result['status'] != 'completed_real_image_experiment' or config['seed'] != seed or config['epochs'] != plan['epochs'] or config['manifests'] != doctor['manifests']:
        raise ValueError('Completed result does not match frozen plan')
    if config['model'] != 'ResNet18' or config['init'] != 'ImageNet1K_V1' or config['resolution'] != 224:
        raise ValueError('Unexpected training configuration')
    root = path.parent
    assets = ('seen.pt', 'unseen.pt', 'predictions.csv', 'results.png')
    if not all((root / asset).is_file() and (root / asset).stat().st_size > 0 for asset in assets):
        raise ValueError('Result JSON exists but required run assets are missing')
    test = read_rows(root / 'test.csv')
    predictions = read_rows(root / 'predictions.csv')
    keys = ('subject_id', 'study_id', 'y')
    if len(predictions) != len(test) or any(any(pred[k] != row[k] for k in keys) for pred, row in zip(predictions, test)):
        raise ValueError('Predictions do not match frozen test cases')
    y = [int(r['y']) for r in test]
    for name in ('seen', 'unseen'):
        p = [float(r[name]) for r in predictions]
        if not all(math.isfinite(score) and 0 <= score <= 1 for score in p):
            raise ValueError('Invalid prediction scores')
        actual = independent_metrics(y, p)
        if any(not math.isclose(actual[k], result[name][k], abs_tol=1e-10, rel_tol=0) for k in METRICS):
            raise ValueError('Independent metric verification failed')
    if any(not math.isclose(result['difference'][k], result['seen'][k]-result['unseen'][k], abs_tol=1e-10, rel_tol=0) for k in METRICS):
        raise ValueError('Metric difference mismatch')
    if result['paired_bootstrap']['requested_replicates'] != 1000 or result['paired_bootstrap']['valid_replicates'] < 900:
        raise ValueError('Insufficient paired bootstrap replicates')
    if len(result['history']) != 2*plan['epochs']:
        raise ValueError('Incomplete training history')
    return result


def train(dest, plan, cache, weights_cache):
    for doctor in plan['doctors']:
        for seed in SEEDS:
            run = dest / doctor['name'] / f'seed_{seed}'
            result_path = run / 'results.json'
            if result_path.exists():
                required = ('seen.pt', 'unseen.pt', 'predictions.csv', 'results.png')
                try:
                    json.loads(result_path.read_text(encoding='utf-8'))
                    readable = True
                except json.JSONDecodeError:
                    readable = False
                if readable and all((run / name).is_file() and (run / name).stat().st_size for name in required):
                    verify_result(result_path, plan, doctor, seed)
                    print(json.dumps(dict(stage='resume_skip_verified', provider=doctor['name'], seed=seed)), flush=True)
                    continue
            if any((run / name).exists() for name in ('seen.pt', 'unseen.pt', 'run_config.json', 'results.json', 'training.log')):
                saved = archive_incomplete(run)
                print(json.dumps(dict(stage='restart_preserving_incomplete_attempt', provider=doctor['name'], seed=seed, attempt=saved.name)), flush=True)
            command = [sys.executable, str(Path(__file__).with_name('pilot_train.py')), '--work', str(run),
                       '--epochs', str(plan['epochs']), '--seed', str(seed), '--cache', str(cache),
                       '--weights-cache', str(weights_cache)]
            started = time.time()
            print(json.dumps(dict(stage='training_start', provider=doctor['name'], seed=seed)), flush=True)
            with open(run / 'training.log', 'x', encoding='utf-8') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            verify_result(result_path, plan, doctor, seed)
            print(json.dumps(dict(stage='training_complete', provider=doctor['name'], seed=seed, seconds=round(time.time()-started, 1))), flush=True)


def archive_incomplete(run):
    """Retain interrupted assets; rerun the SAME seed, never select by score."""
    root = run.resolve()
    assets = ('seen.pt', 'unseen.pt', 'run_config.json', 'results.json',
              'predictions.csv', 'results.png', 'training.log')
    sources = [run / name for name in assets if (run / name).exists()]
    if any(path.resolve().parent != root or not path.is_file() for path in sources):
        raise ValueError('Unsafe interrupted asset path')
    attempt = run / f'interrupted_attempt_{time.time_ns()}'
    if attempt.resolve().parent != root:
        raise ValueError('Unsafe interrupted archive directory')
    attempt.mkdir()
    for path in sources:
        path.rename(attempt / path.name)
    write_json(attempt / 'recovery.json', dict(reason='incomplete assets after interruption', action='same frozen seed restarted from ImageNet; manifests unchanged'))
    return attempt


@contextmanager
def study_lock(dest):
    """A process-owned lock releases automatically after shutdown."""
    with open(dest / 'runner.lock', 'a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another runner owns this experiment; do not start twice') from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def report(dest, plan):
    per_run = []
    for doctor in plan['doctors']:
        for seed in SEEDS:
            path = dest / doctor['name'] / f'seed_{seed}' / 'results.json'
            result = verify_result(path, plan, doctor, seed)
            per_run.append(dict(provider=doctor['name'], seed=seed,
                                sizes=doctor['sizes'], seen_target_studies=doctor['seen_target_studies'],
                                seen=result['seen'], unseen=result['unseen'], difference=result['difference'],
                                constant_baseline=result['constant_baseline'], paired_bootstrap=result['paired_bootstrap'],
                                best_epochs={name: min([r for r in result['history'] if r['model'] == name], key=lambda r: r['valid_loss'])['epoch'] for name in ('seen', 'unseen')}))
    providers = []
    for doctor in plan['doctors']:
        runs = [r for r in per_run if r['provider'] == doctor['name']]
        providers.append(dict(provider=doctor['name'], differences={metric: dict(
            mean=statistics.mean(r['difference'][metric] for r in runs),
            sample_sd=statistics.stdev(r['difference'][metric] for r in runs),
            minimum=min(r['difference'][metric] for r in runs),
            maximum=max(r['difference'][metric] for r in runs)) for metric in METRICS}))
    value = dict(status='completed_real_multi_provider_experiment', runs=per_run, provider_seed_summary=providers,
                 source_cohort_sha256=plan['source_cohort_sha256'], source_labels_sha256=plan['source_labels_sha256'],
                 limitations=['Report-derived labels; no additional independent human or expert image validation.',
                              'Existing patient split reused; extension retest, not new external validation.',
                              'Providers may share test patients; do not pool them as independent samples.',
                              'Each paired patient interval conditions on trained models; seed spread is descriptive, not a confidence interval.',
                              'Provider association is observational, not causal or an assessment of provider skill.',
                              'AUPRC is implemented as average precision.'])
    path = dest / 'summary_results.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Refusing to change completed summary')
    else:
        write_json(path, value)
    print(json.dumps(dict(stage='summary_complete', runs=len(per_run), providers=providers), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'verify', 'cache', 'train', 'report', 'all'))
    parser.add_argument('--source', type=Path, default=Path('private/midterm_v1'))
    parser.add_argument('--dest', type=Path, default=Path('private/final_v1'))
    parser.add_argument('--reuse-cache', type=Path, default=Path('private/midterm_v1/pixels224'))
    parser.add_argument('--zip', type=Path, default=Path('Z:/MIMIC-CXR/BN5212_MIMIC-CXR.zip'))
    parser.add_argument('--weights-cache', type=Path, default=Path('private/midterm_v1/torchhub'))
    args = parser.parse_args()
    if args.dest.resolve() == args.source.resolve() or args.source.resolve() in args.dest.resolve().parents:
        raise ValueError('Destination must not overwrite or nest inside the source experiment')
    if args.stage == 'prepare' or (args.stage == 'all' and not args.dest.exists()):
        prepare(args.source, args.dest)
    plan = verify(args.source, args.dest)
    print(json.dumps(dict(stage='plan_verified', paired_runs=len(plan['doctors'])*len(SEEDS), model_fits=len(plan['doctors'])*len(SEEDS)*2)), flush=True)
    with study_lock(args.dest):
        if args.stage in ('cache', 'train', 'all'):
            cache = cache_images(args.dest, plan, args.reuse_cache, args.zip)
        if args.stage in ('train', 'all'):
            train(args.dest, plan, cache, args.weights_cache)
        if args.stage in ('report', 'train', 'all'):
            report(args.dest, plan)


if __name__ == '__main__':
    main()
