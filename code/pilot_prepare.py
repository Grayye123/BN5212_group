"""Prepare a provisional, reproducible midterm cohort; never edit input files."""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

SEED = 20260927
LABELS = {'1阳性': '1', '0明确阴性': '0', '-1不确定': '-1', '空未提及': ''}


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rule_label(text, original):
    """Conservative text-policy reconciliation, not a clinical diagnosis.

    Only override a single target-bearing sentence with an explicit policy cue.
    Multiple target sentences or competing negation/differential are excluded
    when policy cues are present, rather than assigned random labels.
    """
    text = text.lower()
    sentences = [s for s in re.split(r'[.!?;]\s*', text)
                 if re.search(r'atelecta\w*|volume loss|collaps\w*', s)]
    affected = any(re.search(r'\blikely\b|volume loss', s) for s in sentences)
    if not affected:
        return original, 'retained'
    if len(sentences) != 1:
        return '-1', 'policy_ambiguous_excluded'
    s = sentences[0]
    # Scope is deliberately conservative: unrelated negatives may also exclude.
    if re.search(r'\b(no|not|without|resolved|resolution|versus|vs|or|cannot|may|could|possible|possibly|suspected)\b', s):
        return '-1', 'policy_ambiguous_excluded'
    if 'volume loss' in s:
        if re.search(r'\b(very mild|minimal|trace|slight|very small)\b', s):
            # Explicit atelectasis elsewhere in the same sentence is ambiguous.
            return ('-1', 'policy_ambiguous_excluded') if re.search(r'atelecta|collaps', s) else ('', 'volume_very_mild_not_counted')
        if re.search(r'\blikely\b', s) and not re.search(r'\bmost likely\b', s):
            return '-1', 'likely_uncertain'
        return '1', 'volume_counted'
    if re.search(r'\bmost likely\b', s):
        return '1', 'most_likely_positive'
    if re.search(r'\blikely\b', s):
        return '-1', 'likely_uncertain'
    return original, 'retained'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, default=Path('out/audit'))
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--dest', type=Path, default=Path('private/midterm_v1'))
    ap.add_argument('--max-train-per-group', type=int, default=0)
    ap.add_argument('--max-valid', type=int, default=0)
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    baseline = read_csv(args.audit / 'chexbert_labels.csv')
    assert len(baseline) == len({r['study_id'] for r in baseline}) == 3351
    human = {}
    wb = openpyxl.load_workbook(args.review, read_only=True, data_only=True)
    for sheet, col in [('分歧裁决', 6), ('一致复核', 5)]:
        for r in list(wb[sheet].values)[1:]:
            key = str(r[col]).replace(' ', '')
            assert key in LABELS, 'Missing or invalid review label'
            assert str(r[2]) not in human, 'Duplicate review key'
            human[str(r[2])] = LABELS[key]
    assert len(human) == 272
    contexts = {str(r['study_id']): r['context'] for r in
                (json.loads(line) for line in (args.audit / 'review_judge.jsonl').read_text(encoding='utf-8').splitlines())}
    labels = []
    for r in baseline:
        sid = r['study_id']
        prior = human.get(sid, r['Atelectasis'])
        label, reason = rule_label(contexts.get(sid, ''), prior)
        labels.append(dict(study_id=sid, subject_id=r['subject_id'], original=r['Atelectasis'],
                           human=human.get(sid, 'NOT_REVIEWED'), label=label,
                           source='human' if sid in human else 'CheXbert', reconciliation=reason))
    write_csv(args.dest / 'labels.csv', labels)
    labs = {r['study_id']: r for r in labels}
    manifest = read_csv(args.audit / 'subset_manifest.csv')
    heads = {r['dicom_id']: r for r in read_csv(args.audit / 'dicom_headers.csv')}
    choices = defaultdict(list)
    for r in manifest:
        h = heads[r['dicom_id']]
        assert h['study_id'] == r['study_id'] and h['subject_id'] == r['subject_id']
        if h['ViewPosition'] in ('PA', 'AP'):
            choices[r['study_id']].append(dict(r, view=h['ViewPosition']))
    cohort = []
    for sid, group in sorted(choices.items()):
        if labs[sid]['label'] == '-1':
            continue
        r = sorted(group, key=lambda x: (x['view'] != 'PA', x['dicom_id']))[0]
        assert labs[sid]['subject_id'] == r['subject_id']
        cohort.append(dict(subject_id=r['subject_id'], study_id=sid, dicom_id=r['dicom_id'],
                           doctor=r['attending_provider_id'], view=r['view'], y=int(labs[sid]['label'] == '1')))
    patient_y = defaultdict(int)
    doctor_patients = defaultdict(set)
    for r in cohort:
        patient_y[r['subject_id']] |= r['y']
        doctor_patients[r['doctor']].add(r['subject_id'])
    target = sorted(doctor_patients, key=lambda d: (-len(doctor_patients[d]), d))[0]
    rng = random.Random(SEED)
    split = {}
    for y in (0, 1):
        ids = sorted(p for p in patient_y if patient_y[p] == y)
        rng.shuffle(ids)
        n = len(ids)
        for i, p in enumerate(ids):
            split[p] = 'train' if i < int(.6*n) else ('valid' if i < int(.8*n) else 'test')
    for r in cohort:
        r['split'] = split[r['subject_id']]
    train = [r for r in cohort if r['split'] == 'train']
    valid = [r for r in cohort if r['split'] == 'valid' and r['doctor'] != target]
    test = [r for r in cohort if r['split'] == 'test' and r['doctor'] == target]
    seen, unseen = [], []
    # Match both label and view counts; maximize retained target training studies.
    for view in ('PA', 'AP'):
        for y in (0, 1):
            a = [r for r in train if r['view'] == view and r['y'] == y and r['doctor'] == target]
            b = [r for r in train if r['view'] == view and r['y'] == y and r['doctor'] != target]
            rng.shuffle(a); rng.shuffle(b)
            n = len(b)
            take = min(len(a), n)
            seen.extend(a[:take] + b[:n-take])
            unseen.extend(b)
    def stratified_budget(group, budget, strata):
        if not budget or budget >= len(group):
            return group
        if budget < len(strata):
            raise ValueError('Budget too small for strata')
        buckets = {key: [r for r in group if (r['view'],r['y']) == key] for key in strata}
        quotas = {key: int(budget*len(buckets[key])/len(group)) for key in strata}
        remainder = sorted(strata, key=lambda key: (-(budget*len(buckets[key])/len(group)-quotas[key]),key))
        for key in remainder[:budget-sum(quotas.values())]:
            quotas[key] += 1
        selected = []
        for key in strata:
            bucket = sorted(buckets[key], key=lambda r:r['study_id'])
            selected.extend(rng.sample(bucket,quotas[key]))
        return selected
    strata = [(view,y) for view in ('PA','AP') for y in (0,1)]
    seen = stratified_budget(seen,args.max_train_per_group,strata)
    unseen = stratified_budget(unseen,args.max_train_per_group,strata)
    valid = stratified_budget(valid,args.max_valid,strata)
    assert len(seen) == len(unseen) > 0
    assert Counter((r['view'], r['y']) for r in seen) == Counter((r['view'], r['y']) for r in unseen)
    assert sum(r['doctor'] == target for r in seen) >= 20
    assert not any(r['doctor'] == target for r in unseen + valid)
    groups = [{r['subject_id'] for r in g} for g in [seen + unseen, valid, test]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for group in (seen, unseen, valid, test):
        assert {r['y'] for r in group} == {0, 1}
    for name, group in [('cohort', cohort), ('seen', seen), ('unseen', unseen), ('valid', valid), ('test', test)]:
        write_csv(args.dest / (name + '.csv'), group)
    def summary(group):
        return dict(studies=len(group), patients=len({r['subject_id'] for r in group}),
                    positive_studies=sum(r['y'] for r in group),
                    positive_patients=len({r['subject_id'] for r in group if r['y']}),
                    negative_patients=len({r['subject_id'] for r in group if not r['y']}),
                    views=dict(Counter(r['view'] for r in group)))
    info = dict(version='midterm-v1-provisional', seed=SEED, label_counts=dict(Counter(r['label'] for r in labels)),
                max_train_per_group=args.max_train_per_group, max_valid=args.max_valid,
                reasons=dict(Counter(r['reconciliation'] for r in labels)), human_rows=len(human),
                changed_from_human=sum(r['human'] != 'NOT_REVIEWED' and r['human'] != r['label'] for r in labels),
                cohort=summary(cohort), seen=summary(seen), unseen=summary(unseen), valid=summary(valid), test=summary(test),
                seen_target_studies=sum(r['doctor'] == target for r in seen),
                workbook_sha256=hashlib.sha256(args.review.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((args.dest/'cohort.csv').read_bytes()).hexdigest(),
                limitations=['Provisional report-derived labels, not an expert image gold standard.',
                             'Mechanical unmentioned labels not independently human-validated.',
                             'Rule reconciliation is conservative and may exclude true positives.',
                             'One target, one seed; exploratory comparison, no causal doctor interpretation.',
                             'Positive and negative patient counts may overlap for longitudinal patients.'])
    (args.dest / 'summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
