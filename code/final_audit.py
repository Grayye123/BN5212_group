"""Structural label and observed case-mix audit, never clinical validation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from final_study import GROUPS, digest, read_rows, verify, write_json


def audit(source, dest, headers):
    plan = verify(source, dest)
    labels = read_rows(source / 'labels.csv')
    if len(labels) != len({r['study_id'] for r in labels}):
        raise ValueError('Duplicate frozen label')
    by_study = {r['study_id']: r for r in labels}
    cohort = read_rows(source / 'cohort.csv')
    for row in cohort:
        label = by_study.get(row['study_id'])
        if label is None or label['subject_id'] != row['subject_id'] or label['label'] == '-1' or row['y'] != str(int(label['label'] == '1')):
            raise ValueError('Cohort-label mismatch')
    metadata = read_rows(headers)
    if len(metadata) != len({r['dicom_id'] for r in metadata}):
        raise ValueError('Duplicate image header')
    by_image = {r['dicom_id']: r for r in metadata}
    cases = []
    for doctor in plan['doctors']:
        groups = {}
        for name in GROUPS:
            rows = read_rows(dest / doctor['name'] / f'{name}.csv')
            patient_counts = Counter(r['subject_id'] for r in rows)
            manufacturers = Counter()
            dimensions = Counter()
            for row in rows:
                header = by_image.get(row['dicom_id'])
                if header is None or header['study_id'] != row['study_id'] or header['subject_id'] != row['subject_id'] or header['ViewPosition'] != row['view']:
                    raise ValueError('Image-header mismatch')
                manufacturers[header.get('Manufacturer', '') or 'missing'] += 1
                dimensions[f"{header['Rows']}x{header['Columns']}"] += 1
            groups[name] = dict(patients=len(patient_counts), mean_studies_per_patient=statistics.mean(patient_counts.values()),
                                maximum_studies_per_patient=max(patient_counts.values()),
                                patients_with_repeated_studies=sum(n > 1 for n in patient_counts.values()),
                                manufacturer_counts=dict(manufacturers), original_dimension_counts=dict(dimensions))
        seen = read_rows(dest / doctor['name'] / 'seen.csv')
        unseen = read_rows(dest / doctor['name'] / 'unseen.csv')
        shared_studies = len({r['study_id'] for r in seen} & {r['study_id'] for r in unseen})
        cases.append(dict(provider=doctor['name'], groups=groups, shared_training_studies=shared_studies))
    test_sets = [{r['subject_id'] for r in read_rows(dest / d['name'] / 'test.csv')} for d in plan['doctors']]
    intersections = {f"{plan['doctors'][i]['name']}__{plan['doctors'][j]['name']}": len(test_sets[i] & test_sets[j])
                     for i in range(len(test_sets)) for j in range(i+1, len(test_sets))}
    value = dict(status='structural_audit_passed_not_clinical_validation',
                 frozen_labels=len(labels), frozen_cohort=len(cohort),
                 label_counts=dict(Counter(r['label'] or 'unmentioned' for r in labels)),
                 source_counts=dict(Counter(r['source'] for r in labels)),
                 reconciliation_counts=dict(Counter(r['reconciliation'] for r in labels)),
                 human_rows=sum(r['human'] != 'NOT_REVIEWED' for r in labels),
                 changed_from_human=sum(r['human'] != 'NOT_REVIEWED' and r['human'] != r['label'] for r in labels),
                 additional_independent_human_reviews=0, additional_expert_image_reviews=0,
                 source_cohort_sha256=plan['source_cohort_sha256'], source_labels_sha256=plan['source_labels_sha256'],
                 headers_sha256=digest(headers), case_mix=cases, between_provider_test_patient_overlap=intersections,
                 limitations=['Structural consistency checks cannot estimate label accuracy or establish an image gold standard.',
                              'Unmentioned report labels remain unvalidated and do not establish medical absence.',
                              'Manufacturer, view and repeated-study counts do not fully capture clinical severity or case mix.'])
    path = dest / 'audit_summary.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Refusing to replace a different audit')
    else:
        write_json(path, value)
    print(json.dumps(dict(status=value['status'], labels=len(labels), cohort=len(cohort), human_rows=value['human_rows'],
                          changed_from_human=value['changed_from_human'], between_provider_test_patient_overlap=intersections), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('private/midterm_v1'))
    parser.add_argument('--dest', type=Path, default=Path('private/final_v1'))
    parser.add_argument('--headers', type=Path, default=Path('out/audit/dicom_headers.csv'))
    args = parser.parse_args()
    audit(args.source, args.dest, args.headers)
