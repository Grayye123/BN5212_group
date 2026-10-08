"""Guard against leakage, outcome-based selection, manifest drift and overwrites."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from final_study import archive_incomplete, check_pair, independent_metrics, make_pair, prepare, select_targets, study_lock, verify, write_rows


def example():
    rows = []
    for doctor in ('a', 'b', 'c'):
        for split, count in [('train', 20), ('valid', 4), ('test', 40)]:
            for i in range(count):
                key = f'{doctor}_{split}_{i}'
                rows.append(dict(subject_id=key, study_id=key, dicom_id=key,
                                 doctor=doctor, view='PA' if i % 4 < 2 else 'AP',
                                 y=str(i % 2), split=split))
    return rows


class FinalStudyTests(unittest.TestCase):
    def test_interrupted_assets_preserved_and_lock_exclusive(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp).resolve()
            assert root.is_relative_to(Path.cwd().resolve())
            (root / 'seen.pt').write_bytes(b'incomplete checkpoint')
            (root / 'seen.csv').write_text('frozen manifest', encoding='utf-8')
            saved = archive_incomplete(root)
            self.assertEqual((saved / 'seen.pt').read_bytes(), b'incomplete checkpoint')
            self.assertFalse((root / 'seen.pt').exists())
            self.assertEqual((root / 'seen.csv').read_text(encoding='utf-8'), 'frozen manifest')
            with study_lock(root):
                with self.assertRaises((RuntimeError, OSError)):
                    with study_lock(root):
                        self.fail('Second process lock acquired')

    def test_metrics_ties_and_reversed_ranking(self):
        tied = independent_metrics([0, 1, 1, 0], [.5]*4)
        self.assertEqual(tied, dict(AUROC=.5, AUPRC=.5, Brier=.25))
        bad = independent_metrics([0, 1], [.9, .1])
        self.assertEqual(bad['AUROC'], 0.)
        self.assertEqual(bad['AUPRC'], .5)
        self.assertAlmostEqual(bad['Brier'], .81)

    def test_selection_ignores_input_order(self):
        rows = example()
        self.assertEqual(select_targets(rows), ['a', 'b', 'c'])
        self.assertEqual(select_targets(list(reversed(rows))), ['a', 'b', 'c'])

    def test_ineligible_provider_not_silently_substituted(self):
        rows = [r for r in example() if not (r['doctor'] == 'c' and r['split'] == 'test' and r['y'] == '1')]
        with self.assertRaisesRegex(ValueError, 'Insufficient eligible'):
            select_targets(rows)

    def test_pair_exposure_and_patient_isolation(self):
        groups = make_pair(example(), 'a')
        self.assertEqual(len(groups['seen']), len(groups['unseen']))
        self.assertEqual(sum(r['doctor'] == 'a' for r in groups['seen']), 20)
        groups['test'][0] = dict(groups['test'][0], subject_id=groups['seen'][0]['subject_id'])
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            check_pair(groups, 'a')

    def test_source_patient_leak_rejected(self):
        rows = example()
        rows[-1]['subject_id'] = rows[0]['subject_id']
        with self.assertRaisesRegex(ValueError, 'Patient leakage'):
            select_targets(rows)

    def test_freeze_seed_identity_tamper_and_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            assert Path(tmp).resolve().is_relative_to(Path.cwd().resolve())
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            write_rows(source / 'cohort.csv', example())
            (source / 'labels.csv').write_text('synthetic test fixture\n', encoding='utf-8')
            dest = root / 'new'
            with redirect_stdout(io.StringIO()):
                prepare(source, dest)
            plan = verify(source, dest)
            self.assertEqual(len(plan['doctors']), 3)
            with self.assertRaises(FileExistsError):
                prepare(source, dest)
            path = dest / 'provider_01' / f"seed_{plan['seeds'][0]}" / 'test.csv'
            path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'seed manifests differ'):
                verify(source, dest)


if __name__ == '__main__':
    unittest.main()
