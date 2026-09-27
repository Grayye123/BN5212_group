"""Policy boundary checks and opt-in synthetic GPU smoke test (not results)."""
import argparse
from pathlib import Path
import unittest

from pilot_prepare import rule_label


class PolicyTests(unittest.TestCase):
    def test_strength(self):
        self.assertEqual(rule_label('Most likely atelectasis.', '-1')[0], '1')
        self.assertEqual(rule_label('Likely atelectasis.', '1')[0], '-1')

    def test_volume(self):
        self.assertEqual(rule_label('Only very mild volume loss.', '1')[0], '')
        self.assertEqual(rule_label('Moderate volume loss.', '')[0], '1')

    def test_conflict_excluded(self):
        for text in ['No volume loss.', 'Atelectasis or pneumonia is most likely.',
                     'Likely atelectasis. Another focus of atelectasis.']:
            self.assertEqual(rule_label(text, '1')[0], '-1')


def smoke():
    import numpy as np
    from pilot_train import train_one, bootstrap, image_from_dicom
    import io
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
    ds.Rows, ds.Columns = 32, 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.arange(2048, dtype=np.uint16).reshape(32,64).tobytes()
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    pixels = image_from_dicom(stream.getvalue())
    assert pixels.shape == (224,224) and pixels.dtype == np.uint8
    assert pixels.max() > pixels.min()
    root = Path('private/midterm_v1/smoke')
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(8):
        np.save(root/f'synthetic{i}.npy', np.random.default_rng(i).integers(0, 255, (224,224), dtype=np.uint8))
        rows.append(dict(dicom_id=f'synthetic{i}', subject_id=f'synthetic{i}', study_id=f'synthetic{i}', y=i % 2))
    p, h = train_one('synthetic_smoke', rows[:4], rows[4:6], rows[6:], root, root, 1, None)
    assert len(p) == 2 and np.isfinite(p).all() and len(h) == 1
    assert bootstrap(rows[6:], np.array([0,1]), np.array([.1,.9]), np.array([.2,.8]), 20)['valid_replicates'] > 0
    print('PASS: synthetic GPU backpropagation, checkpoint restore, predictions and patient bootstrap. NOT real-data results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.smoke:
        smoke()
