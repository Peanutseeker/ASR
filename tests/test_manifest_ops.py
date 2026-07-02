from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from asr_noise_robust.manifest import Utterance, read_manifest, write_manifest
from asr_noise_robust.manifest_ops import write_manifest_subset


class ManifestOpsTests(unittest.TestCase):
    def test_write_manifest_subset_keeps_first_n_rows(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.csv"
            target = root / "subset.csv"
            rows = [
                Utterance(str(index), f"{index}.flac", 1.0, "TEXT", "text", "train")
                for index in range(5)
            ]
            write_manifest(source, rows)

            written = write_manifest_subset(source, target, limit=3)

            self.assertEqual(written, 3)
            self.assertEqual([row.utt_id for row in read_manifest(target)], ["0", "1", "2"])


if __name__ == "__main__":
    unittest.main()

