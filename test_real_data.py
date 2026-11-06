"""Интеграционные проверки на скачанных наборах, без подмены картинок заглушками."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lab2 import read_pair
from prepare_data import load_split, CARLA_CLASSES, CAMVID_CLASSES, verify_source_revision
from external_carla import ROOT, MANIFEST, verify_manifest


class RealDataTests(unittest.TestCase):
    def test_source_revision_and_changed_files_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            data = root / "data"
            data.mkdir()
            image = data / "example.txt"
            image.write_text("original")
            subprocess.run(["git", "-C", str(root), "add", "data/example.txt"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Student",
                            "-c", "user.email=student@example.invalid", "-c", "commit.gpgSign=false",
                            "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "fixture"], check=True)
            head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            verify_source_revision(data, head)
            with self.assertRaises(ValueError):
                verify_source_revision(data, "0" * 40)
            image.write_text("changed")
            with self.assertRaises(ValueError):
                verify_source_revision(data, head)

    def test_counts_and_no_duplicate_images_between_parts(self):
        manifest = json.loads(Path("data/manifest.json").read_text())
        for name, counts in [("carla", [848, 106, 106]), ("camvid", [367, 101, 233])]:
            split = load_split(name)
            self.assertEqual([len(v) for v in split.values()], counts)
            names = [{x.name for x, _ in pairs} for pairs in split.values()]
            hashes = [{manifest["files"][str(x)] for x, _ in pairs}
                      for pairs in split.values()]
            for groups in [names, hashes]:
                for i in range(3):
                    for j in range(i):
                        self.assertFalse(groups[i] & groups[j])

    def test_carla_nearby_frames_stay_together(self):
        split = load_split("carla")
        groups = [np.array([int(x.stem) for x, _ in pairs]) for pairs in split.values()]
        for i in range(3):
            for j in range(i):
                distance = np.abs(groups[i][:, None] - groups[j][None, :])
                self.assertGreater(distance.min(), 100)

    def test_real_masks_and_images(self):
        for name, classes in [("carla", CARLA_CLASSES), ("camvid", CAMVID_CLASSES)]:
            for pairs in load_split(name).values():
                for index in [0, len(pairs) // 2, -1]:
                    x, y = read_pair(*pairs[index], (96, 128), len(classes))
                    self.assertEqual(x.shape, (96, 128, 3))
                    self.assertEqual(y.shape, (96, 128, 1))
                    self.assertTrue(np.isfinite(x).all())
                    self.assertGreaterEqual(float(x.min()), 0)
                    self.assertLessEqual(float(x.max()), 1)
                    self.assertLess(int(y.max()), len(classes))
                    self.assertEqual(y.dtype, np.int32)

    def test_independent_carla_pairs_and_checksums(self):
        from lab2 import find_pairs
        manifest = verify_manifest()
        self.assertEqual(manifest["images"], 223)
        self.assertEqual(len(manifest["files"]), 446)
        self.assertTrue(MANIFEST.is_file())
        pairs = find_pairs(ROOT / "images", ROOT / "masks")
        self.assertEqual(len(pairs), 223)
        original = json.loads(Path("data/manifest.json").read_text())
        original_images = {digest for path, digest in original["files"].items()
                           if "/images/" in path}
        for image, _ in pairs:
            self.assertNotIn(manifest["files"][str(image)], original_images)
        for index in [0, len(pairs) // 2, -1]:
            x, y = read_pair(*pairs[index], (96, 128), 23)
            self.assertEqual(x.shape, (96, 128, 3))
            self.assertLessEqual(int(y.max()), 12)


if __name__ == "__main__":
    unittest.main()
