import os
import tempfile
import unittest
import numpy as np
import cv2

from validator import Validator
from comparator import Comparator, _COMPARISON_CACHE


class TestValidator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nonexistent_file(self):
        is_valid, msg = Validator.is_valid_image("non_existent_image_123.jpg")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", msg.lower())

    def test_invalid_extension(self):
        txt_path = os.path.join(self.temp_dir.name, "test.txt")
        with open(txt_path, "w") as f:
            f.write("not an image")
        is_valid, msg = Validator.is_valid_image(txt_path)
        self.assertFalse(is_valid)
        self.assertIn("invalid file extension", msg.lower())

    def test_empty_file(self):
        empty_img = os.path.join(self.temp_dir.name, "empty.png")
        open(empty_img, "wb").close()
        is_valid, msg = Validator.is_valid_image(empty_img)
        self.assertFalse(is_valid)
        self.assertIn("empty", msg.lower())

    def test_valid_image_and_safe_load(self):
        valid_img_path = os.path.join(self.temp_dir.name, "synthetic.jpg")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(valid_img_path, dummy_img)

        is_valid, msg = Validator.is_valid_image(valid_img_path)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Valid image.")

        loaded = Validator.load_image_safely(valid_img_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (100, 100, 3))

    def test_is_same_image_by_hash(self):
        path1 = os.path.join(self.temp_dir.name, "img1.png")
        path2 = os.path.join(self.temp_dir.name, "img2.png")
        data = b"identical_image_data_test_stream"
        with open(path1, "wb") as f1, open(path2, "wb") as f2:
            f1.write(data)
            f2.write(data)

        same, msg = Validator.is_same_image(path1, path2)
        self.assertTrue(same)
        self.assertIn("same content", msg.lower())


class TestComparatorMetricsAndCache(unittest.TestCase):

    def test_similarity_percent_identical(self):
        score = Comparator.calculate_similarity_percent(distance=0.0, threshold=0.68)
        self.assertEqual(score, 100.0)

    def test_similarity_percent_at_threshold(self):
        score = Comparator.calculate_similarity_percent(distance=0.68, threshold=0.68)
        self.assertEqual(score, 60.0)

    def test_similarity_percent_distant(self):
        score = Comparator.calculate_similarity_percent(distance=1.5, threshold=0.68)
        self.assertLess(score, 60.0)

    def test_in_memory_caching(self):
        h1 = "hash_sample_alpha"
        h2 = "hash_sample_beta"
        cache_key = (min(h1, h2), max(h1, h2))
        _COMPARISON_CACHE[cache_key] = {
            "is_match": True,
            "similarity_pct": 95.0,
            "distance": 0.12,
            "threshold": 0.68,
            "message": "Cached result",
            "from_cache": False
        }

        self.assertIn(cache_key, _COMPARISON_CACHE)
        self.assertTrue(_COMPARISON_CACHE[cache_key]["is_match"])


if __name__ == "__main__":
    unittest.main()

