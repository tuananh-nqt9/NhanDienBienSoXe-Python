# tests/test_real.py
# =========================================================================
# KIỂM THỬ TRÊN ẢNH THỰC TẾ
# Kiểm tra các ảnh mẫu biển số thực tế:
# 1. Biển 99-E1 / 222.68 (Ảnh 99-la-tinh-nao-thumbnail.jpg)
# 2. Biển 54-L1 / 9999 (Ảnh bien-so-xe-theo-phong-thuy.jpg)
# =========================================================================

import os
import sys
import unittest
import cv2

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config
from src.pipeline import process_image
from src.ocr_engine import OCREngine


class TestRealPlateSamples(unittest.TestCase):
    """Kiểm thử trên các ảnh thật trong data/samples."""

    @classmethod
    def setUpClass(cls):
        cls.engine = OCREngine()
        cls.engine.initialize()

    def test_sample_99_e1(self):
        """Ảnh 99-la-tinh-nao-thumbnail.jpg: biển 99-E1 / 222.68."""
        path = os.path.join(config.SAMPLES_DIR, "99-la-tinh-nao-thumbnail.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img, f"Không đọc được ảnh: {path}")

        result = process_image(img, ocr_engine=self.engine, use_full_image=False)
        self.assertEqual(result.get("n_plates"), 1, "Chỉ được phát hiện đúng 1 biển số")
        plate = result["plates"][0]
        self.assertEqual(plate["color"], "Trắng")
        self.assertEqual(plate["layout"], "2 dòng")
        self.assertEqual(plate["plate_text"], "99-E1 / 222.68")
        print("✅ test_sample_99_e1 PASSED (99-E1 / 222.68 | Trắng | 2 dòng)")

    def test_sample_54_l1(self):
        """Ảnh bien-so-xe-theo-phong-thuy.jpg: biển 54-L1 / 9999 (chỉ 1 biển, không có biển phụ)."""
        path = os.path.join(config.SAMPLES_DIR, "bien-so-xe-theo-phong-thuy.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img, f"Không đọc được ảnh: {path}")

        result = process_image(img, ocr_engine=self.engine, use_full_image=False)
        self.assertEqual(result.get("n_plates"), 1, f"Chỉ được phát hiện đúng 1 biển số, nhận được {result.get('n_plates')}")
        plate = result["plates"][0]
        self.assertEqual(plate["color"], "Trắng")
        self.assertEqual(plate["layout"], "2 dòng")
        self.assertEqual(plate["plate_text"], "54-L1 / 9999", f"Nhận diện sai: {plate['plate_text']}")
        print("✅ test_sample_54_l1 PASSED (54-L1 / 9999 | Trắng | 2 dòng)")

    def test_sample_80b_5599_full_car(self):
        """Ảnh toàn xe oto-bien-xanh-trung-so-dep-hy-huu-khong-he-hiem-4.jpg: biển 80B-5599."""
        path = os.path.join(config.SAMPLES_DIR, "oto-bien-xanh-trung-so-dep-hy-huu-khong-he-hiem-4.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img, f"Không đọc được ảnh: {path}")

        result = process_image(img, ocr_engine=self.engine, use_full_image=False)
        self.assertEqual(result.get("n_plates"), 1, f"Chỉ được phát hiện đúng 1 biển số, nhận được {result.get('n_plates')}")
        plate = result["plates"][0]
        self.assertEqual(plate["color"], "Xanh dương")
        self.assertEqual(plate["layout"], "1 dòng")
        self.assertEqual(plate["plate_text"], "80B-5599", f"Nhận diện sai: {plate['plate_text']}")
        print("✅ test_sample_80b_5599_full_car PASSED (80B-5599 | Xanh dương | 1 dòng)")

    def test_sample_29d_01245_yellow(self):
        """Ảnh xe-kinh-doanh-van-tai-chuyen-sang-mau-vang.jpg: biển 29D / 012.45."""
        path = os.path.join(config.SAMPLES_DIR, "xe-kinh-doanh-van-tai-chuyen-sang-mau-vang.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img, f"Không đọc được ảnh: {path}")

        result = process_image(img, ocr_engine=self.engine, use_full_image=False)
        self.assertEqual(result.get("n_plates"), 1, f"Chỉ được phát hiện đúng 1 biển số, nhận được {result.get('n_plates')}")
        plate = result["plates"][0]
        self.assertEqual(plate["color"], "Vàng")
        self.assertEqual(plate["layout"], "2 dòng")
        self.assertEqual(plate["plate_text"], "29D / 012.45", f"Nhận diện sai: {plate['plate_text']}")
        print("✅ test_sample_29d_01245_yellow PASSED (29D / 012.45 | Vàng | 2 dòng)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CHẠY KIỂM THỬ TRÊN ẢNH THỰC TẾ")
    print("=" * 60)
    unittest.main()
