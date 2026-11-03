# tests/test_basic.py
# =========================================================================
# KIỂM THỬ CƠ BẢN
# Chạy: python -m pytest tests/ hoặc python tests/test_basic.py
# =========================================================================

import os
import sys
import numpy as np

# Thêm thư mục gốc dự án vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_gaussian_blur,
    reduce_brightness,
)
from src.denoise import apply_median_blur, apply_gaussian_blur, apply_bilateral_filter
from src.color_classifier import classify_plate_color
from src.layout_classifier import classify_layout


def test_gaussian_noise_reproducible():
    """Cùng seed phải cho cùng kết quả nhiễu."""
    img = np.full((100, 200, 3), 128, dtype=np.uint8)

    noisy1 = add_gaussian_noise(img, sigma=25, seed=42)
    noisy2 = add_gaussian_noise(img, sigma=25, seed=42)

    assert np.array_equal(noisy1, noisy2), "Cùng seed phải cho cùng kết quả"
    print("✅ test_gaussian_noise_reproducible PASSED")


def test_salt_pepper_noise_reproducible():
    """Cùng seed phải cho cùng kết quả nhiễu muối tiêu."""
    img = np.full((100, 200, 3), 128, dtype=np.uint8)

    noisy1 = add_salt_pepper_noise(img, ratio=0.1, seed=42)
    noisy2 = add_salt_pepper_noise(img, ratio=0.1, seed=42)

    assert np.array_equal(noisy1, noisy2), "Cùng seed phải cho cùng kết quả"
    print("✅ test_salt_pepper_noise_reproducible PASSED")


def test_noise_zero_level():
    """Nhiễu mức 0 phải giữ ảnh nguyên."""
    img = np.full((100, 200, 3), 128, dtype=np.uint8)

    noisy = add_gaussian_noise(img, sigma=0)
    assert np.array_equal(img, noisy), "Sigma=0 phải giữ ảnh nguyên"

    noisy_sp = add_salt_pepper_noise(img, ratio=0)
    assert np.array_equal(img, noisy_sp), "Ratio=0 phải giữ ảnh nguyên"
    print("✅ test_noise_zero_level PASSED")


def test_noise_clip():
    """Ảnh nhiễu phải nằm trong [0, 255]."""
    img = np.full((100, 200, 3), 200, dtype=np.uint8)
    noisy = add_gaussian_noise(img, sigma=100, seed=1)

    assert noisy.dtype == np.uint8
    assert noisy.min() >= 0
    assert noisy.max() <= 255
    print("✅ test_noise_clip PASSED")


def test_gaussian_blur_odd_kernel():
    """Gaussian blur phải chấp nhận kernel chẵn (tự điều chỉnh)."""
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    # Kernel chẵn → phải tự chuyển thành lẻ
    blurred = add_gaussian_blur(img, ksize=4)
    assert blurred is not None
    print("✅ test_gaussian_blur_odd_kernel PASSED")


def test_brightness_reduction():
    """Giảm độ sáng mô phỏng ban đêm phải làm giảm giá trị trung bình pixel."""
    img = np.full((100, 200, 3), 200, dtype=np.uint8)
    darkened = reduce_brightness(img, factor=0.4)
    assert darkened.mean() < img.mean()
    assert darkened.dtype == np.uint8
    print("✅ test_brightness_reduction PASSED")


def test_median_blur():
    """Median blur giảm nhiễu muối tiêu."""
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    noisy = add_salt_pepper_noise(img, ratio=0.1, seed=42)

    denoised = apply_median_blur(noisy, ksize=5)

    # Ảnh giảm nhiễu phải gần ảnh gốc hơn ảnh nhiễu
    diff_noisy = np.mean(np.abs(noisy.astype(float) - img.astype(float)))
    diff_denoised = np.mean(np.abs(denoised.astype(float) - img.astype(float)))

    assert diff_denoised < diff_noisy, "Giảm nhiễu phải cải thiện"
    print("✅ test_median_blur PASSED")


def test_color_classifier_white():
    """Ảnh trắng phải được phân loại là Trắng."""
    # Tạo ảnh trắng giả lập biển số
    plate = np.full((60, 200, 3), 240, dtype=np.uint8)  # BGR gần trắng
    result = classify_plate_color(plate)
    assert result["color"] == "Trắng", f"Mong đợi Trắng, nhận được {result['color']}"
    print("✅ test_color_classifier_white PASSED")


def test_color_classifier_yellow():
    """Ảnh vàng phải được phân loại là Vàng."""
    import cv2
    # Tạo ảnh vàng trong HSV rồi chuyển sang BGR
    hsv = np.full((60, 200, 3), 0, dtype=np.uint8)
    hsv[:, :, 0] = 30   # Hue = vàng
    hsv[:, :, 1] = 200   # Saturation cao
    hsv[:, :, 2] = 200   # Value cao
    plate = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = classify_plate_color(plate)
    assert result["color"] == "Vàng", f"Mong đợi Vàng, nhận được {result['color']}"
    print("✅ test_color_classifier_yellow PASSED")


def test_color_classifier_blue():
    """Ảnh xanh phải được phân loại là Xanh dương."""
    import cv2
    hsv = np.full((60, 200, 3), 0, dtype=np.uint8)
    hsv[:, :, 0] = 110   # Hue = xanh
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = 200
    plate = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = classify_plate_color(plate)
    assert result["color"] == "Xanh dương", f"Mong đợi Xanh dương, nhận được {result['color']}"
    print("✅ test_color_classifier_blue PASSED")


def test_color_classifier_red():
    """Ảnh đỏ phải được phân loại là Đỏ."""
    import cv2
    hsv = np.full((60, 200, 3), 0, dtype=np.uint8)
    hsv[:, :, 0] = 5     # Hue = đỏ (khoảng 1)
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = 200
    plate = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = classify_plate_color(plate)
    assert result["color"] == "Đỏ", f"Mong đợi Đỏ, nhận được {result['color']}"
    print("✅ test_color_classifier_red PASSED")


def test_layout_wide_plate():
    """Biển rộng (w/h > 3) phải được phân loại 1 dòng."""
    result = classify_layout((0, 0, 400, 80), ocr_boxes=[{"bbox": [[10, 20], [45, 20], [45, 55], [10, 55]]}, {"bbox": [[60, 22], [95, 22], [95, 57], [60, 57]]}])
    assert result["layout"] == "1 dòng", f"Mong đợi 1 dòng, nhận được {result['layout']}"
    print("✅ test_layout_wide_plate PASSED")


def test_layout_square_plate():
    """Biển vuông (w/h < 1.8) phải được phân loại 2 dòng."""
    result = classify_layout((0, 0, 200, 140), ocr_boxes=[{"bbox": [[10, 15], [40, 15], [40, 45], [10, 45]]}, {"bbox": [[10, 85], [40, 85], [40, 115], [10, 115]]}])
    assert result["layout"] == "2 dòng", f"Mong đợi 2 dòng, nhận được {result['layout']}"
    print("✅ test_layout_square_plate PASSED")


def test_empty_ocr_no_crash():
    """OCR rỗng không được gây crash."""
    result = classify_layout((0, 0, 200, 60), ocr_boxes=[])
    assert result["layout"] is not None
    print("✅ test_empty_ocr_no_crash PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  CHẠY KIỂM THỬ CƠ BẢN")
    print("=" * 50 + "\n")

    tests = [
        test_gaussian_noise_reproducible,
        test_salt_pepper_noise_reproducible,
        test_noise_zero_level,
        test_noise_clip,
        test_gaussian_blur_odd_kernel,
        test_brightness_reduction,
        test_median_blur,
        test_color_classifier_white,
        test_color_classifier_yellow,
        test_color_classifier_blue,
        test_color_classifier_red,
        test_layout_wide_plate,
        test_layout_square_plate,
        test_empty_ocr_no_crash,
    ]

    passed = 0
    failed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  KẾT QUẢ: {passed} passed, {failed} failed / {len(tests)} tests")
    print(f"{'=' * 50}")
