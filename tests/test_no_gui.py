# tests/test_no_gui.py
# =========================================================================
# KIỂM THỬ KHÔNG CẦN GUI - chạy từ dòng lệnh để xác nhận các module
# hoạt động sau khi cài thư viện.
#
# Chạy: python tests/test_no_gui.py
# =========================================================================

import os
import sys
import traceback

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✅ {name}")
        PASS += 1
    except ImportError as e:
        print(f"  ❌ {name}: THIẾU THƯ VIỆN - {e}")
        FAIL += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        FAIL += 1


print("\n" + "=" * 60)
print("  KIỂM THỬ IMPORT VÀ CHỨC NĂNG CƠ BẢN")
print("=" * 60)

# --- 1. Kiểm tra import thư viện ---
print("\n[1] Kiểm tra thư viện:")


def test_cv2():
    import cv2
    assert cv2.__version__, "OpenCV không có phiên bản"
    print(f"       OpenCV {cv2.__version__}")


def test_numpy():
    import numpy as np
    a = np.zeros((3, 3), dtype=np.uint8)
    assert a.shape == (3, 3)
    print(f"       NumPy {np.__version__}")


def test_pillow():
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (100, 50), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "TEST", fill=(0, 0, 0))
    print(f"       Pillow OK")


def test_tkinter():
    import tkinter
    print(f"       tkinter {tkinter.TkVersion}")


def test_pytesseract_import():
    import pytesseract
    # Kiểm tra Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = r"C:\msys64\ucrt64\bin\tesseract.exe"
    version = pytesseract.get_tesseract_version()
    print(f"       pytesseract + Tesseract {version}")


check("cv2 (OpenCV)", test_cv2)
check("numpy", test_numpy)
check("Pillow", test_pillow)
check("tkinter", test_tkinter)
check("pytesseract + Tesseract", test_pytesseract_import)

# --- 2. Kiểm tra module dự án ---
print("\n[2] Kiểm tra module dự án:")


def test_config():
    import config
    assert hasattr(config, 'PROJECT_DIR')
    assert hasattr(config, 'OCR_ALLOWLIST')


def test_image_io():
    from src.image_io import load_image, save_image
    # Tạo ảnh tạm để test
    import numpy as np
    import cv2
    import tempfile
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :] = [180, 180, 180]  # Xám
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    cv2.imwrite(tmp, img)
    try:
        loaded, scale, err = load_image(tmp)
        assert err is None, f"Lỗi load: {err}"
        assert loaded is not None
    finally:
        os.unlink(tmp)


def test_noise():
    import numpy as np
    from src.noise import add_gaussian_noise, add_salt_pepper_noise, add_gaussian_blur
    img = np.full((60, 200, 3), 128, dtype=np.uint8)
    n1 = add_gaussian_noise(img, sigma=25, seed=42)
    n2 = add_gaussian_noise(img, sigma=25, seed=42)
    assert np.array_equal(n1, n2), "Seed không tái tạo được"
    n_sp = add_salt_pepper_noise(img, ratio=0.05, seed=1)
    assert n_sp.dtype == np.uint8
    b = add_gaussian_blur(img, ksize=5)
    assert b is not None


def test_denoise():
    import numpy as np
    from src.denoise import apply_denoise
    img = np.full((60, 200, 3), 128, dtype=np.uint8)
    for method in ["median", "gaussian", "bilateral"]:
        d = apply_denoise(img, method)
        assert d is not None, f"denoise {method} trả về None"


def test_color_classifier():
    import numpy as np
    import cv2
    from src.color_classifier import classify_plate_color
    # Ảnh trắng
    white = np.full((60, 200, 3), 240, dtype=np.uint8)
    r = classify_plate_color(white)
    assert r["color"] == "Trắng", f"Mong Trắng, nhận {r['color']}"
    # Ảnh vàng
    hsv = np.full((60, 200, 3), [30, 200, 200], dtype=np.uint8)
    yellow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    r2 = classify_plate_color(yellow)
    assert r2["color"] == "Vàng", f"Mong Vàng, nhận {r2['color']}"


def test_layout_classifier():
    from src.layout_classifier import classify_layout
    # Biển rộng → 1 dòng
    r = classify_layout((0, 0, 400, 80), ocr_boxes=[{"bbox": [[10, 20], [45, 20], [45, 55], [10, 55]]}, {"bbox": [[60, 22], [95, 22], [95, 57], [60, 57]]}])
    assert r["layout"] == "1 dòng", f"Mong 1 dòng, nhận {r['layout']}"
    # Biển gần vuông → 2 dòng
    r2 = classify_layout((0, 0, 200, 140), ocr_boxes=[{"bbox": [[10, 15], [40, 15], [40, 45], [10, 45]]}, {"bbox": [[10, 85], [40, 85], [40, 115], [10, 115]]}])
    assert r2["layout"] == "2 dòng", f"Mong 2 dòng, nhận {r2['layout']}"


def test_visualization():
    import numpy as np
    from src.visualization import draw_results, create_comparison_image
    img = np.full((400, 600, 3), 200, dtype=np.uint8)
    plates = [{
        "bbox": (50, 100, 200, 60),
        "plate_text": "30A-12345",
        "color": "Trắng",
        "layout": "1 dòng",
    }]
    result = draw_results(img, plates)
    assert result is not None
    assert result.shape == img.shape


check("config", test_config)
check("image_io", test_image_io)
check("noise", test_noise)
check("denoise", test_denoise)
check("color_classifier", test_color_classifier)
check("layout_classifier", test_layout_classifier)
check("visualization", test_visualization)

# --- 3. Kiểm tra detector (không cần ảnh thật) ---
print("\n[3] Kiểm tra detector:")


def test_detector_import():
    from src.detector import detect_plates_opencv, crop_plate, manual_roi
    import numpy as np
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Vẽ hình chữ nhật trắng giả biển số
    img[150:210, 200:400] = 230
    result = detect_plates_opencv(img)
    # Không yêu cầu phát hiện đúng - chỉ cần không crash
    assert isinstance(result, list)


check("detector", test_detector_import)

# --- Tổng kết ---
print("\n" + "=" * 60)
print(f"  KẾT QUẢ: {PASS} passed, {FAIL} failed / {PASS + FAIL} tests")
if FAIL == 0:
    print("  ✅ Tất cả kiểm thử thành công! Sẵn sàng chạy: python app.py")
else:
    print("  ⚠️  Có lỗi - xem chi tiết ở trên")
print("=" * 60)
print()

# Hướng dẫn chạy app
if FAIL == 0:
    print("Chạy ứng dụng:")
    print("  python app.py")
    print("  hoặc: run.bat")
else:
    failed_libs = []
    try:
        import cv2
    except ImportError:
        failed_libs.append("opencv-python")
    try:
        import numpy
    except ImportError:
        failed_libs.append("numpy")
    try:
        from PIL import Image
    except ImportError:
        failed_libs.append("Pillow")
    try:
        import pytesseract
    except ImportError:
        failed_libs.append("pytesseract")

    if failed_libs:
        print("Cần cài thêm:")
        print(f"  python -m pip install {' '.join(failed_libs)}")
        print("  hoặc với MSYS2: xem README.md phần 'Cài đặt với MSYS2'")
