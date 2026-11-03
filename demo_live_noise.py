# demo_live_noise.py
# =========================================================================
# SCRIPT DEMO TẠO NHIỄU TRỰC TIẾP (LIVE NOISE INJECTION) QUA PHÍM TẮT
# =========================================================================
# Mục đích:
# - Kiểm tra độ bền vững (robustness) và giới hạn nhận diện tại chỗ khi hội đồng thử thách.
# - Dùng 1 phím bấm duy nhất để biến ảnh thành ảnh nhiễu và chạy pipeline nhận diện tức thì.
#
# Cách sử dụng:
#   python demo_live_noise.py [duong_dan_anh]
#
# Phím tắt điều khiển:
#   [B] - Làm mờ ảnh (Gaussian Blur)
#   [N] - Thêm nhiễu hạt cảm biến (Gaussian Noise)
#   [S] - Thêm nhiễu hạt muối tiêu (Salt & Pepper)
#   [D] - Giảm sáng mô phỏng ban đêm (Night Mode)
#   [G] - Khử/Giảm nhiễu (Denoise)
#   [R] - Khôi phục ảnh gốc (Reset)
#   [Q] hoặc [ESC] - Thoát chương trình
# =========================================================================

import os
import sys
import time
import cv2
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.image_io import load_image
from src.noise import apply_noise
from src.denoise import apply_median_blur, apply_bilateral_filter
from src.pipeline import process_image
from src.ocr_engine import OCREngine
from src.visualization import draw_results


def find_default_image():
    """Tìm ảnh mẫu có sẵn trong thư mục samples."""
    samples_dir = config.SAMPLES_DIR
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.lower().endswith(config.SUPPORTED_FORMATS):
                return os.path.join(samples_dir, f)
    return None


def add_hud_banner(image, current_mode, last_result=None, fps_ms=0):
    """Vẽ thanh thông tin trạng thái và phím tắt lên trên ảnh hiển thị."""
    h, w = image.shape[:2]
    # Tạo banner trên đầu ảnh
    banner_height = 85
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)  # Nền xám đen

    # Dòng 1: Hướng dẫn phím tắt
    text_shortcuts = "[B] Mo  |  [N] Hat  |  [S] Muoi tieu  |  [D] Ban dem  |  [G] Giam nhieu  |  [R] Goc  |  [Q] Thoat"
    cv2.putText(banner, text_shortcuts, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # Dòng 2: Trạng thái hiện tại
    text_mode = f"Che do: {current_mode}  |  Xu ly: {fps_ms:.0f}ms"
    cv2.putText(banner, text_mode, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Dòng 3: Kết quả nhận diện
        p = last_result["plates"][0]
        p_text = p.get('plate_text', p.get('text', 'N/A'))
        conf = p.get('ocr_confidence', 0)
        text_plate = f"Bien so: {p_text} (OCR: {conf:.2f})  |  Mau: {p.get('color', 'N/A')}"
        cv2.putText(banner, text_plate, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        text_plate = "Ket qua: Khong tim thay bien so"
        cv2.putText(banner, text_plate, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # Ghép banner và ảnh
    return np.vstack([banner, image])


def main():
    print("=" * 65)
    print("  CHUONG TRINH DEMO TAO NHIEU TRUC TIEP (LIVE NOISE INJECTION)")
    print("=" * 65)

    # Xác định đường dẫn ảnh
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
    else:
        image_path = find_default_image()

    if not image_path or not os.path.exists(image_path):
        print("❌ Khong tim thay anh dau vao. Hay truyen duong dan anh: python demo_live_noise.py [file.jpg]")
        return

    print(f"📷 Dang tai anh: {image_path}")
    original_img, scale, err = load_image(image_path)
    if err or original_img is None:
        print(f"❌ Loi tai anh: {err}")
        return

    print("🧠 Dang khoi tao OCR engine...")
    ocr = OCREngine()
    ocr.initialize()

    current_img = original_img.copy()
    current_mode = "Anh goc (Chua nhieu)"

    window_name = "Live Noise Injection Demo - Nhan dien bien so xe"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "-" * 65)
    print("  PHIM TAT DIEU KHIEN:")
    print("    [B] : Lam mo anh (Gaussian Blur, k=11)")
    print("    [N] : Them nhieu hat cam bien (Gaussian Noise, sigma=30)")
    print("    [S] : Them nhieu hat muoi tieu (Salt & Pepper, 8%)")
    print("    [D] : Giam sang mo phong ban dem (Night Mode, 35% sang)")
    print("    [G] : Giam nhieu (Median / Bilateral filter)")
    print("    [R] : Khoi phuc anh goc")
    print("    [Q] / [ESC] : Thoat")
    print("-" * 65 + "\n")

    def run_inference(img, mode_desc):
        t0 = time.time()
        res = process_image(img, ocr_engine=ocr, save_debug=False)
        duration = (time.time() - t0) * 1000

        # In log ra terminal
        n_plates = len(res.get("plates", []))
        if n_plates > 0:
            p = res["plates"][0]
            print(f"👉 [{mode_desc}] -> Bien: {p.get('text')} | Mau: {p.get('color')} | Bo cuc: {p.get('layout')} ({duration:.0f}ms)")
        else:
            print(f"👉 [{mode_desc}] -> Khong tim thay bien so ({duration:.0f}ms)")

        annotated = draw_results(img, res.get("plates", []))
        display_frame = add_hud_banner(annotated, mode_desc, res, duration)
        cv2.imshow(window_name, display_frame)
        return res

    # Chạy lần đầu với ảnh gốc
    last_res = run_inference(current_img, current_mode)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("\n👋 Dang thoat chuong trinh...")
            break

        elif key in (ord('b'), ord('B')):
            current_mode = "Lam mo (Gaussian Blur, k=11)"
            current_img = apply_noise(original_img, "blur", {"ksize": 11})
            run_inference(current_img, current_mode)

        elif key in (ord('n'), ord('N')):
            current_mode = "Nhieu hat Gaussian (sigma=30)"
            current_img = apply_noise(original_img, "gaussian", {"sigma": 30, "seed": 42})
            run_inference(current_img, current_mode)

        elif key in (ord('s'), ord('S')):
            current_mode = "Nhieu muoi tieu (Salt&Pepper 8%)"
            current_img = apply_noise(original_img, "salt_pepper", {"ratio": 0.08, "seed": 42})
            run_inference(current_img, current_mode)

        elif key in (ord('d'), ord('D')):
            current_mode = "Ban dem / Thieu sang (Factor=0.35)"
            current_img = apply_noise(original_img, "brightness", {"factor": 0.35})
            run_inference(current_img, current_mode)

        elif key in (ord('g'), ord('G')):
            current_mode = "Giam nhieu (Median 5x5 + Bilateral)"
            # Áp dụng bộ lọc giảm nhiễu
            denoised = apply_median_blur(current_img, ksize=5)
            denoised = apply_bilateral_filter(denoised, d=9, sigma_color=75, sigma_space=75)
            current_img = denoised
            run_inference(current_img, current_mode)

        elif key in (ord('r'), ord('R')):
            current_mode = "Anh goc (Reset)"
            current_img = original_img.copy()
            run_inference(current_img, current_mode)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
