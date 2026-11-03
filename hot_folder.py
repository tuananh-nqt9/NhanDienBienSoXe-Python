# hot_folder.py
# =========================================================================
# CHƯƠNG TRÌNH THEO DÕI THƯ MỤC NÓNG (HOT FOLDER WATCHER)
# =========================================================================
# Mục đích:
# - Tự động phát hiện ảnh mới khi giáo viên/hội đồng nạp ảnh qua USB hoặc Zalo.
# - Tự động chạy pipeline nhận diện biển số và hiển thị kết quả lên màn hình.
# - Thể hiện tư duy tự động hóa chuyên nghiệp, tránh lúng túng khi đổi ảnh đột xuất.
#
# Cách sử dụng:
#   python hot_folder.py
#
# Thao tác:
#   Chỉ cần copy hoặc thả file ảnh (.jpg, .jpeg, .png, .bmp, .webp)
#   vào thư mục "test_live/".
# =========================================================================

import os
import sys
import time
import shutil
import cv2

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.image_io import load_image
from src.pipeline import process_image
from src.ocr_engine import OCREngine
from src.visualization import draw_results


def main():
    print("=" * 65)
    print("  HE THONG THEO DOI THU MUC NONG (HOT FOLDER WATCHER)")
    print("=" * 65)

    hot_dir = config.HOT_FOLDER_DIR
    out_dir = config.LIVE_OUTPUT_DIR
    os.makedirs(hot_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"📂 Thu muc theo doi: {hot_dir}")
    print(f"💾 Thu muc luu ket qua: {out_dir}")
    print("\n🧠 Dang khoi tao OCR engine...")
    ocr = OCREngine()
    ocr.initialize()

    # Ghi nhận các file đã có từ trước
    seen_files = set()
    for f in os.listdir(hot_dir):
        if f.lower().endswith(config.SUPPORTED_FORMATS):
            seen_files.add(f)

    print("\n" + "-" * 65)
    print(f"✅ HE THONG DANG HOAT DONG! (Da ghi nhan {len(seen_files)} file cu)")
    print("👉 Hay copy / tha anh bat ky vao thu muc 'test_live/' de demo...")
    print("👉 Nhan Ctrl + C de dung theo doi.")
    print("-" * 65 + "\n")

    window_name = "Hot Folder Demo - Ket qua Nhan dien tu dong"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Kiểm tra phím tắt OpenCV (nếu người dùng bấm Q để đóng cửa sổ kết quả)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                cv2.destroyAllWindows()

            current_files = os.listdir(hot_dir)
            for fname in current_files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in config.SUPPORTED_FORMATS and fname not in seen_files:
                    full_path = os.path.join(hot_dir, fname)

                    # Đợi một chút để file copy hoàn tất (tránh đọc khi file đang ghi)
                    time.sleep(0.3)
                    if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                        continue

                    seen_files.add(fname)
                    print(f"\n🔥 [PHAT HIEN ANH MOI]: {fname}")
                    print(f"   Dang doc va chay nhan dien...")

                    # Tải ảnh
                    img, scale, err = load_image(full_path)
                    if err or img is None:
                        print(f"   ❌ Loi doc anh: {err}")
                        continue

                    # Chạy pipeline nhận diện
                    t0 = time.time()
                    res = process_image(img, ocr_engine=ocr, save_debug=False)
                    duration = (time.time() - t0) * 1000

                    plates = res.get("plates", [])
                    print(f"   ⏱️ Thoi gian xu ly: {duration:.0f} ms")

                    if plates:
                        print(f"   🎉 Tim thay {len(plates)} bien so:")
                        for idx, p in enumerate(plates, 1):
                            print(f"      [{idx}] Bien so : {p.get('text', 'N/A')}")
                            print(f"          Mau nen : {p.get('color', 'N/A')}")
                            print(f"          Bo cuc  : {p.get('layout', 'N/A')}")
                            print(f"          Do tin cay: {p.get('confidence', 0):.2f}")
                    else:
                        print("   ⚠️ Khong tim thay bien so tren anh.")

                    # Vẽ kết quả
                    annotated = draw_results(img, plates)

                    # Lưu ảnh kết quả vào outputs/live/
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    out_filename = f"result_{timestamp}_{fname}"
                    out_path = os.path.join(out_dir, out_filename)
                    cv2.imwrite(out_path, annotated)
                    print(f"   💾 Da luu ket qua: {out_path}")

                    # Bung kết quả lên màn hình
                    cv2.imshow(window_name, annotated)
                    cv2.waitKey(100)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n⏹️ Da dung chuong trinh Hot Folder.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
