# app.py
# =========================================================================
# CHƯƠNG TRÌNH CHÍNH - NHẬN DIỆN VÀ PHÂN LOẠI BIỂN SỐ XE TỪ ẢNH TĨNH
#
# Cách chạy:
#   python app.py
#
# Yêu cầu:
#   - Python 3.10+
#   - Cài requirements: pip install -r requirements.txt
#   - Lần đầu chạy cần Internet để tải trọng số EasyOCR (~100MB)
# =========================================================================

import os
import sys
import tkinter as tk

# Đảm bảo thư mục dự án nằm trong PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.gui import MainApp


def main():
    """Khởi chạy ứng dụng."""
    # Tạo thư mục outputs nếu chưa có
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)

    print("=" * 60)
    print("  NHẬN DIỆN VÀ PHÂN LOẠI BIỂN SỐ XE TỪ ẢNH TĨNH")
    print("=" * 60)
    print(f"  Thư mục dự án : {config.PROJECT_DIR}")
    print(f"  Thư mục output: {config.OUTPUT_DIR}")
    print(f"  Python         : {sys.version}")
    print("=" * 60)
    print()

    # Khởi tạo cửa sổ tkinter
    root = tk.Tk()

    # Đặt icon nếu có
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    # Tạo ứng dụng
    app = MainApp(root)

    # Chạy vòng lặp sự kiện
    root.mainloop()


if __name__ == "__main__":
    main()
