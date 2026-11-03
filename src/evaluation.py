# src/evaluation.py
# =========================================================================
# MODULE ĐÁNH GIÁ VÀ XUẤT KẾT QUẢ
# - Xuất CSV UTF-8 BOM (mở tiếng Việt bằng Excel)
# - In tóm tắt ra terminal
# - So sánh kết quả các trạng thái (gốc / nhiễu / giảm nhiễu)
# =========================================================================

import os
import csv
import time
import config


def export_csv(results_list, filepath):
    """
    Xuất kết quả ra file CSV (UTF-8 BOM để Excel hiển thị tiếng Việt).

    Tham số:
        results_list: danh sách dict kết quả
        filepath: đường dẫn file CSV

    Trả về:
        (success, error_message)
    """
    if not results_list:
        return False, "Không có kết quả để xuất."

    # Tạo thư mục cha nếu chưa có
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Kiểm tra file đã tồn tại
    if os.path.isfile(filepath):
        # Thêm timestamp vào tên file để tránh ghi đè
        base, ext = os.path.splitext(filepath)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"{base}_{timestamp}{ext}"

    # Định nghĩa các cột CSV
    fieldnames = [
        "ten_anh",             # Tên ảnh
        "ma_bien",             # Mã vùng biển (nếu nhiều biển)
        "che_do_phat_hien",    # auto / manual
        "trang_thai_anh",      # gốc / nhiễu / giảm nhiễu
        "ocr_goc",             # OCR nguyên bản
        "ocr_chuan_hoa",       # OCR chuẩn hóa
        "mau_nen",             # Màu nền biển
        "so_dong",             # 1 dòng / 2 dòng
        "diem_ocr",            # Điểm tin cậy OCR
        "loai_nhieu",          # Loại nhiễu
        "thong_so_nhieu",      # Tham số nhiễu
        "seed",                # Random seed
        "bo_loc",              # Bộ lọc giảm nhiễu
        "thong_so_bo_loc",     # Tham số bộ lọc
        "thoi_gian_xu_ly",     # Thời gian (giây)
        "trang_thai",          # Thành công / Thất bại
        "ghi_chu",             # Nguyên nhân thất bại
    ]

    try:
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results_list:
                row = {field: result.get(field, "") for field in fieldnames}
                writer.writerow(row)

        return True, filepath

    except Exception as e:
        return False, f"Lỗi khi xuất CSV: {str(e)}"


def format_result_for_csv(image_name, plate_result, state="gốc",
                          noise_info=None, denoise_info=None):
    """
    Chuyển đổi kết quả xử lý thành dict phù hợp cho CSV.

    Tham số:
        image_name: tên file ảnh
        plate_result: dict kết quả từ pipeline
        state: "gốc" / "nhiễu" / "giảm nhiễu"
        noise_info: dict thông tin nhiễu (nếu có)
        denoise_info: dict thông tin giảm nhiễu (nếu có)

    Trả về:
        dict phù hợp cho CSV
    """
    plate_text = plate_result.get("plate_text", "")
    has_error = plate_result.get("error", None)

    row = {
        "ten_anh": image_name,
        "ma_bien": plate_result.get("plate_idx", 0),
        "che_do_phat_hien": plate_result.get("method", ""),
        "trang_thai_anh": state,
        "ocr_goc": plate_result.get("raw_text", ""),
        "ocr_chuan_hoa": plate_text,
        "mau_nen": plate_result.get("color", ""),
        "so_dong": plate_result.get("layout", ""),
        "diem_ocr": plate_result.get("ocr_confidence", 0),
        "loai_nhieu": "",
        "thong_so_nhieu": "",
        "seed": "",
        "bo_loc": "",
        "thong_so_bo_loc": "",
        "thoi_gian_xu_ly": plate_result.get("processing_time", 0),
        "trang_thai": "Thất bại" if has_error else "Thành công",
        "ghi_chu": has_error or "",
    }

    # Thêm thông tin nhiễu
    if noise_info:
        row["loai_nhieu"] = noise_info.get("type", "")
        row["thong_so_nhieu"] = str(noise_info.get("params", ""))
        row["seed"] = noise_info.get("seed", "")

    # Thêm thông tin giảm nhiễu
    if denoise_info:
        row["bo_loc"] = denoise_info.get("method", "")
        row["thong_so_bo_loc"] = str(denoise_info.get("params", ""))

    return row


def print_summary(plate_result, detection_method="auto"):
    """
    In tóm tắt kết quả ra terminal.
    """
    print("\n" + "=" * 50)
    print("  KẾT QUẢ NHẬN DIỆN BIỂN SỐ XE")
    print("=" * 50)
    print(f"  Biển số    : {plate_result.get('plate_text', 'N/A')}")
    print(f"  Màu nền    : {plate_result.get('color', 'N/A')}")
    print(f"  Bố cục     : {plate_result.get('layout', 'N/A')}")
    print(f"  Phát hiện  : {'Thủ công' if detection_method == 'manual' else 'Tự động'}")
    print(f"  Điểm OCR   : {plate_result.get('ocr_confidence', 0):.3f}")
    print(f"  Thời gian  : {plate_result.get('processing_time', 0):.3f}s")

    if plate_result.get("error"):
        print(f"  Lỗi        : {plate_result['error']}")

    print("=" * 50)
