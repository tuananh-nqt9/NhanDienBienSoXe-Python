# src/color_classifier.py
# =========================================================================
# MODULE PHÂN LOẠI MÀU NỀN BIỂN SỐ
# - Phân loại 4 màu: Trắng, Vàng, Xanh dương, Đỏ
# - Dùng không gian màu HSV
# - Loại bỏ viền và ký tự trước khi phân loại
# =========================================================================

import cv2
import numpy as np
import config


def classify_plate_color(plate_bgr):
    """
    Phân loại màu nền biển số từ ảnh màu.

    Quy trình:
    1. Lấy phần trung tâm (loại viền)
    2. Chuyển sang HSV
    3. Tạo mặt nạ cho từng màu
    4. Tính tỷ lệ pixel mỗi màu
    5. Kết luận hoặc trả về "Không xác định"

    Tham số:
        plate_bgr: ảnh vùng biển số (BGR)

    Trả về:
        dict chứa:
        - color: tên màu tiếng Việt
        - ratios: dict tỷ lệ pixel từng màu
        - confident: True/False
    """
    h, w = plate_bgr.shape[:2]

    # Bước 1: Lấy phần trung tâm để loại viền
    # Viền biển số thường khác màu nền
    inner_ratio = config.COLOR_INNER_RATIO
    margin_x = int(w * (1 - inner_ratio) / 2)
    margin_y = int(h * (1 - inner_ratio) / 2)

    # Đảm bảo vùng trung tâm có kích thước hợp lý
    margin_x = max(margin_x, 2)
    margin_y = max(margin_y, 2)

    inner = plate_bgr[margin_y:h-margin_y, margin_x:w-margin_x]

    if inner.size == 0:
        return {
            "color": "Không xác định",
            "ratios": {},
            "confident": False
        }

    # Bước 2: Chuyển sang HSV
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    if total_pixels == 0:
        return {
            "color": "Không xác định",
            "ratios": {},
            "confident": False
        }

    # Bước 3: Tạo mặt nạ cho từng màu
    ratios = {}

    # --- Trắng: bão hòa thấp + sáng cao ---
    mask_white = cv2.inRange(
        hsv,
        np.array([0, 0, config.WHITE_V_MIN]),
        np.array([179, config.WHITE_S_MAX, 255])
    )
    ratios["Trắng"] = np.count_nonzero(mask_white) / total_pixels

    # --- Vàng ---
    mask_yellow = cv2.inRange(
        hsv,
        np.array([config.YELLOW_H_MIN, config.YELLOW_S_MIN, config.YELLOW_V_MIN]),
        np.array([config.YELLOW_H_MAX, 255, 255])
    )
    ratios["Vàng"] = np.count_nonzero(mask_yellow) / total_pixels

    # --- Xanh dương ---
    mask_blue = cv2.inRange(
        hsv,
        np.array([config.BLUE_H_MIN, config.BLUE_S_MIN, config.BLUE_V_MIN]),
        np.array([config.BLUE_H_MAX, 255, 255])
    )
    ratios["Xanh dương"] = np.count_nonzero(mask_blue) / total_pixels

    # --- Đỏ: hai khoảng Hue vì đỏ nằm ở biên 0 và 179 ---
    mask_red1 = cv2.inRange(
        hsv,
        np.array([config.RED_H_MIN1, config.RED_S_MIN, config.RED_V_MIN]),
        np.array([config.RED_H_MAX1, 255, 255])
    )
    mask_red2 = cv2.inRange(
        hsv,
        np.array([config.RED_H_MIN2, config.RED_S_MIN, config.RED_V_MIN]),
        np.array([config.RED_H_MAX2, 255, 255])
    )
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    ratios["Đỏ"] = np.count_nonzero(mask_red) / total_pixels

    # Bước 4: Xác định màu chiếm ưu thế
    # Sắp xếp theo tỷ lệ giảm dần
    sorted_colors = sorted(ratios.items(), key=lambda x: x[1], reverse=True)

    best_color, best_ratio = sorted_colors[0]
    second_color, second_ratio = sorted_colors[1] if len(sorted_colors) > 1 else ("", 0)

    # Kiểm tra tỷ lệ tối thiểu
    if best_ratio < config.COLOR_MIN_RATIO:
        return {
            "color": "Không xác định",
            "ratios": ratios,
            "confident": False
        }

    # Kiểm tra khoảng cách giữa hai màu hàng đầu
    # Nếu quá gần nhau → không đủ tin cậy
    diff = best_ratio - second_ratio
    if diff < config.COLOR_DIFF_THRESHOLD:
        return {
            "color": "Không xác định",
            "ratios": ratios,
            "confident": False
        }

    return {
        "color": best_color,
        "ratios": ratios,
        "confident": True
    }
