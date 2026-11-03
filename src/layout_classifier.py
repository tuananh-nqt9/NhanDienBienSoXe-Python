# src/layout_classifier.py
# =========================================================================
# MODULE PHÂN LOẠI BỐ CỤC BIỂN SỐ (1 DÒNG / 2 DÒNG)
# - Kết hợp tỷ lệ w/h của vùng biển
# - Phân bố ký tự theo chiều dọc (từ OCR bounding boxes)
# =========================================================================

import numpy as np
import config


def classify_layout(plate_bbox, ocr_boxes=None):
    """
    Phân loại biển số là 1 dòng hay 2 dòng.

    Kết hợp 3 dấu hiệu:
    1. Tỷ lệ w/h của vùng biển
    2. Phân bố tọa độ Y trung tâm các hộp ký tự
    3. Số nhóm ký tự theo chiều dọc

    Tham số:
        plate_bbox: (x, y, w, h) của vùng biển
        ocr_boxes: danh sách hộp ký tự từ OCR,
                   mỗi hộp là dict có "bbox" = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    Trả về:
        dict chứa:
        - layout: "1 dòng" / "2 dòng" / "Không xác định"
        - evidence: dict giải thích
    """
    _, _, pw, ph = plate_bbox
    aspect_ratio = pw / ph if ph > 0 else 0

    evidence = {
        "aspect_ratio": round(aspect_ratio, 2),
        "plate_width_px": pw,
        "plate_height_px": ph,
    }

    # --- Dấu hiệu 1: Tỷ lệ w/h ---
    # Biển 1 dòng: thường w/h >= 2.5 (chuẩn 520x110mm ~ 4.7)
    # Biển 2 dòng: thường w/h < 2.5 (chuẩn 330x165mm ~ 2.0, hoặc 190x140mm ~ 1.35)
    aspect_vote = None
    if aspect_ratio >= config.LAYOUT_ASPECT_THRESHOLD:
        aspect_vote = "1 dòng"
    else:
        aspect_vote = "2 dòng"

    evidence["aspect_vote"] = aspect_vote

    # --- Dấu hiệu 2: Phân bố ký tự theo chiều dọc ---
    char_vote = None
    n_lines_detected = 0

    if ocr_boxes and len(ocr_boxes) >= 1:
        # Nếu có 1 hộp ký tự nhưng chứa từ 2 ký tự trở lên (toàn bộ dòng nằm trong 1 box)
        if len(ocr_boxes) == 1:
            txt = ocr_boxes[0].get("text", "")
            if len(txt) >= 2:
                n_lines_detected = 1
                # Chỉ vote 1 dòng nếu tỷ lệ khung không phải biển vuông 2 dòng
                char_vote = "1 dòng" if aspect_ratio >= 2.3 else "2 dòng"
        else:
            # Có từ 2 hộp ký tự trở lên: phân tích khoảng cách trục Y
            y_centers = []
            for box_info in ocr_boxes:
                box = box_info["bbox"]
                ys = [pt[1] for pt in box]
                y_center = sum(ys) / len(ys)
                y_centers.append(y_center)

            y_centers = np.array(y_centers)
            y_sorted = np.sort(y_centers)
            gaps = np.diff(y_sorted)
            max_gap = np.max(gaps) if len(gaps) > 0 else 0
            gap_ratio = max_gap / ph if ph > 0 else 0

            evidence["max_vertical_gap_ratio"] = round(gap_ratio, 3)

            if gap_ratio > config.LAYOUT_VERTICAL_GAP_RATIO:
                n_lines_detected = 2
                char_vote = "2 dòng"
            else:
                n_lines_detected = 1
                char_vote = "1 dòng"

    evidence["char_vote"] = char_vote
    evidence["n_lines_detected"] = n_lines_detected

    # --- Kết hợp các dấu hiệu ---
    votes = [v for v in [aspect_vote, char_vote] if v is not None]

    if not votes:
        layout = "Không xác định"
    elif len(set(votes)) == 1:
        layout = votes[0]
    else:
        # Nếu có bằng chứng từ ký tự, ưu tiên ký tự
        if char_vote is not None and n_lines_detected > 0:
            layout = char_vote
        elif aspect_vote is not None:
            layout = aspect_vote
        else:
            layout = "Không xác định"

    evidence["final"] = layout
    evidence["votes"] = votes
    return {
        "layout": layout,
        "evidence": evidence,
    }


def group_chars_by_line(ocr_boxes, plate_height):
    """
    Nhóm các ký tự OCR theo dòng (trên/dưới).

    Tham số:
        ocr_boxes: danh sách hộp ký tự
        plate_height: chiều cao vùng biển

    Trả về:
        danh sách các dòng, mỗi dòng là danh sách ký tự
        đã sắp xếp từ trái sang phải
    """
    if not ocr_boxes:
        return []

    # Tính Y trung tâm cho mỗi hộp
    items = []
    for box_info in ocr_boxes:
        box = box_info["bbox"]
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        y_center = sum(ys) / len(ys)
        x_center = sum(xs) / len(xs)
        items.append({
            "y_center": y_center,
            "x_center": x_center,
            "info": box_info,
        })

    # Sắp xếp theo Y center
    items.sort(key=lambda it: it["y_center"])

    # Phân nhóm: nếu khoảng cách Y > ngưỡng → dòng mới
    lines = [[items[0]]]
    threshold = plate_height * config.LAYOUT_VERTICAL_GAP_RATIO

    for i in range(1, len(items)):
        if items[i]["y_center"] - items[i-1]["y_center"] > threshold:
            lines.append([items[i]])
        else:
            lines[-1].append(items[i])

    # Sắp xếp ký tự trong mỗi dòng từ trái sang phải
    result = []
    for line in lines:
        line.sort(key=lambda it: it["x_center"])
        result.append([it["info"] for it in line])

    return result
