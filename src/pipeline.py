# src/pipeline.py
# =========================================================================
# MODULE PIPELINE - LUỒNG XỬ LÝ CHÍNH
# Kết nối tất cả module thành luồng hoàn chỉnh:
# Ảnh → Phát hiện biển → Cắt → Phân loại màu → OCR → Hiển thị
#
# Hỗ trợ 3 chế độ:
#   1. "auto"        - Phát hiện tự động từ ảnh toàn xe
#   2. "manual"      - Người dùng chỉ tọa độ vùng biển
#   3. "full_image"  - Ảnh đã cắt sát biển, dùng toàn ảnh làm ROI
# =========================================================================

import os
import json
from datetime import datetime
import time
import cv2
import numpy as np
import config
from src.detector import detect_plates_opencv, crop_plate, manual_roi, full_image_as_plate
from src.preprocessing import create_ocr_variants
from src.color_classifier import classify_plate_color
from src.layout_classifier import classify_layout
from src.ocr_engine import OCREngine


def process_image(img_bgr, ocr_engine=None, manual_bbox=None,
                  use_full_image=False, save_debug=False):
    """
    Xử lý toàn bộ pipeline cho một ảnh.

    Luồng:
    1. Phát hiện vùng biển số (tự động / thủ công / toàn ảnh)
    2. Cắt vùng biển
    3. Phân loại màu (trên ảnh MÀU)
    4. Tiền xử lý phục vụ OCR
    5. Nhận diện ký tự (Tesseract)
    6. Phân loại bố cục (1/2 dòng)
    7. Tổng hợp kết quả

    Tham số:
        img_bgr: ảnh BGR đã chuẩn hóa
        ocr_engine: OCREngine instance (tránh khởi tạo lại)
        manual_bbox: (x, y, w, h) nếu chọn vùng thủ công
        use_full_image: True = coi toàn ảnh là biển (ảnh đã cắt sẵn)
        save_debug: True = lưu ảnh ROI và tiền xử lý vào outputs/debug/

    Trả về:
        dict chứa:
        - plates: danh sách kết quả cho từng biển
        - total_time: tổng thời gian xử lý
        - n_plates: số biển phát hiện được
        - detection_method: "auto" / "manual" / "full_image"
    """
    start_time = time.time()
    debug_dir = _create_debug_dir() if save_debug else None
    detector_debug = None

    # Khởi tạo OCR engine nếu chưa có
    if ocr_engine is None:
        ocr_engine = OCREngine()
        ocr_engine.initialize()

    # Bước 1: Phát hiện biển số
    if use_full_image:
        # Chế độ 3: Toàn ảnh là biển - dùng khi ảnh đã được cắt sát biển
        candidate = full_image_as_plate(img_bgr)
        candidates = [candidate]
        detection_method = "full_image"

    elif manual_bbox is not None:
        # Chế độ 2: Thủ công
        x, y, w, h = manual_bbox
        candidate = manual_roi(img_bgr, x, y, w, h)
        if candidate is None:
            return {
                "plates": [],
                "total_time": time.time() - start_time,
                "n_plates": 0,
                "detection_method": "manual",
                "error": "Vùng chọn thủ công không hợp lệ (quá nhỏ).",
            }
        candidates = [candidate]
        detection_method = "manual"

    else:
        # Chế độ 1: Phát hiện tự động
        detection_output = detect_plates_opencv(img_bgr, return_debug=save_debug)
        if save_debug:
            candidates, detector_debug = detection_output
            _save_debug_json(detector_debug, "detector_trace", debug_dir)
        else:
            candidates = detection_output
        detection_method = "auto"

        if not candidates:
            return {
                "plates": [],
                "total_time": time.time() - start_time,
                "n_plates": 0,
                "detection_method": "auto",
                "error": (
                    "Không phát hiện được biển số tự động. "
                    "Gợi ý: chọn 'Ảnh biển đã cắt' nếu ảnh đã cắt sát biển, "
                    "hoặc dùng 'Chọn vùng thủ công' để chỉ rõ vùng biển."
                ),
            }

    # Xử lý từng biển số
    plates_results = []
    for idx, candidate in enumerate(candidates):
        plate_result = _process_single_plate(
            img_bgr, candidate, ocr_engine, idx, save_debug=save_debug, debug_dir=debug_dir
        )
        plates_results.append(plate_result)

    # Nếu phát hiện nhiều biển ở chế độ tự động, lọc bỏ vùng nhiễu (lưới tản nhiệt, cản xe, viền ảnh)
    # và ưu tiên xếp hạng ứng viên có cấu trúc biển số Việt Nam chuẩn (có cả chữ cái + chữ số, 6-10 ký tự)
    if len(plates_results) > 1 and detection_method == "auto":
        img_area_total = max(1, img_bgr.shape[0] * img_bgr.shape[1])

        def _is_strong_plate(p):
            n_let = len(p.get("letters", ""))
            n_dig = len(p.get("digits", ""))
            total_c = n_let + n_dig
            return n_let >= 1 and n_dig >= 4 and 6 <= total_c <= 10 and p.get("ocr_confidence", 0) >= 0.30

        strong_plates = [p for p in plates_results if _is_strong_plate(p)]
        if strong_plates:
            # Nếu có biển đạt chuẩn nằm gọn trên xe, loại bỏ khung bao toàn ảnh (>80% diện tích ảnh)
            non_full_strong = [
                p for p in strong_plates
                if (p["bbox"][2] * p["bbox"][3]) / img_area_total <= 0.80
            ]
            plates_results = non_full_strong if non_full_strong else strong_plates
        else:
            valid_plates = [
                p for p in plates_results
                if (len(p.get("letters", "") + p.get("digits", "")) >= 4 and p.get("ocr_confidence", 0) >= 0.20)
            ]
            if valid_plates:
                plates_results = valid_plates

        # Sắp xếp ưu tiên biển có độ tin cậy OCR và số ký tự hợp lệ cao nhất lên đầu
        plates_results.sort(
            key=lambda p: (
                1 if _is_strong_plate(p) else 0,
                len(p.get("letters", "")) >= 1 and len(p.get("digits", "")) >= 4,
                round(p.get("ocr_confidence", 0), 2),
                p.get("detection_score", 0),
            ),
            reverse=True,
        )
        for new_idx, p in enumerate(plates_results):
            p["plate_idx"] = new_idx

    total_time = time.time() - start_time

    return {
        "plates": plates_results,
        "total_time": round(total_time, 3),
        "n_plates": len(plates_results),
        "detection_method": detection_method,
        "debug_dir": debug_dir,
        "detector_debug": detector_debug,
    }


def _process_single_plate(img_bgr, candidate, ocr_engine, plate_idx,
                           save_debug=False, debug_dir=None):
    """
    Xử lý một biển số đơn lẻ.

    Trả về dict chứa mọi thông tin về biển số.
    """
    start_time = time.time()
    bbox = candidate["bbox"]
    method = candidate.get("method", "auto")

    # Bước 2: Cắt vùng biển
    # Với full_image, không thêm lề (tránh cắt ngoài biên)
    margin = 0.0 if method == "full_image" else None
    plate_bgr, adjusted_bbox = crop_plate(img_bgr, bbox, margin_ratio=margin)

    if plate_bgr.size == 0:
        return _empty_plate_result(bbox, method, plate_idx,
                                   "Vùng biển rỗng sau khi cắt.")

    # Lưu ảnh debug nếu cần
    if save_debug:
        _save_debug_image(plate_bgr, f"roi_{plate_idx}", debug_dir)

    # Bước 3: Phân loại màu (dùng ảnh MÀU, trước khi chuyển xám)
    # QUAN TRỌNG: phân loại màu trên vùng màu đầy đủ, KHÔNG trên ảnh nhị phân
    color_result = classify_plate_color(plate_bgr)

    # Bước 4: Tiền xử lý phục vụ OCR
    variants = create_ocr_variants(plate_bgr)

    # Lưu ảnh tiền xử lý debug
    if save_debug and variants:
        for var_name, var_img in variants:
            _save_debug_image(var_img, f"preproc_{plate_idx}_{var_name}", debug_dir)

    # Bước 5: Nhận diện ký tự
    plate_gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    try:
        ocr_result = ocr_engine.read_plate(plate_gray, adjusted_bbox, variants)
    except Exception as e:
        # Ghi rõ lỗi OCR, không âm thầm trả về rỗng
        ocr_result = {
            "raw_text": "",
            "normalized_text": f"[Lỗi OCR: {e}]",
            "lines": [],
            "letters": "",
            "digits": "",
            "avg_confidence": 0.0,
            "best_variant": "",
            "all_attempts": [],
            "processing_time": 0.0,
            "ocr_boxes": [],
            "error": str(e),
        }
        print(f"[PIPELINE] Lỗi OCR biển {plate_idx}: {e}")

    # Bước 6: Phân loại bố cục
    # Dùng adjusted_bbox vì đó là kích thước thực tế của vùng ảnh đã cắt
    layout_result = classify_layout(adjusted_bbox, ocr_result.get("ocr_boxes"))

    processing_time = time.time() - start_time

    # Tổng hợp
    plate_text = ocr_result.get("normalized_text", "Không đọc được")
    color_name = color_result.get("color", "Không xác định")
    layout_name = layout_result.get("layout", "Không xác định")

    if save_debug:
        _save_debug_json({
            "candidate": candidate,
            "bbox": bbox,
            "adjusted_bbox": adjusted_bbox,
            "color": color_result,
            "layout": layout_result,
            "ocr": ocr_result,
        }, f"plate_{plate_idx}_result", debug_dir)

    return {
        "plate_idx": plate_idx,
        "bbox": bbox,
        "adjusted_bbox": adjusted_bbox,
        "method": method,
        "detection_score": candidate.get("score", -1),
        "plate_bgr": plate_bgr,
        "plate_text": plate_text,
        "raw_text": ocr_result.get("raw_text", ""),
        "ocr_lines": ocr_result.get("lines", []),
        "letters": ocr_result.get("letters", ""),
        "digits": ocr_result.get("digits", ""),
        "ocr_confidence": ocr_result.get("avg_confidence", 0),
        "best_variant": ocr_result.get("best_variant", ""),
        "color": color_name,
        "color_ratios": color_result.get("ratios", {}),
        "color_confident": color_result.get("confident", False),
        "layout": layout_name,
        "layout_evidence": layout_result.get("evidence", {}),
        "processing_time": round(processing_time, 3),
        "ocr_boxes": ocr_result.get("ocr_boxes", []),
        "ocr_error": ocr_result.get("error"),
    }


def _empty_plate_result(bbox, method, plate_idx, error_msg):
    """Tạo kết quả rỗng khi không xử lý được biển số."""
    return {
        "plate_idx": plate_idx,
        "bbox": bbox,
        "method": method,
        "plate_text": "Không đọc được",
        "raw_text": "",
        "ocr_lines": [],
        "letters": "",
        "digits": "",
        "ocr_confidence": 0,
        "color": "Không xác định",
        "layout": "Không xác định",
        "processing_time": 0,
        "error": error_msg,
    }


def _create_debug_dir():
    """Create a unique folder so a later run cannot overwrite its evidence."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_dir = os.path.join(config.OUTPUT_DIR, "debug", run_id)
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _save_debug_json(payload, name, debug_dir):
    if not debug_dir:
        return
    try:
        path = os.path.join(debug_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        print(f"[DEBUG] Saved data: {path}")
    except Exception as e:
        print(f"[DEBUG] Could not save data: {e}")


def _save_debug_image(img, name, debug_dir=None):
    """Lưu ảnh debug vào outputs/debug/."""
    try:
        if debug_dir is None:
            debug_dir = os.path.join(config.OUTPUT_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, f"{name}.png")
        cv2.imwrite(path, img)
        print(f"[DEBUG] Lưu ảnh: {path}")
    except Exception as e:
        print(f"[DEBUG] Không lưu được ảnh debug: {e}")


def process_plate_region(plate_bgr, plate_bbox, ocr_engine=None,
                         save_debug=False):
    """
    Xử lý riêng một vùng biển đã cắt (dùng cho chế độ thử nghiệm vùng biển).

    Không chạy lại detector, chỉ chạy phân loại và OCR.

    Tham số:
        plate_bgr: ảnh BGR vùng biển
        plate_bbox: (x, y, w, h)
        ocr_engine: OCREngine instance
        save_debug: True = lưu ảnh debug

    Trả về: dict kết quả tương tự _process_single_plate
    """
    start_time = time.time()

    if ocr_engine is None:
        ocr_engine = OCREngine()
        ocr_engine.initialize()

    if plate_bgr is None or plate_bgr.size == 0:
        return _empty_plate_result(plate_bbox, "region", 0, "Ảnh vùng biển rỗng.")

    if save_debug:
        _save_debug_image(plate_bgr, "region_input")

    # Phân loại màu (trên ảnh MÀU)
    color_result = classify_plate_color(plate_bgr)

    # Tiền xử lý và OCR
    variants = create_ocr_variants(plate_bgr)

    if save_debug and variants:
        for var_name, var_img in variants:
            _save_debug_image(var_img, f"region_preproc_{var_name}")

    plate_gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    try:
        ocr_result = ocr_engine.read_plate(plate_gray, plate_bbox, variants)
    except Exception as e:
        print(f"[PIPELINE] Lỗi OCR vùng region: {e}")
        ocr_result = {
            "raw_text": "",
            "normalized_text": f"[Lỗi OCR: {e}]",
            "lines": [],
            "letters": "",
            "digits": "",
            "avg_confidence": 0.0,
            "best_variant": "",
            "all_attempts": [],
            "processing_time": 0.0,
            "ocr_boxes": [],
            "error": str(e),
        }

    # Phân loại bố cục
    layout_result = classify_layout(plate_bbox, ocr_result.get("ocr_boxes"))

    processing_time = time.time() - start_time

    return {
        "plate_idx": 0,
        "bbox": plate_bbox,
        "method": "region",
        "plate_bgr": plate_bgr,
        "plate_text": ocr_result.get("normalized_text", "Không đọc được"),
        "raw_text": ocr_result.get("raw_text", ""),
        "ocr_lines": ocr_result.get("lines", []),
        "letters": ocr_result.get("letters", ""),
        "digits": ocr_result.get("digits", ""),
        "ocr_confidence": ocr_result.get("avg_confidence", 0),
        "best_variant": ocr_result.get("best_variant", ""),
        "color": color_result.get("color", "Không xác định"),
        "color_ratios": color_result.get("ratios", {}),
        "color_confident": color_result.get("confident", False),
        "layout": layout_result.get("layout", "Không xác định"),
        "layout_evidence": layout_result.get("evidence", {}),
        "processing_time": round(processing_time, 3),
        "ocr_error": ocr_result.get("error"),
    }
