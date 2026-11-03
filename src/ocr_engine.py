# src/ocr_engine.py
# =========================================================================
# MODULE OCR - NHẬN DIỆN KÝ TỰ TRÊN BIỂN SỐ
# - Dùng Tesseract OCR (chạy cục bộ, không gửi ảnh ra ngoài)
# - Khởi tạo một lần, tái sử dụng cho mọi ảnh
# - Hỗ trợ nhiều biến thể tiền xử lý để chọn kết quả tốt nhất
# - Sắp xếp kết quả theo bố cục biển số
#
# Tesseract cần cài:
#   MSYS2: pacman -S mingw-w64-ucrt-x86_64-tesseract-ocr
#                    mingw-w64-ucrt-x86_64-tesseract-data-eng
#   Hoặc: https://github.com/UB-Mannheim/tesseract/wiki
# =========================================================================

import re
import time
import numpy as np
import cv2
import pytesseract
from PIL import Image

import config
from src.layout_classifier import group_chars_by_line

# Đường dẫn Tesseract (MSYS2 trên Windows)
TESSERACT_CMD = r"C:\msys64\ucrt64\bin\tesseract.exe"


class OCREngine:
    """
    Bộ nhận diện ký tự dùng Tesseract OCR.
    Singleton pattern: chỉ khởi tạo 1 lần trong suốt vòng đời ứng dụng.
    """

    _instance = None  # Singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_time = 0
        self._ready = False

    def initialize(self):
        """
        Kiểm tra Tesseract có khả dụng không.
        Tesseract không cần nạp model vào RAM như DL models.
        """
        if self._ready:
            return

        start = time.time()
        try:
            # Thiết lập đường dẫn tesseract
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            # Kiểm tra hoạt động
            version = pytesseract.get_tesseract_version()
            print(f"[OCR] Tesseract version: {version}")
        except Exception as e:
            print(f"[OCR] CẢNH BÁO: Tesseract không tìm thấy tại {TESSERACT_CMD}")
            print(f"[OCR] Lỗi: {e}")
            print("[OCR] Thử dùng PATH mặc định...")
            # Fallback: dùng PATH hệ thống
            pytesseract.pytesseract.tesseract_cmd = "tesseract"

        self._load_time = time.time() - start
        self._ready = True

    @property
    def load_time(self):
        """Thời gian khởi tạo (giây)."""
        return self._load_time

    def read_plate(self, plate_gray, plate_bbox, variants=None):
        """
        Nhận diện ký tự trên biển số.

        Quy trình:
        1. Thử OCR trên từng biến thể tiền xử lý
        2. Chấm điểm kết quả dựa trên số ký tự hợp lệ
        3. Chọn biến thể tốt nhất
        4. Sắp xếp ký tự theo bố cục (1 dòng hoặc 2 dòng)
        5. Chuẩn hóa kết quả

        Tham số:
            plate_gray: ảnh xám vùng biển số (numpy array)
            plate_bbox: (x, y, w, h) của vùng biển
            variants: danh sách (tên, ảnh_xám) từ preprocessing

        Trả về:
            dict chứa đầy đủ kết quả OCR
        """
        if not self._ready:
            self.initialize()

        start_time = time.time()

        # Nếu không có biến thể, dùng ảnh gốc
        if variants is None:
            variants = [("gray", plate_gray)]

        best_result = None
        best_score = -1
        best_variant = ""
        all_attempts = []

        for variant_name, variant_img in variants:
            try:
                ocr_boxes = self._run_tesseract_boxes(variant_img)

                # Lưu kết quả thô của engine
                engine_results = [
                    {
                        "bbox": item["bbox"],
                        "text": item["text"],
                        "confidence": item["confidence"],
                    }
                    for item in ocr_boxes
                ]

                # Lọc kết quả hợp lệ
                filtered = []
                for item in ocr_boxes:
                    text = item["text"].strip()
                    conf = item["confidence"]
                    # Giữ ký tự trong allowlist (gồm cả dấu - và .)
                    text_clean = re.sub(r"[^A-Z0-9.-]", "", text.upper()).strip(".-")
                    confidence_threshold = config.OCR_MIN_CONFIDENCE * 100
                    # Tesseract có thể trả về conf == 0 cho từ nhận diện đúng
                    confidence_unavailable = conf == 0
                    if text_clean and (conf >= confidence_threshold or confidence_unavailable):
                        filtered.append({
                            "bbox": item["bbox"],
                            "text": text_clean,
                            "confidence": conf / 100.0 if conf > 0 else 0.05,
                            "reported_confidence": conf,
                            "confidence_unavailable": confidence_unavailable,
                        })

                # Chấm điểm thông minh:
                # Đếm số ký tự chữ/số (bỏ dấu chấm/gạch)
                if filtered:
                    alnum_chars = [c for f in filtered for c in f["text"] if c.isalnum()]
                    n_alnum = len(alnum_chars)
                    avg_conf = float(np.mean([f["confidence"] for f in filtered]))

                    if n_alnum > 14:
                        # Biển số xe VN chuẩn không vượt quá 12-14 ký tự; >14 là nhiễu ảnh vỡ vụn
                        score = 0.0
                    elif n_alnum < 4:
                        # Quá ít ký tự để là biển số hợp lệ
                        score = n_alnum * avg_conf * 0.5
                    else:
                        # 6 đến 10 ký tự là khoảng kích thước chuẩn của biển số VN
                        bonus = 2.0 if (7 <= n_alnum <= 10) else (1.0 if (5 <= n_alnum <= 12) else 0.0)
                        # Nếu là biển vuông 2 dòng (w/h < 2.4), ưu tiên biến thể đọc đủ cả 2 dòng
                        _, _, bw, bh = plate_bbox
                        if bh > 0 and (bw / bh) < 2.4:
                            lines_found = group_chars_by_line(filtered, bh)
                            if len(lines_found) >= 2:
                                bonus += 2.5
                        score = n_alnum * avg_conf + bonus
                else:
                    score = 0.0

                attempt = {
                    "variant": variant_name,
                    "engine_results": engine_results,
                    "raw_results": filtered,
                    "score": score,
                    "n_chars": sum(len(f["text"]) for f in filtered) if filtered else 0,
                }
                all_attempts.append(attempt)

                if score > best_score:
                    best_score = score
                    best_result = filtered
                    best_variant = variant_name

            except Exception as e:
                all_attempts.append({
                    "variant": variant_name,
                    "error": str(e),
                    "score": 0,
                })

        processing_time = time.time() - start_time

        # Nếu không nhận được gì
        if best_result is None or len(best_result) == 0:
            # Fallback: chạy toàn bộ ảnh lấy string thô
            fallback_text = self._run_tesseract_string(plate_gray)
            fallback_text = self._normalize_text(fallback_text)
            return {
                "raw_text": fallback_text,
                "normalized_text": fallback_text if fallback_text else "Không đọc được",
                "lines": [fallback_text] if fallback_text else [],
                "letters": re.sub(r"[^A-Z]", "", fallback_text),
                "digits": re.sub(r"[^0-9]", "", fallback_text),
                "avg_confidence": 0.0,
                "best_variant": best_variant,
                "all_attempts": all_attempts,
                "processing_time": round(processing_time, 3),
                "ocr_boxes": [],
            }

        # Sắp xếp theo bố cục biển (1 dòng / 2 dòng)
        _, _, pw, ph = plate_bbox
        lines = group_chars_by_line(best_result, ph)

        # Ghép thành chuỗi theo từng dòng
        line_texts = []
        for line in lines:
            line_text = "".join([item["text"] for item in line])
            line_texts.append(line_text)

        if len(line_texts) > 1:
            raw_text = " / ".join(line_texts)
        else:
            raw_text = line_texts[0] if line_texts else ""
        normalized = self._normalize_text(raw_text)

        letters = re.sub(r"[^A-Z]", "", normalized)
        digits = re.sub(r"[^0-9]", "", normalized)
        avg_conf = np.mean([r["confidence"] for r in best_result]) if best_result else 0.0

        return {
            "raw_text": raw_text,
            "normalized_text": normalized,
            "lines": line_texts,
            "letters": letters,
            "digits": digits,
            "avg_confidence": round(float(avg_conf), 3),
            "best_variant": best_variant,
            "all_attempts": all_attempts,
            "processing_time": round(processing_time, 3),
            "ocr_boxes": best_result,
        }

    # ------------------------------------------------------------------
    # Phương thức nội bộ
    # ------------------------------------------------------------------

    def _run_tesseract_boxes(self, gray_img):
        """
        Chạy Tesseract ở chế độ word boxes - hỗ trợ cả biển 1 dòng và 2 dòng.
        
        Quy trình chuẩn hóa:
        1. Cắt nhẹ viền ngoài nếu ảnh lớn để tránh đọc nhầm viền khung thành '1'
        2. Tách dòng nếu là biển 2 dòng (aspect ratio < 2.4 và có khoảng trống giữa)
        3. Thêm đệm trắng (quiet zone) và resize về chiều cao chuẩn (~80px)
        4. Chạy Tesseract và quy đổi tọa độ về ảnh gốc
        """
        ph, pw = gray_img.shape[:2]
        if ph < 10 or pw < 20:
            return []

        # 1. Tránh đọc nhầm viền đen ngoài cùng của biển số thành '1' hay '.'
        if ph >= 60 and pw >= 100:
            mx = int(pw * 0.065)
            my = int(ph * 0.065)
            inner = gray_img[my:ph-my, mx:pw-mx]
        else:
            mx, my = 0, 0
            inner = gray_img

        ih, iw = inner.shape[:2]
        aspect = iw / ih if ih > 0 else 1.0

        # 2. Kiểm tra xem có phải biển 2 dòng không (aspect ratio < 2.4)
        lines_crops = []
        if aspect < 2.4 and ih >= 50:
            proj = np.mean(inner[:, int(iw * 0.1):int(iw * 0.9)], axis=1)
            mid_min = int(ih * 0.35)
            mid_max = int(ih * 0.65)
            if mid_max > mid_min:
                valley = mid_min + int(np.argmax(proj[mid_min:mid_max]))
                lines_crops = [
                    (inner[:valley, :], 0),
                    (inner[valley:, :], valley)
                ]
        if not lines_crops:
            lines_crops = [(inner, 0)]

        all_boxes = []
        line_configs = [
            r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
            r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
            r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
        ]

        for line_img, y_off in lines_crops:
            lh, lw = line_img.shape[:2]
            if lh < 10 or lw < 20:
                continue
            # Thêm padding trắng (quiet zone)
            pad = 25
            padded = cv2.copyMakeBorder(line_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            # Chuẩn hóa chiều cao dòng về ~80px (tối ưu cho mô hình LSTM Tesseract)
            target_h = 80
            scale = target_h / padded.shape[0]
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            resized = cv2.resize(padded, (int(padded.shape[1] * scale), target_h), interpolation=interp)

            line_boxes = []
            for cfg in line_configs:
                try:
                    data = pytesseract.image_to_data(
                        resized,
                        config=cfg,
                        output_type=pytesseract.Output.DICT,
                    )
                except Exception:
                    continue

                n = len(data["text"])
                for i in range(n):
                    t = data["text"][i].strip()
                    try:
                        conf = float(data["conf"][i])
                    except (TypeError, ValueError):
                        continue
                    if conf < 0 or not t:
                        continue
                    t_clean = re.sub(r"[^A-Z0-9.-]", "", t.upper()).strip(".-")
                    if not t_clean:
                        continue

                    rx = data["left"][i]
                    ry = data["top"][i]
                    rw = data["width"][i]
                    rh = data["height"][i]

                    # Quy đổi tọa độ về ảnh gray_img gốc
                    ox = int(rx / scale) - pad + mx
                    oy = int(ry / scale) - pad + my + y_off
                    ow = int(rw / scale)
                    oh = int(rh / scale)

                    box_4pts = [[ox, oy], [ox+ow, oy], [ox+ow, oy+oh], [ox, oy+oh]]
                    line_boxes.append({
                        "bbox": box_4pts,
                        "text": t_clean,
                        "confidence": conf,
                    })
                if line_boxes:
                    break

            all_boxes.extend(line_boxes)

        if all_boxes:
            return all_boxes

        # Fallback: Chạy toàn bộ ảnh với các PSM nếu line crops không cho ra box
        configs_to_try = [
            r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
            r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
            r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
        ]
        pil_img = Image.fromarray(gray_img) if isinstance(gray_img, np.ndarray) else gray_img
        for c in configs_to_try:
            try:
                data = pytesseract.image_to_data(pil_img, config=c, output_type=pytesseract.Output.DICT)
                boxes = []
                for i in range(len(data["text"])):
                    t = data["text"][i].strip()
                    try:
                        conf = float(data["conf"][i])
                    except (TypeError, ValueError):
                        continue
                    if conf < 0 or not t:
                        continue
                    t_clean = re.sub(r"[^A-Z0-9.-]", "", t.upper()).strip(".-")
                    if not t_clean:
                        continue
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    boxes.append({
                        "bbox": [[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                        "text": t_clean,
                        "confidence": conf,
                    })
                if boxes:
                    return boxes
            except Exception:
                continue

        return []

    def _run_tesseract_string(self, gray_img):
        """
        Chạy Tesseract trả về chuỗi thô (fallback khi không có boxes).
        """
        if isinstance(gray_img, np.ndarray):
            pil_img = Image.fromarray(gray_img)
        else:
            pil_img = gray_img

        cfg = r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
        try:
            return pytesseract.image_to_string(pil_img, config=cfg).strip()
        except Exception:
            return ""

    def _normalize_text(self, text):
        """
        Chuẩn hóa chuỗi OCR và loại bỏ ký tự nhiễu viền biển số (ví dụ cờ đỏ/viền dọc đọc nhầm thành '1').
        """
        text = text.upper()
        text = re.sub(r"[^A-Z0-9 ./-]", "", text)
        text = " ".join(text.split()).strip()
        # Loại bỏ chữ số '1' thừa ở đầu do viền trái/tem cờ đọc nhầm trước mã tỉnh 2 số (VD: 130E-921.15 -> 30E-921.15)
        text = re.sub(r"^[1I]([1-9][0-9][A-Z])", r"\1", text)
        # Loại bỏ chữ số '1' thừa ở cuối sau cụm 5 số (VD: 921.151 -> 921.15)
        text = re.sub(r"([0-9]{3}\.[0-9]{2})[1I]$", r"\1", text)
        return text.strip()
