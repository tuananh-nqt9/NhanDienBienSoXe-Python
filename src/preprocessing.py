# src/preprocessing.py
# =========================================================================
# MODULE TIỀN XỬ LÝ ẢNH BIỂN SỐ
# - Tạo các biến thể phục vụ OCR
# - CLAHE, threshold, giảm nhiễu
# - Hiệu chỉnh phối cảnh (khi có đủ thông tin)
# =========================================================================

import cv2
import numpy as np
import config


def try_perspective_correction(plate_bgr, contour=None):
    """
    Thử hiệu chỉnh phối cảnh cho biển số.

    Chỉ áp dụng khi tìm được 4 góc đáng tin cậy từ contour.
    Nếu không tìm được, trả về ảnh gốc và ghi rõ không hiệu chỉnh.

    Tham số:
        plate_bgr: ảnh vùng biển số (BGR)
        contour: contour của vùng biển (nếu có)

    Trả về:
        (corrected_img, was_corrected)
    """
    if contour is None:
        return plate_bgr, False

    # Xấp xỉ contour thành đa giác
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # Chỉ hiệu chỉnh khi tìm được đúng 4 góc
    if len(approx) != 4:
        return plate_bgr, False

    # Sắp xếp 4 góc: trên-trái, trên-phải, dưới-phải, dưới-trái
    pts = approx.reshape(4, 2).astype(np.float32)

    # Sắp xếp theo tổng tọa độ (trên-trái có tổng nhỏ nhất)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    src_pts = np.array([top_left, top_right, bottom_right, bottom_left],
                       dtype=np.float32)

    # Tính kích thước đầu ra
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    out_w = int(max(width_top, width_bottom))
    out_h = int(max(height_left, height_right))

    if out_w < 20 or out_h < 10:
        return plate_bgr, False

    dst_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    # Biến đổi phối cảnh
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(plate_bgr, M, (out_w, out_h))

    return corrected, True


def create_ocr_variants(plate_bgr):
    """
    Tạo các biến thể tiền xử lý phục vụ OCR.
    Mỗi biến thể nhấn mạnh ký tự theo cách khác nhau.

    Không áp dụng tất cả bộ lọc liên tiếp một cách máy móc.
    Mỗi biến thể là một cách tiền xử lý độc lập từ ảnh gốc.

    Trả về: danh sách (tên_biến_thể, ảnh_xám)
    """
    variants = []
    h, w = plate_bgr.shape[:2]

    # Resize biển số để OCR ổn định hơn (chiều cao tối thiểu 60px)
    if h < 60:
        scale = 60 / h
        plate_bgr = cv2.resize(plate_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    # Chuyển xám
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # Tự động đảo ngược sáng-tối nếu là biển nền tối chữ trắng (Biển Xanh dương / Biển Đỏ)
    # Kiểm tra vùng trung tâm biển: nếu nền tối chiếm đa số (Otsu mean < 120), đảo thành chữ đen nền trắng
    ih, iw = gray.shape[:2]
    inner_check = gray[int(ih * 0.15):int(ih * 0.85), int(iw * 0.1):int(iw * 0.9)]
    if inner_check.size > 0:
        _, check_otsu = cv2.threshold(inner_check, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(check_otsu) < 115:
            gray = 255 - gray

    # --- Biến thể 1: CLAHE (tăng tương phản cục bộ) ---
    # CLAHE giúp cải thiện ký tự khi biển bị chói, tối một phần hoặc mờ nhẹ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(("clahe", clahe_img))

    # --- Biến thể 2: Ảnh xám gốc ---
    variants.append(("gray", gray.copy()))

    # --- Biến thể 3: Otsu threshold ---
    # Otsu tự động tìm ngưỡng phân tách nền và chữ
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))

    # --- Biến thể 4: Giảm nhiễu + Adaptive threshold ---
    # Adaptive threshold tốt khi biển có độ sáng không đều
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(("adaptive", adaptive))

    # --- Biến thể 5: Làm nét (Unsharp Masking) khi ảnh bị mờ / out-focus ---
    # Kiểm tra phương sai Laplacian: nếu < 350 thì biển bị mờ, cần làm nét biên nét chữ
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 350:
        blur_mask = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpen = cv2.addWeighted(gray, 1.6, blur_mask, -0.6, 0)
        variants.append(("sharpen", sharpen))

    # --- Biến thể 6: Khử nhiễu giữ cạnh + Otsu (Bilateral Filter + Otsu) ---
    # Làm mượt phẳng nhiễu cảm biến/hạt trên nền biển nhưng giữ nguyên cạnh chữ
    bilateral = cv2.bilateralFilter(gray, 5, 50, 50)
    _, bi_otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_otsu", bi_otsu))

    # Giới hạn số biến thể theo config
    return variants[:config.MAX_OCR_VARIANTS]


def get_intermediate_images(plate_bgr):
    """
    Tạo các ảnh trung gian để hiển thị, phục vụ giải thích thuật toán.

    Trả về: dict {tên: ảnh_xám}
    """
    result = {}
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    result["Ảnh xám"] = gray

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result["CLAHE"] = clahe.apply(gray)

    # Làm nét (Sharpen) cho ảnh mờ
    blur_mask = cv2.GaussianBlur(gray, (0, 0), 2.0)
    result["Làm nét (Sharpen)"] = cv2.addWeighted(gray, 1.8, blur_mask, -0.8, 0)

    # Cạnh (Canny)
    result["Cạnh (Canny)"] = cv2.Canny(gray, 100, 200)

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result["Otsu threshold"] = otsu

    # Bilateral Otsu
    bi = cv2.bilateralFilter(gray, 5, 50, 50)
    _, bi_otsu = cv2.threshold(bi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result["Bilateral Otsu"] = bi_otsu

    # Adaptive threshold
    result["Adaptive threshold"] = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return result

