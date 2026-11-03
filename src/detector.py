# src/detector.py
# =========================================================================
# MODULE PHÁT HIỆN BIỂN SỐ
# - Phương pháp chính: OpenCV contour-based (không cần mô hình ngoài)
# - Hỗ trợ chọn vùng thủ công (manual ROI)
# - Hỗ trợ chế độ "ảnh biển đã cắt" (toàn ảnh là ROI)
# - Chấm điểm ứng viên dựa trên nhiều dấu hiệu kết hợp
# =========================================================================

import cv2
import numpy as np
import config


def _compute_edge_density(gray_region):
    """
    Tính mật độ cạnh trong một vùng ảnh xám.
    Biển số thường có nhiều cạnh do chữ và số.
    """
    edges = cv2.Canny(gray_region, 100, 200)
    total_pixels = gray_region.shape[0] * gray_region.shape[1]
    if total_pixels == 0:
        return 0
    return np.count_nonzero(edges) / total_pixels


def _count_char_candidates(gray_region, min_chars=4):
    """
    Đếm số ký tự ứng viên trong vùng ảnh bằng cách phân tích
    thành phần liên thông sau ngưỡng hóa.

    So với hàm cũ, hàm này:
    - Chuẩn hóa kích thước nhất quán
    - Dùng phạm vi kích thước ký tự chặt hơn
    - Tính thêm "mật độ ký tự" để phân biệt biển vs dấu phân cách
    - Yêu cầu ít nhất min_chars ký tự ứng viên (mặc định 4)

    Trả về: (has_enough_chars, char_count, density_score)
    """
    h_orig, w_orig = gray_region.shape[:2]
    if h_orig < 10 or w_orig < 20:
        return False, 0, 0.0

    # Độ tương phản tối thiểu: vùng mờ nhạt hoặc đồng màu không thể có ký tự (hỗ trợ cả ảnh mờ/tối)
    contrast = float(gray_region.std())
    if contrast < 18.0:
        return False, 0, 0.0

    # Chuẩn hóa về kích thước tham chiếu
    # Dùng kích thước lớn hơn để không mất chi tiết
    target_w = 240
    target_h = max(60, int(60 * h_orig / max(w_orig / 4, 1)))
    target_h = min(target_h, 120)
    resized = cv2.resize(gray_region, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Cắt nhẹ 6% viền ngoài để khung viền biển số (trắng hoặc đen) không nối thành 1 khối bao quanh chữ
    my = max(2, int(target_h * 0.06))
    mx = max(4, int(target_w * 0.03))
    inner = resized[my:target_h - my, mx:target_w - mx]
    h_img, w_img = inner.shape[:2]

    # Thử cả 2 chiều nhị phân:
    # - THRESH_BINARY_INV: chữ đen trên nền sáng (biển trắng, biển vàng)
    # - THRESH_BINARY: chữ trắng trên nền tối (biển xanh, biển đỏ)
    best_count = 0
    best_density = 0.0

    for thresh_mode in (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU):
        _, binary = cv2.threshold(inner, 0, 255, thresh_mode)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_chars = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            char_area = w * h
            h_ratio = h / h_img
            w_ratio = w / w_img
            area_ratio = char_area / (h_img * w_img)
            char_aspect = w / h if h > 0 else 0
            # Ký tự biển số luôn cao hơn rộng (char_aspect <= 1.2)
            if (0.28 <= h_ratio <= 0.95 and
                    0.025 <= w_ratio <= 0.35 and
                    0.12 <= char_aspect <= 1.15 and
                    area_ratio >= 0.005):
                raw_chars.append((x, y, w, h))

        # Lọc các ký tự có chiều cao tương đồng nhau (tránh đốm nhiễu ngẫu nhiên trên toàn ảnh xe)
        valid_chars = []
        if len(raw_chars) >= 2:
            heights = np.array([c[3] for c in raw_chars], dtype=float)
            med_h = float(np.median(heights))
            valid_chars = [
                c for c in raw_chars
                if 0.60 * med_h <= c[3] <= 1.45 * med_h
            ]
        else:
            valid_chars = raw_chars

        c_count = len(valid_chars)
        d_score = 0.0
        if c_count >= 2:
            x_centers = [(x + w / 2) for (x, y, w, h) in valid_chars]
            x_centers.sort()
            gaps = [x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)]
            avg_gap = np.mean(gaps) if gaps else 0
            std_gap = np.std(gaps) if len(gaps) > 1 else 0
            if avg_gap > 0:
                evenness = 1.0 - min(std_gap / avg_gap, 1.0)
                d_score = c_count * evenness

        if (c_count > best_count) or (c_count == best_count and d_score > best_density):
            best_count = c_count
            best_density = d_score

    has_enough = best_count >= min_chars
    return has_enough, best_count, best_density


def _score_candidate(gray, x, y, w, h, img_area):
    """
    Chấm điểm một vùng ứng viên dựa trên nhiều dấu hiệu kết hợp.

    Dấu hiệu:
    1. Tỷ lệ w/h phù hợp với biển số Việt Nam
    2. Diện tích hợp lý so với ảnh
    3. Mật độ cạnh (biển số có nhiều cạnh)
    4. Số ký tự ứng viên + phân bố đều
    5. Độ tương phản (biển số thường tương phản cao)

    Trả về: (điểm số, thông tin debug)
    """
    score = 0.0
    area = w * h
    aspect = w / h if h > 0 else 0
    details = {}

    # 1. Tỷ lệ w/h
    # Biển 1 dòng: 3.0 - 6.0, biển 2 dòng: 1.2 - 2.5
    if 1.2 <= aspect <= 6.5:
        score += 20.0
        # Tỷ lệ đặc trưng nhất của biển VN
        if 1.5 <= aspect <= 2.3:   # 2 dòng (30×15cm ≈ 2.0)
            score += 15.0
        elif 3.8 <= aspect <= 5.2:  # 1 dòng (33×6.5cm ≈ 5.1)
            score += 15.0
        elif 2.3 < aspect < 3.8:   # Vùng trung gian
            score += 5.0
    details["aspect"] = round(aspect, 2)

    # 2. Diện tích tương đối
    area_ratio = area / img_area if img_area > 0 else 0
    if 0.003 <= area_ratio <= 0.85:
        score += 15.0
        if 0.006 <= area_ratio <= 0.15:  # Kích thước "chuẩn" của biển trong ảnh toàn xe
            score += 12.0
        elif 0.15 < area_ratio <= 0.80:  # Biển chụp cận cảnh
            score += 10.0
    elif area_ratio > 0.85:
        # Khung bao toàn bộ ảnh (thường là viền ảnh, không phải biển số trong ảnh toàn xe)
        score -= 10.0
    details["area_ratio"] = round(area_ratio, 4)

    # 3. Mật độ cạnh
    roi_gray = gray[y:y+h, x:x+w]
    if roi_gray.size > 0:
        edge_density = _compute_edge_density(roi_gray)
        if edge_density >= config.MIN_EDGE_DENSITY:
            score += 10.0
            score += min(edge_density * 30, 8.0)
        details["edge_density"] = round(edge_density, 3)
    else:
        details["edge_density"] = 0

    # 4. Dấu hiệu ký tự (quan trọng nhất)
    has_text, char_count, density_score = _count_char_candidates(roi_gray)
    if has_text:
        score += 25.0  # Trọng số cao - đây là bằng chứng mạnh nhất
        # Thêm điểm cho phân bố đều và số ký tự hợp lý
        if 6 <= char_count <= 12:
            score += 10.0
        score += min(density_score * 2, 8.0)
    elif char_count >= 2:
        # Có một ít ký tự - không đủ nhưng vẫn tích cực
        score += 5.0
    details["char_count"] = char_count
    details["density_score"] = round(density_score, 2)

    # 5. Tương phản (biển thường có tương phản cao)
    if roi_gray.size > 0:
        contrast = float(roi_gray.std())
        if contrast > 40:
            score += 5.0
        details["contrast"] = round(contrast, 1)

    return score, details


def _non_max_suppression(candidates, iou_threshold):
    """
    Loại bỏ các vùng trùng lặp (Non-Maximum Suppression).
    Giữ lại vùng có điểm cao nhất khi hai vùng chồng nhau nhiều.

    Bổ sung: NMS không chỉ dùng IoU mà còn xét "contained" -
    nếu một vùng nằm gần hoàn toàn bên trong vùng khác thì bỏ.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    keep = []
    for cand in candidates:
        x1, y1, w1, h1 = cand["bbox"]
        is_redundant = False

        for idx_k, kept in enumerate(keep):
            x2, y2, w2, h2 = kept["bbox"]

            # Tính IoU
            xi = max(x1, x2)
            yi = max(y1, y2)
            xa = min(x1 + w1, x2 + w2)
            ya = min(y1 + h1, y2 + h2)

            inter_w = max(0, xa - xi)
            inter_h = max(0, ya - yi)
            inter_area = inter_w * inter_h

            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            # Kiểm tra "contained": vùng nhỏ nằm trong vùng lớn hơn
            if inter_area > 0:
                overlap_of_small = inter_area / min(area1, area2) if min(area1, area2) > 0 else 0
                if overlap_of_small > 0.7:
                    # Nếu cand là khung ôm sát biển số nằm gọn bên trong khung kept bị dư viền/cản xe:
                    # ưu tiên giữ khung ôm sát biển số (area1 < area2) nếu vẫn đủ ký tự
                    cand_chars = cand.get("details", {}).get("char_count", 0)
                    if (area1 < area2 and area2 >= 1.25 * area1 and
                            cand_chars >= 5 and cand["score"] >= 0.82 * kept["score"]):
                        keep[idx_k] = cand
                    is_redundant = True
                    break

            if iou > iou_threshold:
                is_redundant = True
                break

        if not is_redundant:
            keep.append(cand)

    return keep


def detect_plates_opencv(img_bgr, return_debug=False):
    """
    Phát hiện biển số xe bằng OpenCV (contour-based).

    Luồng xử lý:
    1. Resize về kích thước xử lý (giữ tỷ lệ, quy đổi tọa độ về gốc)
    2. Chuyển ảnh xám
    3. Giảm nhiễu nhẹ (bilateral filter giữ cạnh)
    4. Phát hiện cạnh bằng Canny
    5. Đóng khe hở bằng morphology
    6. Tìm contour
    7. Lọc sơ bộ (kích thước, tỷ lệ, rectangularity)
    8. Chấm điểm đa dấu hiệu
    9. Loại vùng trùng lặp (NMS với IoU và contained)
    10. Lọc điểm tối thiểu

    Trả về: danh sách dict, mỗi dict chứa:
        - bbox: (x, y, w, h) trên ảnh GỐC (chưa resize)
        - score: điểm số ứng viên
        - method: "auto"
        - details: dict debug
    """
    orig_h, orig_w = img_bgr.shape[:2]
    debug_candidates = []
    img_area_orig = orig_h * orig_w

    # Bước 1: Resize để phát hiện (giảm nhiễu, tăng tốc)
    # Tỷ lệ scale để dùng quy đổi tọa độ sau
    max_dim = config.DETECTOR_RESIZE_MAX
    scale = min(max_dim / max(orig_w, orig_h), 1.0)
    if scale < 1.0:
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)
        img_proc = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        img_proc = img_bgr
        scale = 1.0

    proc_h, proc_w = img_proc.shape[:2]
    img_area_proc = proc_h * proc_w

    # Bước 2: Chuyển ảnh xám
    gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)

    # Bước 3: Giảm nhiễu nhẹ - bilateral filter giữ cạnh
    denoised = cv2.bilateralFilter(gray_proc, 11, 17, 17)

    # Bước 4 & 5 & 6: Đa tầng phát hiện vùng biển số
    # Tầng 1 (Chính): Canny + Morph Close chuẩn (17x5) + Dilate
    # Tầng 2 (Mở rộng cho ảnh toàn cảnh xe có lưới tản nhiệt/cản xe phức tạp):
    #   - Canny + Morph Close hẹp (11x3) không dilate
    #   - Blackhat/Tophat + Gradient dọc Sobel-X (bỏ qua thanh ngang lưới tản nhiệt)
    #   - Mặt nạ màu biển số HSV (Vàng, Xanh dương, Trắng sáng)
    edges = cv2.Canny(denoised, config.CANNY_LOW, config.CANNY_HIGH)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.MORPH_KERNEL_WIDTH, config.MORPH_KERNEL_HEIGHT)
    )
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)
    primary_contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def _build_extra_contours():
        extra = []
        # Kênh 2: Hẹp (không dilate để tách biển khỏi lưới tản nhiệt)
        kernel_tight = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        closed_tight = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_tight)
        cnts2, _ = cv2.findContours(closed_tight, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        extra.extend(cnts2)

        # Kênh 3: Sobel-X trên Blackhat (chữ tối nền sáng) & Tophat (chữ sáng nền tối)
        rect_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        for morph_op in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
            hat = cv2.morphologyEx(denoised, morph_op, rect_bh)
            grad_x = cv2.Sobel(hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            grad_x = np.absolute(grad_x)
            max_val = np.max(grad_x)
            if max_val > 0:
                grad_x = (255 * (grad_x / max_val)).astype("uint8")
                grad_x = cv2.GaussianBlur(grad_x, (3, 3), 0)
                grad_closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_bh)
                _, grad_thresh = cv2.threshold(grad_closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts_g, _ = cv2.findContours(grad_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                extra.extend(cnts_g)

        # Kênh 4: Mặt nạ màu biển số HSV (Biển Vàng, Biển Xanh, Biển Trắng trên thân xe)
        hsv_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(
            hsv_proc,
            (config.YELLOW_H_MIN, config.YELLOW_S_MIN, config.YELLOW_V_MIN),
            (config.YELLOW_H_MAX, 255, 255)
        )
        mask_blue = cv2.inRange(
            hsv_proc,
            (config.BLUE_H_MIN, config.BLUE_S_MIN, config.BLUE_V_MIN),
            (config.BLUE_H_MAX, 255, 255)
        )
        mask_white = cv2.inRange(
            hsv_proc,
            (0, 0, config.WHITE_V_MIN),
            (179, config.WHITE_S_MAX, 255)
        )
        color_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        for cmask in (mask_yellow, mask_blue, mask_white):
            cm_closed = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, color_close_k)
            cnts_c, _ = cv2.findContours(cm_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            extra.extend(cnts_c)
        return extra

    contours = list(primary_contours)

    # Bước 7: Lọc sơ bộ và chấm điểm
    def _evaluate_contours(contour_list):
        cands = []
        seen_boxes = set()
        for cnt in contour_list:
            x_p, y_p, w_p, h_p = cv2.boundingRect(cnt)
            if h_p == 0 or w_p == 0:
                continue
            box_key = (x_p, y_p, w_p, h_p)
            if box_key in seen_boxes:
                continue
            seen_boxes.add(box_key)

            aspect = w_p / h_p
            area = w_p * h_p
            area_ratio = area / img_area_proc

            if not (config.MIN_ASPECT_RATIO <= aspect <= config.MAX_ASPECT_RATIO):
                continue
            if not (config.MIN_AREA_RATIO <= area_ratio <= config.MAX_AREA_RATIO):
                continue
            if w_p < 40 or h_p < 15:
                continue

            cnt_area = cv2.contourArea(cnt)
            rect_area = w_p * h_p
            rectangularity = cnt_area / rect_area if rect_area > 0 else 0
            if rectangularity < config.MIN_RECTANGULARITY:
                continue

            x_orig = int(x_p / scale)
            y_orig = int(y_p / scale)
            w_orig = int(w_p / scale)
            h_orig_coord = int(h_p / scale)

            x_orig = max(0, min(x_orig, orig_w - 1))
            y_orig = max(0, min(y_orig, orig_h - 1))
            w_orig = max(1, min(w_orig, orig_w - x_orig))
            h_orig_coord = max(1, min(h_orig_coord, orig_h - y_orig))

            score, details = _score_candidate(
                gray_orig, x_orig, y_orig, w_orig, h_orig_coord, img_area_orig
            )

            debug_entry = {
                "bbox_processing": (x_p, y_p, w_p, h_p),
                "bbox_original": (x_orig, y_orig, w_orig, h_orig_coord),
                "score_before_rectangularity_bonus": round(score, 3),
                "rectangularity": round(rectangularity, 3),
                "details": details,
                "reason": "pending",
            }
            if return_debug:
                debug_candidates.append(debug_entry)

            if details["char_count"] < config.DETECTOR_MIN_CHAR_CANDIDATES:
                debug_entry["reason"] = "rejected_insufficient_character_evidence"
                debug_entry["required_char_count"] = config.DETECTOR_MIN_CHAR_CANDIDATES
                continue

            if details["contrast"] < 18.0 or details["edge_density"] < 0.012:
                debug_entry["reason"] = "rejected_low_contrast_or_edges"
                continue

            if rectangularity > 0.7:
                score += 8.0

            cands.append({
                "bbox": (x_orig, y_orig, w_orig, h_orig_coord),
                "score": score,
                "method": "auto",
                "rectangularity": rectangularity,
                "details": details,
            })
        return cands

    primary_cands = _evaluate_contours(primary_contours)
    extra_contours = _build_extra_contours()
    contours.extend(extra_contours)
    extra_cands = _evaluate_contours(extra_contours)

    # Gộp ứng viên Tầng 2 vào Tầng 1:
    # - Không ghi đè ứng viên Tầng 1 nếu trùng vị trí (IoU > 0.45) trừ khi Tầng 2 là khung con
    #   ôm sát biển số hơn rõ rệt (nhỏ hơn ít nhất 25% diện tích của khung Tầng 1 bị dính cản/lưới xe)
    candidates = list(primary_cands)
    for ec in extra_cands:
        ex, ey, ew, eh = ec["bbox"]
        e_area = ew * eh
        should_add = True
        for idx_p, pc in enumerate(candidates):
            px, py, pw, ph = pc["bbox"]
            p_area = pw * ph
            xi = max(ex, px)
            yi = max(ey, py)
            xa = min(ex + ew, px + pw)
            ya = min(ey + eh, py + ph)
            inter = max(0, xa - xi) * max(0, ya - yi)
            if inter <= 0:
                continue
            union = e_area + p_area - inter
            iou = inter / union if union > 0 else 0
            overlap_small = inter / min(e_area, p_area) if min(e_area, p_area) > 0 else 0

            if overlap_small > 0.75 and e_area < 0.75 * p_area and ec["details"].get("char_count", 0) >= 5:
                # Khung Tầng 2 tách rời được biển số nằm gọn trong khung Tầng 1 bị dính cản xe
                candidates[idx_p] = ec
                should_add = False
                break
            if iou > 0.45 or overlap_small > 0.70:
                should_add = False
                break
        if should_add:
            candidates.append(ec)

    # Bước 9: Loại vùng trùng lặp
    candidates = _non_max_suppression(candidates, config.NMS_IOU_THRESHOLD)

    # Bước 10: Lọc ngưỡng điểm tối thiểu
    MIN_SCORE_THRESHOLD = 45.0
    candidates = [c for c in candidates if c["score"] >= MIN_SCORE_THRESHOLD]

    # Bước 11: Lọc ứng viên phụ quá nhỏ so với ứng viên tốt nhất (tránh bóng/nhiễu thành biển 2)
    if candidates:
        candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
        best_cand = candidates[0]
        best_area = best_cand["bbox"][2] * best_cand["bbox"][3]
        best_score = best_cand["score"]

        filtered_cands = [best_cand]
        for c in candidates[1:]:
            c_area = c["bbox"][2] * c["bbox"][3]
            c_area_ratio = c["details"].get("area_ratio", 0)
            c_chars = c["details"].get("char_count", 0)
            # Giữ ứng viên phụ nếu đủ tỷ lệ diện tích so với best_cand HOẶC có kích thước chuẩn biển số toàn xe
            if (c_area >= 0.15 * best_area or (c_area_ratio >= 0.004 and c_chars >= 5)) and c["score"] >= 0.65 * best_score:
                filtered_cands.append(c)
        candidates = filtered_cands

    if return_debug:
        accepted_boxes = {candidate["bbox"] for candidate in candidates}
        for item in debug_candidates:
            if item["reason"] == "pending":
                item["reason"] = (
                    "accepted" if item["bbox_original"] in accepted_boxes
                    else "rejected_score_or_nms"
                )
                if item["reason"] == "rejected_score_or_nms":
                    item["minimum_score"] = MIN_SCORE_THRESHOLD

    # Giữ tối đa MAX_CANDIDATES vùng
    candidates = candidates[:config.MAX_CANDIDATES]

    if return_debug:
        return candidates, {
            "image_shape": (orig_h, orig_w),
            "scale": scale,
            "contours_total": len(contours),
            "candidates": debug_candidates,
        }
    return candidates


def manual_roi(img_bgr, x, y, w, h):
    """
    Tạo vùng biển số từ tọa độ do người dùng chỉ định.

    Tham số:
        img_bgr: ảnh BGR
        x, y: tọa độ góc trên-trái
        w, h: chiều rộng và chiều cao

    Trả về:
        dict chứa bbox và method="manual", hoặc None nếu tọa độ không hợp lệ
    """
    img_h, img_w = img_bgr.shape[:2]

    # Ép tọa độ về trong biên ảnh
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    # Kiểm tra vùng có đủ lớn không
    if w < 20 or h < 10:
        return None

    return {
        "bbox": (x, y, w, h),
        "score": -1,  # Không chấm điểm cho vùng thủ công
        "method": "manual",
    }


def full_image_as_plate(img_bgr):
    """
    Coi toàn bộ ảnh là một biển số (dùng khi ảnh đã cắt sát biển).

    Thêm lề nhỏ để không bị cắt cụt ký tự ở biên.
    Trả về: dict candidate với method="full_image"
    """
    h, w = img_bgr.shape[:2]
    return {
        "bbox": (0, 0, w, h),
        "score": 100.0,  # Điểm tối đa - người dùng đã xác nhận
        "method": "full_image",
    }


def crop_plate(img_bgr, bbox, margin_ratio=None):
    """
    Cắt vùng biển số từ ảnh với lề hợp lý.

    Tham số:
        img_bgr: ảnh BGR gốc
        bbox: (x, y, w, h) của vùng biển
        margin_ratio: tỷ lệ lề (mặc định dùng config)

    Trả về:
        (plate_bgr, adjusted_bbox)
        - plate_bgr: ảnh biển số đã cắt
        - adjusted_bbox: tọa độ thực tế sau khi thêm lề và clip
    """
    if margin_ratio is None:
        margin_ratio = config.CROP_MARGIN_RATIO

    x, y, w, h = bbox
    img_h, img_w = img_bgr.shape[:2]

    # Tính lề
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    # Áp dụng lề và clip về trong biên ảnh
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    plate_bgr = img_bgr[y1:y2, x1:x2].copy()
    adjusted_bbox = (x1, y1, x2 - x1, y2 - y1)

    return plate_bgr, adjusted_bbox
