# src/visualization.py
# =========================================================================
# MODULE VẼ KẾT QUẢ VÀ HIỂN THỊ
# - Vẽ khung xanh lá quanh biển số
# - Vẽ nhãn tiếng Việt có dấu bằng Pillow
# - Tự điều chỉnh vị trí nhãn khi sát mép ảnh
# - Hiển thị ảnh bằng OpenCV
# =========================================================================

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config


def _find_font(size=None):
    """
    Tìm font hỗ trợ tiếng Việt trên Windows.

    Tìm theo thứ tự ưu tiên trong config.FONT_CANDIDATES.
    Nếu không tìm thấy font nào, hiển thị cảnh báo và dùng font mặc định.

    Trả về:
        PIL.ImageFont object
    """
    if size is None:
        size = config.LABEL_FONT_SIZE

    # Thử tìm font trong thư mục Windows
    for font_name in config.FONT_CANDIDATES:
        font_path = os.path.join(config.WINDOWS_FONT_DIR, font_name)
        if os.path.isfile(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    # Nếu không tìm thấy, thử tải font theo tên
    for font_name in config.FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue

    # Cuối cùng dùng font mặc định (không hỗ trợ tiếng Việt tốt)
    print("[CẢNH BÁO] Không tìm thấy font hỗ trợ tiếng Việt trên hệ thống.")
    print(f"  Đã tìm trong: {config.WINDOWS_FONT_DIR}")
    print(f"  Các font thử: {config.FONT_CANDIDATES}")
    print("  Sẽ dùng font mặc định (có thể không hiển thị dấu tiếng Việt).")
    return ImageFont.load_default()


def draw_results(img_bgr, plates_info):
    """
    Vẽ khung và nhãn lên ảnh cho tất cả biển số tìm được.

    Tham số:
        img_bgr: ảnh BGR gốc (sẽ được copy, không sửa trực tiếp)
        plates_info: danh sách dict, mỗi dict chứa:
            - bbox: (x, y, w, h)
            - plate_text: chuỗi biển số
            - color: màu nền biển
            - layout: "1 dòng" / "2 dòng"

    Trả về:
        ảnh BGR đã vẽ kết quả
    """
    result = img_bgr.copy()
    img_h, img_w = result.shape[:2]

    # Chuyển sang PIL để vẽ chữ tiếng Việt
    img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    for plate in plates_info:
        x, y, w, h = plate["bbox"]
        plate_text = plate.get("plate_text", "Không đọc được")
        color_name = plate.get("color", "Không xác định")
        layout = plate.get("layout", "Không xác định")

        # Tạo nội dung nhãn: [Biển số] (kèm điểm OCR) | [Màu] | [Bố cục]
        conf = plate.get("ocr_confidence", plate.get("confidence"))
        if conf is not None and conf > 0:
            label_text = f"{plate_text} (OCR: {conf:.2f}) | {color_name} | {layout}"
        else:
            label_text = f"{plate_text} | {color_name} | {layout}"

        # Tự động nhân tỷ lệ cỡ chữ theo độ phân giải ảnh so với cửa sổ hiển thị
        # để khi ảnh lớn bị thu nhỏ về MAX_DISPLAY_WIDTH x MAX_DISPLAY_HEIGHT,
        # chữ trên dòng màu xanh lá vẫn luôn to, đậm, rõ nét đúng bằng LABEL_FONT_SIZE
        scale_factor = max(
            1.0,
            img_w / max(1, config.MAX_DISPLAY_WIDTH),
            img_h / max(1, config.MAX_DISPLAY_HEIGHT),
        )
        font_size = max(config.LABEL_FONT_SIZE, int(round(config.LABEL_FONT_SIZE * scale_factor * 0.85)))
        font = _find_font(font_size)

        # Tính kích thước nhãn
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Nếu nhãn rộng hơn ảnh, giảm cỡ chữ vừa khít chiều ngang ảnh
        while text_w > img_w - 16 and font_size > 12:
            font_size -= 2
            font = _find_font(font_size)
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

        # Padding cho nhãn (tỷ lệ theo cỡ chữ)
        pad_x = max(6, int(round(font_size * 0.35)))
        pad_y = max(4, int(round(font_size * 0.22)))

        # Tính vị trí nhãn (mặc định: phía trên khung)
        label_x = x
        label_y = y - text_h - pad_y * 2 - 2

        # Nếu phía trên không đủ chỗ, đặt bên dưới khung
        if label_y < 0:
            label_y = y + h + 2

        # Nếu bên dưới cũng không đủ, đặt bên trong khung (trên cùng)
        if label_y + text_h + pad_y * 2 > img_h:
            label_y = y + 2

        # Điều chỉnh X nếu nhãn vượt ra ngoài phải
        if label_x + text_w + pad_x * 2 > img_w:
            label_x = max(0, img_w - text_w - pad_x * 2)

        # Điều chỉnh X nếu nhãn vượt ra ngoài trái
        label_x = max(0, label_x)

        # Vẽ khung chữ nhật xanh lá quanh biển số
        # Màu cho PIL là RGB
        box_color_rgb = (
            config.BOX_COLOR[2],
            config.BOX_COLOR[1],
            config.BOX_COLOR[0]
        )
        label_bg_rgb = (
            config.LABEL_BG_COLOR[2],
            config.LABEL_BG_COLOR[1],
            config.LABEL_BG_COLOR[0]
        )
        label_text_rgb = (
            config.LABEL_TEXT_COLOR[2],
            config.LABEL_TEXT_COLOR[1],
            config.LABEL_TEXT_COLOR[0]
        )

        # Vẽ khung biển số
        for t in range(config.BOX_THICKNESS):
            draw.rectangle(
                [x - t, y - t, x + w + t, y + h + t],
                outline=box_color_rgb
            )

        # Vẽ nền nhãn
        draw.rectangle(
            [label_x, label_y,
             label_x + text_w + pad_x * 2,
             label_y + text_h + pad_y * 2],
            fill=label_bg_rgb
        )

        # Vẽ chữ nhãn
        draw.text(
            (label_x + pad_x, label_y + pad_y),
            label_text,
            fill=label_text_rgb,
            font=font
        )

    # Chuyển lại sang BGR cho OpenCV
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result


def show_result(img_bgr, window_title=None):
    """
    Hiển thị ảnh kết quả trong cửa sổ OpenCV.

    - Giữ đúng tỷ lệ ảnh
    - Cho phép resize cửa sổ
    - Nhấn Q, Esc hoặc đóng cửa sổ để thoát

    Tham số:
        img_bgr: ảnh BGR
        window_title: tiêu đề cửa sổ
    """
    if window_title is None:
        window_title = config.WINDOW_TITLE

    # Resize ảnh nếu quá lớn cho màn hình
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if w > config.MAX_DISPLAY_WIDTH or h > config.MAX_DISPLAY_HEIGHT:
        scale = min(
            config.MAX_DISPLAY_WIDTH / w,
            config.MAX_DISPLAY_HEIGHT / h
        )
        display_img = cv2.resize(
            img_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    else:
        display_img = img_bgr

    # Tạo cửa sổ cho phép resize
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, display_img.shape[1], display_img.shape[0])
    try:
        cv2.moveWindow(window_title, 530, 20)
    except Exception:
        pass
    cv2.imshow(window_title, display_img)

    # Chờ phím (Q hoặc Esc để đóng)
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # Q hoặc Esc
            break
        # Kiểm tra nếu cửa sổ bị đóng bằng nút X
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyWindow(window_title)


def show_result_nonblocking(img_bgr, window_title=None):
    """
    Hiển thị ảnh kết quả KHÔNG chặn luồng giao diện tkinter.
    Chỉ gọi imshow, không gọi waitKey vòng lặp.

    Trả về window_title để có thể đóng sau.
    """
    if window_title is None:
        window_title = config.WINDOW_TITLE

    h, w = img_bgr.shape[:2]
    scale = 1.0
    if w > config.MAX_DISPLAY_WIDTH or h > config.MAX_DISPLAY_HEIGHT:
        scale = min(
            config.MAX_DISPLAY_WIDTH / w,
            config.MAX_DISPLAY_HEIGHT / h
        )
        display_img = cv2.resize(
            img_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    else:
        display_img = img_bgr.copy()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, display_img.shape[1], display_img.shape[0])
    try:
        cv2.moveWindow(window_title, 530, 20)
    except Exception:
        pass
    cv2.imshow(window_title, display_img)
    cv2.waitKey(1)  # Cập nhật cửa sổ một lần

    return window_title


def create_comparison_image(images_dict):
    """
    Tạo ảnh so sánh nhiều trạng thái cạnh nhau.

    Tham số:
        images_dict: OrderedDict hoặc dict {tên: ảnh_bgr}

    Trả về:
        ảnh BGR ghép ngang
    """
    if not images_dict:
        return None

    items = list(images_dict.items())

    # Chuẩn hóa chiều cao
    target_h = 400
    resized = []
    for name, img in items:
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        r = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

        # Đảm bảo tất cả là 3 kênh
        if len(r.shape) == 2:
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

        resized.append((name, r))

    if not resized:
        return None

    # Thêm nhãn phía trên mỗi ảnh
    labeled = []
    font = _find_font(16)
    for name, img in resized:
        h, w = img.shape[:2]

        # Tạo vùng nhãn phía trên (30px)
        label_bar = np.zeros((30, w, 3), dtype=np.uint8)

        # Vẽ tên bằng PIL
        bar_rgb = cv2.cvtColor(label_bar, cv2.COLOR_BGR2RGB)
        pil_bar = Image.fromarray(bar_rgb)
        draw = ImageDraw.Draw(pil_bar)
        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = max(0, (w - text_w) // 2)
        draw.text((text_x, 5), name, fill=(255, 255, 255), font=font)
        label_bar = cv2.cvtColor(np.array(pil_bar), cv2.COLOR_RGB2BGR)

        combined = np.vstack([label_bar, img])
        labeled.append(combined)

    # Chuẩn hóa chiều cao (có thể khác do nhãn)
    max_h = max(img.shape[0] for img in labeled)
    padded = []
    for img in labeled:
        h, w = img.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    # Thêm đường kẻ phân cách (2px trắng)
    separator = np.ones((max_h, 2, 3), dtype=np.uint8) * 128
    result_parts = []
    for i, img in enumerate(padded):
        result_parts.append(img)
        if i < len(padded) - 1:
            result_parts.append(separator)

    return np.hstack(result_parts)
