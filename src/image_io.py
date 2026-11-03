# src/image_io.py
# =========================================================================
# MODULE ĐỌC VÀ CHUẨN HÓA ẢNH
# - Đọc ảnh từ đường dẫn (hỗ trợ tiếng Việt, khoảng trắng)
# - Xử lý hướng ảnh theo EXIF
# - Resize ảnh quá lớn
# - Kiểm tra tính hợp lệ
# =========================================================================

import os
import cv2
import numpy as np
from PIL import Image, ExifTags
import config


def _fix_orientation(pil_img):
    """
    Xoay ảnh theo thông tin EXIF (nếu có).
    Nhiều ảnh chụp từ điện thoại bị xoay ngang nếu không xử lý EXIF.
    """
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img

        # Tìm tag Orientation trong EXIF
        orientation_key = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orientation_key = tag
                break

        if orientation_key is None or orientation_key not in exif:
            return pil_img

        orientation = exif[orientation_key]

        # Xoay ảnh theo giá trị orientation
        if orientation == 2:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            pil_img = pil_img.rotate(180, expand=True)
        elif orientation == 4:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                270, expand=True
            )
        elif orientation == 6:
            pil_img = pil_img.rotate(270, expand=True)
        elif orientation == 7:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                90, expand=True
            )
        elif orientation == 8:
            pil_img = pil_img.rotate(90, expand=True)
    except Exception:
        # Nếu không đọc được EXIF thì bỏ qua
        pass

    return pil_img


def load_image(filepath):
    """
    Đọc ảnh từ đường dẫn, hỗ trợ tiếng Việt và khoảng trắng.

    Trả về:
        (image_bgr, scale_factor, error_message)
        - image_bgr: ảnh BGR (numpy array) hoặc None nếu lỗi
        - scale_factor: hệ số resize (1.0 nếu không resize)
        - error_message: chuỗi mô tả lỗi hoặc None nếu thành công
    """
    # Kiểm tra file tồn tại
    if not os.path.isfile(filepath):
        return None, 1.0, f"Không tìm thấy file: {filepath}"

    # Kiểm tra định dạng
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in config.SUPPORTED_FORMATS:
        return None, 1.0, (
            f"Định dạng '{ext}' không được hỗ trợ. "
            f"Chỉ hỗ trợ: {', '.join(config.SUPPORTED_FORMATS)}"
        )

    try:
        # Dùng Pillow để đọc (hỗ trợ đường dẫn Unicode tốt hơn OpenCV)
        pil_img = Image.open(filepath)

        # Xử lý hướng EXIF
        pil_img = _fix_orientation(pil_img)

        # Chuyển sang RGB
        pil_img = pil_img.convert("RGB")

        # Chuyển sang numpy array (RGB)
        img_rgb = np.array(pil_img)

        # Chuyển sang BGR cho OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    except Exception as e:
        return None, 1.0, f"Không thể đọc ảnh: {str(e)}"

    # Kiểm tra ảnh rỗng
    if img_bgr is None or img_bgr.size == 0:
        return None, 1.0, "Ảnh rỗng hoặc bị hỏng."

    h, w = img_bgr.shape[:2]

    # Kiểm tra kích thước tối thiểu
    if w < config.MIN_IMAGE_WIDTH or h < config.MIN_IMAGE_HEIGHT:
        return None, 1.0, (
            f"Ảnh quá nhỏ ({w}x{h}). "
            f"Yêu cầu tối thiểu {config.MIN_IMAGE_WIDTH}x{config.MIN_IMAGE_HEIGHT}."
        )

    # Resize nếu ảnh quá lớn
    scale_factor = 1.0
    max_dim = max(w, h)
    if max_dim > config.MAX_PROCESSING_DIMENSION:
        scale_factor = config.MAX_PROCESSING_DIMENSION / max_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img_bgr, scale_factor, None


def bgr_to_rgb(img_bgr):
    """Chuyển ảnh BGR (OpenCV) sang RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb):
    """Chuyển ảnh RGB sang BGR (OpenCV)."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def save_image(img_bgr, filepath):
    """
    Lưu ảnh BGR ra file. Hỗ trợ đường dẫn tiếng Việt.

    Trả về:
        (success, error_message)
    """
    try:
        # Tạo thư mục cha nếu chưa có
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        # Dùng Pillow để lưu (hỗ trợ Unicode path tốt hơn)
        img_rgb = bgr_to_rgb(img_bgr)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(filepath)
        return True, None

    except Exception as e:
        return False, f"Không thể lưu ảnh: {str(e)}"
