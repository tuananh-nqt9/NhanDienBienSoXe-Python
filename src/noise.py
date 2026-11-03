# src/noise.py
# =========================================================================
# MODULE TẠO NHIỄU
# - Gaussian noise: nhiễu ngẫu nhiên phân phối chuẩn
# - Salt-and-pepper noise: nhiễu muối tiêu (pixel trắng/đen ngẫu nhiên)
# - Gaussian blur: làm mờ ảnh (không phải nhiễu ngẫu nhiên)
# =========================================================================

import cv2
import numpy as np
import config


def add_gaussian_noise(image, sigma=None, seed=None):
    """
    Thêm nhiễu Gaussian vào ảnh.

    Nhiễu Gaussian mô phỏng nhiễu cảm biến máy ảnh.
    Mỗi pixel được cộng thêm giá trị ngẫu nhiên từ phân phối N(0, sigma).

    Tham số:
        image: ảnh numpy array (uint8)
        sigma: độ lệch chuẩn (lớn hơn = nhiễu mạnh hơn)
        seed: random seed để tái tạo cùng mẫu nhiễu

    Trả về:
        ảnh đã thêm nhiễu (uint8)
    """
    if sigma is None:
        sigma = config.DEFAULT_GAUSSIAN_SIGMA

    if sigma <= 0:
        return image.copy()

    # Đặt seed để tái tạo được
    rng = np.random.RandomState(seed)

    # Tạo nhiễu với kiểu float để tính toán chính xác
    noise = rng.normal(0, sigma, image.shape).astype(np.float64)

    # Cộng nhiễu và clip về [0, 255]
    noisy = image.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def add_salt_pepper_noise(image, ratio=None, seed=None):
    """
    Thêm nhiễu muối tiêu vào ảnh.

    Nhiễu muối tiêu thay đổi ngẫu nhiên một số pixel
    thành trắng (muối) hoặc đen (tiêu).

    Tham số:
        image: ảnh numpy array (uint8)
        ratio: tỷ lệ pixel bị ảnh hưởng (0.0 - 1.0)
        seed: random seed

    Trả về:
        ảnh đã thêm nhiễu (uint8)
    """
    if ratio is None:
        ratio = config.DEFAULT_SP_RATIO

    if ratio <= 0:
        return image.copy()

    ratio = min(ratio, config.MAX_SP_RATIO)

    rng = np.random.RandomState(seed)
    noisy = image.copy()

    # Tổng số pixel
    total_pixels = image.shape[0] * image.shape[1]
    n_affected = int(total_pixels * ratio)

    # Thêm muối (pixel trắng)
    n_salt = n_affected // 2
    salt_coords = (
        rng.randint(0, image.shape[0], n_salt),
        rng.randint(0, image.shape[1], n_salt),
    )
    noisy[salt_coords[0], salt_coords[1]] = 255

    # Thêm tiêu (pixel đen)
    n_pepper = n_affected - n_salt
    pepper_coords = (
        rng.randint(0, image.shape[0], n_pepper),
        rng.randint(0, image.shape[1], n_pepper),
    )
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    return noisy


def add_gaussian_blur(image, ksize=None):
    """
    Áp dụng Gaussian blur (làm mờ ảnh).

    LƯU Ý: Gaussian blur KHÔNG phải nhiễu ngẫu nhiên.
    Nó mô phỏng ảnh mất nét do lấy nét sai hoặc rung camera.

    Tham số:
        image: ảnh numpy array (uint8)
        ksize: kích thước kernel (phải lẻ, lớn hơn = mờ hơn)

    Trả về:
        ảnh đã làm mờ (uint8)
    """
    if ksize is None:
        ksize = config.DEFAULT_BLUR_KSIZE

    # Đảm bảo ksize là số lẻ
    if ksize % 2 == 0:
        ksize += 1

    ksize = max(1, min(ksize, config.MAX_BLUR_KSIZE))

    if ksize <= 1:
        return image.copy()

    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def reduce_brightness(image, factor=0.4):
    """
    Giảm độ sáng ảnh để mô phỏng điều kiện ban đêm hoặc thiếu sáng.

    Tham số:
        image: ảnh numpy array (uint8)
        factor: hệ số độ sáng (0.05 - 1.0), nhỏ hơn = tối hơn. Mặc định 0.4.

    Trả về:
        ảnh đã giảm sáng (uint8)
    """
    if factor is None:
        factor = 0.4
    factor = max(0.05, min(float(factor), 1.0))

    if factor >= 0.99:
        return image.copy()

    # Chuyển sang không gian màu HSV và giảm kênh V (Value - độ sáng)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_noise(image, noise_type, params=None, scope="full", roi_bbox=None):
    """
    Áp dụng nhiễu theo loại và phạm vi đã chọn.

    Tham số:
        image: ảnh nguồn (uint8)
        noise_type: "gaussian" / "salt_pepper" / "blur" / "brightness" / "night"
        params: dict tham số (sigma, ratio, ksize, seed, factor)
        scope: "full" (toàn ảnh) hoặc "roi" (chỉ vùng biển)
        roi_bbox: (x, y, w, h) khi scope="roi"

    Trả về:
        ảnh đã thêm nhiễu (uint8)
    """
    if params is None:
        params = {}

    def _apply_single(img):
        if noise_type == "gaussian":
            return add_gaussian_noise(
                img, sigma=params.get("sigma"), seed=params.get("seed")
            )
        elif noise_type == "salt_pepper":
            return add_salt_pepper_noise(
                img, ratio=params.get("ratio"), seed=params.get("seed")
            )
        elif noise_type == "blur":
            return add_gaussian_blur(img, ksize=params.get("ksize"))
        elif noise_type in ("brightness", "night"):
            return reduce_brightness(img, factor=params.get("factor", 0.4))
        return img.copy()

    if scope == "roi" and roi_bbox is not None:
        # Chỉ tạo nhiễu trong vùng biển
        x, y, w, h = roi_bbox
        result = image.copy()
        roi = result[y:y+h, x:x+w]
        result[y:y+h, x:x+w] = _apply_single(roi)
        return result
    else:
        return _apply_single(image)

