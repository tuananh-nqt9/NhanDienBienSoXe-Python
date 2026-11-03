# src/denoise.py
# =========================================================================
# MODULE GIẢM NHIỄU
# - Median blur: tốt cho nhiễu muối tiêu
# - Gaussian blur: làm mượt tổng quát
# - Bilateral filter: giảm nhiễu giữ cạnh
# - Non-local Means: giảm nhiễu mạnh nhưng chậm hơn
# =========================================================================

import cv2
import config


def apply_median_blur(image, ksize=None):
    """
    Giảm nhiễu bằng Median blur.

    Đặc biệt hiệu quả với nhiễu muối tiêu (salt-and-pepper).
    Thay thế mỗi pixel bằng giá trị trung vị của vùng lân cận.

    Tham số:
        image: ảnh numpy array (uint8)
        ksize: kích thước kernel (phải lẻ)

    Trả về:
        ảnh đã giảm nhiễu
    """
    if ksize is None:
        ksize = config.DEFAULT_MEDIAN_KSIZE

    # Đảm bảo ksize là số lẻ và >= 3
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    return cv2.medianBlur(image, ksize)


def apply_gaussian_blur(image, ksize=None):
    """
    Giảm nhiễu bằng Gaussian blur.

    Là bộ lọc thông thấp, làm mượt ảnh nhưng cũng làm mờ cạnh.
    Phù hợp cho nhiễu Gaussian nhẹ.

    LƯU Ý: Gaussian blur KHÔNG khôi phục được ảnh đã bị mờ.
    Nó chỉ giúp giảm nhiễu ngẫu nhiên.

    Tham số:
        image: ảnh numpy array (uint8)
        ksize: kích thước kernel (phải lẻ)
    """
    if ksize is None:
        ksize = config.DEFAULT_BLUR_KSIZE

    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)

    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_bilateral_filter(image, d=None, sigma_color=None, sigma_space=None):
    """
    Giảm nhiễu bằng Bilateral filter.

    Ưu điểm: giữ được cạnh (edge-preserving) trong khi làm mượt
    vùng phẳng. Tốt khi cần giữ rõ ký tự trên biển số.

    Tham số:
        image: ảnh numpy array (uint8)
        d: đường kính lân cận pixel
        sigma_color: sigma trong không gian màu
        sigma_space: sigma trong không gian tọa độ
    """
    if d is None:
        d = config.DEFAULT_BILATERAL_D
    if sigma_color is None:
        sigma_color = config.DEFAULT_BILATERAL_SIGMA_COLOR
    if sigma_space is None:
        sigma_space = config.DEFAULT_BILATERAL_SIGMA_SPACE

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_nlm(image, h=None, template_window=None, search_window=None):
    """
    Giảm nhiễu bằng Non-local Means.

    Thuật toán so sánh các vùng nhỏ (patch) trong ảnh để tìm
    pixel tương tự, cho kết quả tốt hơn bộ lọc cục bộ nhưng chậm hơn.

    Phù hợp cho nhiễu Gaussian mức trung bình đến mạnh.

    Tham số:
        image: ảnh numpy array (uint8)
        h: cường độ lọc (lớn hơn = mạnh hơn nhưng mất chi tiết)
        template_window: kích thước cửa sổ template (phải lẻ)
        search_window: kích thước cửa sổ tìm kiếm (phải lẻ)
    """
    if h is None:
        h = config.DEFAULT_NLM_H
    if template_window is None:
        template_window = config.DEFAULT_NLM_TEMPLATE_WINDOW
    if search_window is None:
        search_window = config.DEFAULT_NLM_SEARCH_WINDOW

    # Kiểm tra ảnh màu hay xám
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h, h,
            template_window, search_window
        )
    else:
        return cv2.fastNlMeansDenoising(
            image, None, h,
            template_window, search_window
        )


def apply_denoise(image, method, params=None):
    """
    Áp dụng bộ lọc giảm nhiễu theo tên phương pháp.

    Tham số:
        image: ảnh (uint8)
        method: "median" / "gaussian" / "bilateral" / "nlm"
        params: dict tham số tùy chọn

    Trả về:
        ảnh đã giảm nhiễu
    """
    if params is None:
        params = {}

    if method == "median":
        return apply_median_blur(image, ksize=params.get("ksize"))
    elif method == "gaussian":
        return apply_gaussian_blur(image, ksize=params.get("ksize"))
    elif method == "bilateral":
        return apply_bilateral_filter(
            image,
            d=params.get("d"),
            sigma_color=params.get("sigma_color"),
            sigma_space=params.get("sigma_space"),
        )
    elif method == "nlm":
        return apply_nlm(
            image,
            h=params.get("h"),
            template_window=params.get("template_window"),
            search_window=params.get("search_window"),
        )
    else:
        return image.copy()
