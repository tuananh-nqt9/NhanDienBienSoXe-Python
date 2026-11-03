# config.py
# =========================================================================
# CẤU HÌNH CHUNG CHO DỰ ÁN NHẬN DIỆN BIỂN SỐ XE
# Tất cả ngưỡng và tham số được tập trung tại đây để dễ điều chỉnh.
# =========================================================================

import os

# -------------------------------------------------------------------------
# ĐƯỜNG DẪN
# -------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
HOT_FOLDER_DIR = os.path.join(PROJECT_DIR, "test_live")
LIVE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "live")

# -------------------------------------------------------------------------
# GIỚI HẠN ẢNH ĐẦU VÀO
# -------------------------------------------------------------------------
# Kích thước tối thiểu ảnh (pixel) - ảnh nhỏ hơn sẽ bị từ chối
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# Kích thước tối đa để xử lý - ảnh lớn hơn sẽ được resize
# (giữ tỷ lệ, cạnh dài nhất không vượt giá trị này)
MAX_PROCESSING_DIMENSION = 1600

# Định dạng ảnh được hỗ trợ
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")

# -------------------------------------------------------------------------
# PHÁT HIỆN BIỂN SỐ (DETECTOR) - OpenCV contour-based
# -------------------------------------------------------------------------
# Kích thước ảnh xám sau resize để phát hiện cạnh (cạnh dài nhất)
DETECTOR_RESIZE_MAX = 800

# Tham số Canny edge detection
CANNY_LOW = 50       # Ngưỡng thấp của Canny
CANNY_HIGH = 200     # Ngưỡng cao của Canny

# Tham số morphology (đóng các khe hở trên cạnh)
MORPH_KERNEL_WIDTH = 17   # Chiều rộng kernel morphology
MORPH_KERNEL_HEIGHT = 5   # Chiều cao kernel morphology

# Tỷ lệ chiều rộng/chiều cao hợp lệ cho biển số
# Biển 1 dòng: thường 3.0 - 6.0
# Biển 2 dòng: thường 1.0 - 2.5
MIN_ASPECT_RATIO = 1.0    # Tỷ lệ w/h tối thiểu
MAX_ASPECT_RATIO = 6.5    # Tỷ lệ w/h tối đa

# Tỷ lệ diện tích vùng ứng viên so với diện tích ảnh
MIN_AREA_RATIO = 0.002    # Diện tích tối thiểu (0.2% ảnh)
MAX_AREA_RATIO = 0.98      # Diện tích tối đa (hỗ trợ cả ảnh cận cảnh / ảnh cắt)

# Độ chữ nhật tối thiểu (diện tích contour / diện tích bounding rect)
MIN_RECTANGULARITY = 0.4

# Mật độ cạnh tối thiểu trong vùng ứng viên
# (tỷ lệ pixel cạnh / tổng pixel vùng)
MIN_EDGE_DENSITY = 0.05

# Ngưỡng IoU để loại vùng trùng lặp (Non-Maximum Suppression)
NMS_IOU_THRESHOLD = 0.3

# Số ứng viên tối đa giữ lại sau lọc
MAX_CANDIDATES = 5

# Minimum character evidence required for an automatically detected plate region.
DETECTOR_MIN_CHAR_CANDIDATES = 4

# Lề khi cắt vùng biển (tỷ lệ so với kích thước biển)
CROP_MARGIN_RATIO = 0.04   # 4% mỗi phía

# -------------------------------------------------------------------------
# PHÂN LOẠI MÀU (HSV)
# -------------------------------------------------------------------------
# Khoảng HSV cho từng màu nền biển
# H: 0-179 (OpenCV), S: 0-255, V: 0-255

# Trắng: bão hòa thấp, sáng cao
WHITE_S_MAX = 60          # Bão hòa tối đa
WHITE_V_MIN = 160         # Độ sáng tối thiểu

# Vàng: H khoảng 15-40
YELLOW_H_MIN = 15
YELLOW_H_MAX = 45
YELLOW_S_MIN = 60
YELLOW_V_MIN = 100

# Xanh dương: H khoảng 90-130
BLUE_H_MIN = 90
BLUE_H_MAX = 135
BLUE_S_MIN = 60
BLUE_V_MIN = 50

# Đỏ: hai khoảng H (0-10 và 160-179) vì đỏ nằm ở biên Hue
RED_H_MIN1 = 0
RED_H_MAX1 = 10
RED_H_MIN2 = 160
RED_H_MAX2 = 179
RED_S_MIN = 60
RED_V_MIN = 50

# Tỷ lệ pixel tối thiểu để kết luận màu
COLOR_MIN_RATIO = 0.15    # Ít nhất 15% pixel thuộc màu đó

# Khoảng cách tối thiểu giữa hai màu hàng đầu để kết luận
# Nếu hai màu quá gần nhau → "Không xác định"
COLOR_DIFF_THRESHOLD = 0.08

# Tỷ lệ thu nhỏ vùng biển khi phân loại màu
# (loại bỏ viền bằng cách chỉ lấy phần trung tâm)
COLOR_INNER_RATIO = 0.7   # Lấy 70% phần giữa

# -------------------------------------------------------------------------
# PHÂN LOẠI BỐ CỤC (1 DÒNG / 2 DÒNG)
# -------------------------------------------------------------------------
# Ngưỡng tỷ lệ w/h để hỗ trợ phân loại
# Biển 1 dòng thường rộng hơn (w/h > 2.5)
# Biển 2 dòng thường vuông hơn (w/h < 2.5)
LAYOUT_ASPECT_THRESHOLD = 2.5

# Khoảng cách dọc tối thiểu giữa hai nhóm ký tự
# (tỷ lệ so với chiều cao biển) để coi là 2 dòng
LAYOUT_VERTICAL_GAP_RATIO = 0.15

# -------------------------------------------------------------------------
# OCR (EasyOCR)
# -------------------------------------------------------------------------
# Ngôn ngữ OCR
OCR_LANGUAGES = ["en"]

# Ký tự cho phép trên biển số Việt Nam
OCR_ALLOWLIST = "ABCDEFGHKLMNPRSTUVXYZ0123456789-."

# Điểm tin cậy tối thiểu cho một ký tự
OCR_MIN_CONFIDENCE = 0.1

# Số biến thể tiền xử lý tối đa để thử OCR (thử cả sharpen, clahe, otsu, adaptive, bilateral)
MAX_OCR_VARIANTS = 6

# -------------------------------------------------------------------------
# TẠO NHIỄU
# -------------------------------------------------------------------------
# Giá trị mặc định cho nhiễu Gaussian
DEFAULT_GAUSSIAN_SIGMA = 25.0

# Giá trị mặc định cho nhiễu muối tiêu
DEFAULT_SP_RATIO = 0.05     # 5% pixel bị ảnh hưởng

# Giá trị mặc định cho Gaussian blur
DEFAULT_BLUR_KSIZE = 5      # Kích thước kernel (phải lẻ)

# Giới hạn thông số nhiễu
MAX_GAUSSIAN_SIGMA = 100.0
MAX_SP_RATIO = 0.5          # Tối đa 50% pixel
MAX_BLUR_KSIZE = 31         # Kernel tối đa

# -------------------------------------------------------------------------
# GIẢM NHIỄU
# -------------------------------------------------------------------------
# Kích thước kernel mặc định cho median blur
DEFAULT_MEDIAN_KSIZE = 5

# Tham số mặc định cho bilateral filter
DEFAULT_BILATERAL_D = 9          # Đường kính lân cận
DEFAULT_BILATERAL_SIGMA_COLOR = 75
DEFAULT_BILATERAL_SIGMA_SPACE = 75

# Tham số mặc định cho Non-local Means
DEFAULT_NLM_H = 10               # Cường độ lọc
DEFAULT_NLM_TEMPLATE_WINDOW = 7
DEFAULT_NLM_SEARCH_WINDOW = 21

# -------------------------------------------------------------------------
# HIỂN THỊ KẾT QUẢ
# -------------------------------------------------------------------------
# Màu khung và nền nhãn (BGR cho OpenCV)
LABEL_BG_COLOR = (0, 255, 0)     # Xanh lá
LABEL_TEXT_COLOR = (0, 0, 0)     # Đen
BOX_COLOR = (0, 255, 0)         # Xanh lá
BOX_THICKNESS = 3

# Cỡ chữ mặc định cho nhãn dòng màu xanh lá (pixel trên cửa sổ hiển thị)
LABEL_FONT_SIZE = 20

# Kích thước tối đa cửa sổ hiển thị ảnh (thu nhỏ vừa phải để không che khuất bảng điều khiển)
MAX_DISPLAY_WIDTH = 560
MAX_DISPLAY_HEIGHT = 420

# Tên cửa sổ
WINDOW_TITLE = "Nhan dien bien so xe"

# -------------------------------------------------------------------------
# FONT CHỮ TIẾNG VIỆT (Windows)
# -------------------------------------------------------------------------
# Danh sách font ưu tiên tìm trên Windows (ưu tiên Arial Bold để chữ đậm rõ)
FONT_CANDIDATES = [
    "arialbd.ttf",
    "arial.ttf",
    "tahomabd.ttf",
    "tahoma.ttf",
    "segoeui.ttf",
    "calibri.ttf",
    "times.ttf",
]

# Đường dẫn thư mục font Windows
WINDOWS_FONT_DIR = r"C:\Windows\Fonts"
