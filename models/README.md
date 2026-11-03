# models/README.md
# Thư mục chứa trọng số mô hình

## EasyOCR
- Trọng số EasyOCR được tải tự động vào `~/.EasyOCR/` khi chạy lần đầu.
- Cần Internet cho lần tải đầu tiên (~100MB).
- Sau khi tải xong, chạy cục bộ hoàn toàn.

## Mô hình phát hiện biển số (tùy chọn)
- Hiện tại dùng OpenCV contour-based detector (không cần trọng số).
- Nếu muốn dùng mô hình phát hiện biển số chuyên dụng (VD: YOLO),
  đặt file trọng số vào thư mục này và cấu hình trong `config.py`.

### Yêu cầu khi dùng mô hình ngoài:
1. Ghi rõ nguồn và giấy phép
2. Kiểm tra kiến trúc và nhãn tương thích
3. Ghi phiên bản thư viện cần thiết
4. Không dùng mô hình YOLO tổng quát (COCO) vì không có lớp biển số
