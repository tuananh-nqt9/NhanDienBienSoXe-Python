# Giải thích thuật toán

## 1. Tổng quan luồng xử lý

```
Ảnh đầu vào
→ Đọc và chuẩn hóa (EXIF, resize)
→ Phát hiện vùng biển số (contour-based)
→ Cắt biển và thêm lề
→ Phân loại màu nền (HSV)
→ Tiền xử lý phục vụ OCR (CLAHE, threshold)
→ Nhận diện ký tự (EasyOCR)
→ Phân loại bố cục (1/2 dòng)
→ Vẽ khung, nhãn và hiển thị
```

## 2. Phát hiện biển số (detector.py)

### Vì sao dùng contour?

Biển số xe là vùng chữ nhật có đặc điểm:
- Tỷ lệ chiều rộng/chiều cao đặc trưng
- Nhiều cạnh (do chữ và số)
- Tương phản rõ với nền xe

Contour detection tìm đường bao quanh các vùng có cạnh mạnh,
phù hợp để tìm vùng chữ nhật trong ảnh.

### Các bước chi tiết

**Bước 1: Bilateral filter**
- Giảm nhiễu NHƯNG giữ cạnh rõ
- Khác với Gaussian blur thường làm mờ cạnh

**Bước 2: Canny edge detection**
- Phát hiện cạnh bằng gradient
- Dùng hai ngưỡng (low=50, high=200) để lọc cạnh mạnh/yếu
- Cạnh yếu chỉ giữ nếu nối với cạnh mạnh

**Bước 3: Morphology (đóng khe hở)**
- Kernel hình chữ nhật 17×5 pixel
- Phép CLOSE = dilate rồi erode
- Nối các cạnh gần nhau thành vùng liên tục
- Kernel ngang dài hơn dọc vì biển số thường nằm ngang

**Bước 4: Tìm contour**
- Duyệt ảnh nhị phân tìm đường bao
- Lấy bounding rectangle cho mỗi contour

**Bước 5: Chấm điểm ứng viên**

Mỗi vùng được chấm điểm dựa trên:
| Dấu hiệu | Điểm tối đa | Giải thích |
|---|---|---|
| Tỷ lệ w/h phù hợp | 35 | Biển 1 dòng: 3-5.5, biển 2 dòng: 1.2-2.3 |
| Diện tích hợp lý | 30 | 0.2% - 30% diện tích ảnh |
| Mật độ cạnh | 25 | Biển số có nhiều cạnh do ký tự |
| Dấu hiệu ký tự | 30 | Đếm thành phần liên thông có kích thước hợp lý |
| Độ chữ nhật | 10 | Diện tích contour / diện tích bbox > 0.7 |

→ KHÔNG kết luận biển số chỉ dựa trên một dấu hiệu duy nhất.

**Bước 6: Non-Maximum Suppression (NMS)**
- Nếu hai vùng chồng nhau > 30% (IoU), chỉ giữ vùng điểm cao hơn
- Tránh phát hiện trùng

## 3. Phân loại màu (color_classifier.py)

### Vì sao dùng HSV?

Không gian màu RGB bị ảnh hưởng mạnh bởi ánh sáng.
HSV tách biệt:
- **H (Hue)**: Sắc màu (0-179 trong OpenCV)
- **S (Saturation)**: Độ bão hòa
- **V (Value)**: Độ sáng

→ Phân loại màu dễ hơn vì chỉ cần quan tâm H và S,
ít bị ảnh hưởng bởi thay đổi ánh sáng.

### Khoảng HSV cho từng màu

| Màu | Hue | Saturation | Value |
|---|---|---|---|
| Trắng | Bất kỳ | < 60 | > 160 |
| Vàng | 15-45 | > 60 | > 100 |
| Xanh dương | 90-135 | > 60 | > 50 |
| Đỏ | 0-10 HOẶC 160-179 | > 60 | > 50 |

> Đỏ dùng **hai khoảng Hue** vì trong hệ HSV, đỏ nằm ở cả hai đầu
> vòng tròn màu (0° và 360°, tức 0 và 179 trong OpenCV).

### Xử lý viền và ký tự
- Chỉ lấy 70% vùng trung tâm biển (loại viền)
- Viền và ký tự thường khác màu nền

### Khi nào trả về "Không xác định"?
- Tỷ lệ pixel cao nhất < 15%
- Hai màu hàng đầu cách nhau < 8%

## 4. CLAHE và Threshold (preprocessing.py)

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Tăng tương phản **cục bộ** (từng vùng nhỏ 8×8 pixel)
- Khác histogram equalization thường (toàn ảnh)
- Giúp hiện rõ ký tự khi biển bị chói một phần hoặc tối một phần

### Otsu threshold
- Tự động tìm ngưỡng phân tách nền và chữ
- Dựa trên phân tích histogram (tìm ngưỡng giảm thiểu phương sai trong lớp)
- Tốt khi nền và chữ có độ sáng khác biệt rõ

### Adaptive threshold
- Tính ngưỡng riêng cho từng vùng nhỏ
- Tốt khi biển có ánh sáng không đều
- Kernel Gaussian 11×11

### Tại sao tạo nhiều biến thể?
- Mỗi biến thể phù hợp với điều kiện khác nhau
- Chương trình thử OCR trên tất cả, chọn kết quả tốt nhất
- Tiêu chí: **số ký tự hợp lệ × điểm tin cậy trung bình**

## 5. Phân loại 1 dòng / 2 dòng (layout_classifier.py)

### Kết hợp 2 dấu hiệu

**Dấu hiệu 1: Tỷ lệ w/h**
- w/h > 3.0 → khả năng cao 1 dòng
- w/h < 1.8 → khả năng cao 2 dòng
- 1.8 - 3.0 → không rõ, cần thêm bằng chứng

**Dấu hiệu 2: Phân bố ký tự**
- Tính tọa độ Y trung tâm của mỗi hộp ký tự từ OCR
- Sắp xếp theo Y và tìm khoảng cách lớn nhất
- Nếu khoảng cách > 15% chiều cao biển → 2 dòng

### Ưu tiên khi mâu thuẫn
Nếu hai dấu hiệu không đồng thuận, ưu tiên dấu hiệu từ OCR boxes
vì nó dựa trên vị trí thực tế của ký tự.

## 6. OCR và sắp xếp ký tự (ocr_engine.py)

### EasyOCR
- Mô hình deep learning đã được huấn luyện sẵn
- Chạy hoàn toàn cục bộ (CPU)
- Trả về: vị trí hộp, chuỗi ký tự, điểm tin cậy

### Sắp xếp theo bố cục biển
1. Tính Y trung tâm mỗi hộp ký tự
2. Phân nhóm theo chiều dọc (cùng dòng nếu Y gần nhau)
3. Dòng trên đọc trước, dòng dưới đọc sau
4. Trong mỗi dòng: từ trái sang phải

### Chuẩn hóa
- Chuyển chữ IN HOA
- Giữ A-Z, 0-9, dấu chấm, dấu gạch
- **KHÔNG** tự động thay O↔0, I↔1, B↔8

## 7. Tạo nhiễu (noise.py)

### Gaussian noise
- Cộng giá trị ngẫu nhiên N(0, σ) vào mỗi pixel
- σ lớn → nhiễu mạnh
- Mô phỏng nhiễu cảm biến máy ảnh

### Salt-and-pepper noise
- Thay ngẫu nhiên một số pixel thành trắng (255) hoặc đen (0)
- Tỷ lệ = phần trăm pixel bị ảnh hưởng
- Mô phỏng lỗi truyền tín hiệu

### Gaussian blur
- **Không phải nhiễu ngẫu nhiên** mà là làm mờ
- Mỗi pixel = trung bình có trọng số vùng lân cận
- Kernel lớn → mờ nhiều hơn
- Mô phỏng ảnh mất nét

### Vì sao nhiễu mạnh có thể làm mất thông tin?
- Nhiễu Gaussian σ > 50: pixel gốc bị thay đổi lớn, cạnh ký tự bị phá
- S&P ratio > 20%: quá nhiều pixel bị thay thế hoàn toàn
- Blur kernel > 15: chi tiết nhỏ (nét chữ) bị hòa lẫn với nền
→ OCR không đọc được vì ký tự không còn đủ đặc trưng

## 8. Giảm nhiễu (denoise.py)

### Median blur (tốt cho muối tiêu)
- Thay mỗi pixel bằng **giá trị trung vị** của vùng lân cận
- Loại bỏ hiệu quả các pixel cực trị (trắng/đen ngẫu nhiên)
- Giữ cạnh tốt hơn mean filter

### Gaussian blur (nhiễu Gaussian nhẹ)
- Trung bình có trọng số Gaussian
- Làm mượt nhiễu nhưng cũng làm mờ cạnh
- **KHÔNG khôi phục** được ảnh đã bị mờ

### Bilateral filter (giữ cạnh)
- Kết hợp Gaussian trong không gian TỌA ĐỘ và không gian MÀU
- Pixel có màu khác biệt nhiều → ít ảnh hưởng → giữ cạnh
- Chậm hơn Gaussian nhưng giữ chi tiết tốt hơn

### Non-local Means (nhiễu mạnh)
- So sánh từng vùng nhỏ (patch) với các vùng tương tự trong ảnh
- Tìm pixel giống nhau ở xa → trung bình thông minh hơn
- Cho kết quả tốt nhưng **chậm nhất**
