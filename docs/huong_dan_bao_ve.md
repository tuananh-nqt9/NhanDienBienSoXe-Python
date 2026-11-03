# Hướng dẫn bảo vệ bài tập lớn

## 1. Luồng chương trình

```
Người dùng chạy: python app.py
    ↓
Cửa sổ tkinter (bảng điều khiển) mở ra
    ↓
Người dùng chọn ảnh xe
    ↓
Nhấn "Nhận diện" → Worker thread chạy pipeline:
    ├── detector.py: Tìm vùng biển số (contour + chấm điểm)
    ├── preprocessing.py: Tạo biến thể tiền xử lý
    ├── color_classifier.py: Phân loại màu HSV
    ├── ocr_engine.py: EasyOCR đọc ký tự
    └── layout_classifier.py: Xác định 1/2 dòng
    ↓
Kết quả trả về luồng GUI → Hiển thị cửa sổ OpenCV
    ↓
Người dùng có thể: tạo nhiễu → giảm nhiễu → so sánh → lưu
```

## 2. Vai trò từng module

| Module | Vai trò | Tự viết? | Thư viện dùng |
|---|---|---|---|
| `app.py` | Điểm vào, khởi tạo | ✅ | tkinter |
| `config.py` | Cấu hình tập trung | ✅ | - |
| `gui.py` | Giao diện điều khiển | ✅ | tkinter |
| `image_io.py` | Đọc/lưu ảnh, EXIF | ✅ | Pillow, OpenCV |
| `detector.py` | Phát hiện biển số | ✅ | OpenCV |
| `preprocessing.py` | Tiền xử lý OCR | ✅ | OpenCV |
| `color_classifier.py` | Phân loại 4 màu | ✅ | OpenCV, NumPy |
| `layout_classifier.py` | Phân loại 1/2 dòng | ✅ | NumPy |
| `ocr_engine.py` | Nhận diện ký tự | ⚡ | EasyOCR (thư viện) |
| `noise.py` | Tạo 3 loại nhiễu | ✅ | NumPy, OpenCV |
| `denoise.py` | 4 bộ lọc giảm nhiễu | ✅ | OpenCV |
| `visualization.py` | Vẽ khung + nhãn | ✅ | Pillow, OpenCV |
| `pipeline.py` | Kết nối pipeline | ✅ | - |
| `evaluation.py` | Xuất CSV, tóm tắt | ✅ | csv (stdlib) |

**Phần tự triển khai**: Toàn bộ logic detector, phân loại màu, phân loại bố cục,
tạo/giảm nhiễu, giao diện, pipeline.

**Phần dùng thư viện có sẵn**: EasyOCR (mô hình OCR), OpenCV (xử lý ảnh cơ bản),
Pillow (vẽ chữ tiếng Việt).

## 3. Cách demo

### Demo 1: Nhận diện cơ bản
1. Mở chương trình: `python app.py`
2. Chọn ảnh xe rõ ràng
3. Nhấn "Nhận diện"
4. Chỉ ra: khung xanh, nhãn (biển số | màu | bố cục)

### Demo 2: Chọn vùng thủ công
1. Nhấn "Kéo chọn trên ảnh"
2. Kéo chuột chọn vùng biển
3. Nhấn "Nhận diện vùng thủ công"
4. So sánh kết quả với nhận diện tự động

### Demo 3: Tạo nhiễu và đánh giá
1. Nhận diện ảnh gốc trước
2. Chọn Gaussian noise, sigma=30
3. Nhấn "Tạo nhiễu"
4. Xem kết quả OCR thay đổi như thế nào
5. Tăng sigma=60, 80 → OCR dần thất bại

### Demo 4: Giảm nhiễu
1. Sau khi tạo nhiễu muối tiêu (ratio=0.1)
2. Chọn Median blur, ksize=5
3. Nhấn "Giảm nhiễu và nhận diện lại"
4. So sánh: ảnh gốc → nhiễu → giảm nhiễu

### Demo 5: Xuất kết quả
1. Nhấn "Xem so sánh" → Ảnh 3 trạng thái cạnh nhau
2. Nhấn "Lưu ảnh kết quả"
3. Nhấn "Xuất CSV"
4. Mở CSV bằng Excel → tiếng Việt hiển thị đúng

## 4. Câu hỏi thường gặp

### Q: Tại sao dùng HSV chứ không phải RGB để phân loại màu?
**A**: RGB bị ảnh hưởng mạnh bởi ánh sáng. Ví dụ, trắng dưới ánh đèn vàng
sẽ có giá trị R,G,B khác trắng ngoài trời. HSV tách riêng sắc màu (H)
khỏi độ sáng (V), nên phân loại ổn định hơn khi ánh sáng thay đổi.
Xem code: `src/color_classifier.py`, hàm `classify_plate_color()`.

### Q: Contour detection hoạt động thế nào?
**A**: Chương trình dùng Canny để tìm cạnh, rồi dùng morphology (CLOSE)
để nối các cạnh gần nhau thành vùng kín, rồi tìm đường bao (contour)
của các vùng kín đó. Mỗi contour được chấm điểm dựa trên tỷ lệ w/h,
diện tích, mật độ cạnh, và số lượng ký tự bên trong.
Xem code: `src/detector.py`, hàm `detect_plates_opencv()`.

### Q: CLAHE khác gì histogram equalization thường?
**A**: Histogram equalization thường áp dụng trên TOÀN bộ ảnh, có thể
làm mất chi tiết ở vùng đã tốt. CLAHE chia ảnh thành ô nhỏ (8×8) và
cân bằng từng ô → tăng tương phản cục bộ mà không phá vùng khác.
Xem code: `src/preprocessing.py`, hàm `create_ocr_variants()`.

### Q: Tại sao nhiễu mạnh làm OCR thất bại?
**A**: OCR dựa vào đặc trưng thị giác của ký tự: đường nét, góc, tỷ lệ.
Nhiễu mạnh phá vỡ các đặc trưng này. Gaussian noise σ>50 làm nhiễu lớn
hơn bề dày nét chữ. Salt-and-pepper >20% thay đổi quá nhiều pixel.
Blur kernel lớn hòa lẫn chữ với nền.

### Q: Median filter tốt cho muối tiêu vì sao?
**A**: Nhiễu muối tiêu tạo pixel cực trị (0 hoặc 255). Giá trị trung vị
(median) không bị ảnh hưởng bởi cực trị (robust statistic), nên pixel
nhiễu bị thay bằng giá trị bình thường xung quanh. Gaussian blur dùng
trung bình → pixel cực trị vẫn ảnh hưởng đến kết quả.
Xem code: `src/denoise.py`.

### Q: Gaussian blur trong tạo nhiễu và giảm nhiễu khác nhau không?
**A**: Cùng là Gaussian blur nhưng mục đích khác:
- Tạo nhiễu: MÔ PHỎNG ảnh bị mờ (thêm suy giảm)
- Giảm nhiễu: LÀM MƯỢT nhiễu ngẫu nhiên (giảm suy giảm)
Gaussian blur KHÔNG THỂ khôi phục ảnh đã bị mờ. Nó chỉ giảm nhiễu
ngẫu nhiên bằng cách làm mượt.

### Q: Phân loại 1 dòng / 2 dòng dựa vào đâu?
**A**: Kết hợp 2 dấu hiệu:
1. Tỷ lệ w/h của vùng biển (>3 thường 1 dòng, <1.8 thường 2 dòng)
2. Phân bố Y của các hộp ký tự từ OCR (nếu có khoảng trống > 15% chiều cao → 2 dòng)
Nếu không đủ bằng chứng → trả về "Không xác định".
Xem code: `src/layout_classifier.py`.

### Q: Tại sao cần worker thread?
**A**: OCR mất 2-10 giây. Nếu chạy trên luồng chính (GUI), giao diện
tkinter sẽ bị đóng băng (not responding). Worker thread chạy OCR ở nền,
khi xong gọi `root.after()` để cập nhật GUI an toàn trên luồng chính.
Xem code: `src/gui.py`, hàm `_run_detection()`.

### Q: Tại sao không dùng YOLO?
**A**: YOLO cần trọng số đã huấn luyện cho biển số Việt Nam. Mô hình YOLO
tổng quát (COCO) không có lớp "license_plate". Nếu có trọng số phù hợp,
có thể thay thế detector. Hiện tại dùng OpenCV contour-based vì không
cần trọng số ngoài và đủ tốt cho ảnh rõ ràng.

## 5. Hạn chế và hướng phát triển

### Đã kiểm thử
- Chọn ảnh, nhận diện, hiển thị kết quả
- Chọn vùng thủ công
- Tạo 3 loại nhiễu với seed tái tạo được
- 4 bộ lọc giảm nhiễu
- So sánh 3 trạng thái
- Lưu ảnh và xuất CSV
- Đóng cửa sổ an toàn

### Chưa kiểm thử đầy đủ
- Biển số tất cả tỉnh thành
- Biển xe ngoại giao, quân sự, đặc biệt
- Ảnh chụp ban đêm hoặc ngược sáng mạnh
- Nhiều biển số trong cùng một ảnh (> 3 biển)
- Hiệu suất với ảnh > 4000px

### Hạn chế
- Detector dựa trên contour, thất bại khi nền phức tạp
- OCR phụ thuộc chất lượng ảnh
- Phân loại màu dùng ngưỡng cố định
- Chưa hỗ trợ video, camera
- Thời gian xử lý trên CPU có thể chậm

### Hướng phát triển
- Thêm mô hình deep learning cho detector (YOLO, SSD)
- Fine-tune OCR cho biển Việt Nam
- Phân loại màu bằng machine learning
- Hỗ trợ batch processing nhiều ảnh
