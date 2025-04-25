# Giao diện người dùng cho ứng dụng Synthetic CT Generator

Thư mục này chứa các module liên quan đến giao diện người dùng đồ họa (GUI) của ứng dụng Synthetic CT Generator.

## Cấu trúc

- `run.py`: Điểm khởi đầu cho việc chạy ứng dụng GUI
- `main_window.py`: Triển khai của cửa sổ chính và các tính năng cơ bản
- `enhanced_gui.py`: Phiên bản nâng cao của GUI với nhiều tính năng bổ sung

## Khởi động giao diện người dùng

Để khởi động giao diện người dùng, thực hiện một trong các lệnh sau:

```bash
# Từ thư mục gốc của dự án
python app/gui/run.py

# Hoặc thông qua module main
python app/main.py gui

# Với tùy chọn theme
python app/main.py gui --theme dark
```

## Tính năng

Giao diện người dùng hỗ trợ đầy đủ quy trình chuyển đổi MRI sang CT tổng hợp:

1. **Tiền xử lý ảnh MRI**
   - Hiệu chỉnh trường bias N4
   - Giảm nhiễu ảnh
   - Chuẩn hóa cường độ tín hiệu

2. **Phân đoạn mô**
   - Phân đoạn các cấu trúc xương
   - Phân đoạn mô mềm
   - Phân đoạn khoang khí

3. **Chuyển đổi MRI sang CT tổng hợp**
   - Mô hình CNN
   - Mô hình GAN
   - Phương pháp dựa trên Atlas

4. **Đánh giá kết quả**
   - So sánh với CT thực
   - Tính toán các metrics: MAE, MSE, PSNR, SSIM
   - Hiển thị trực quan kết quả

5. **Huấn luyện mô hình**
   - Huấn luyện các mô hình chuyển đổi mới
   - Tùy chỉnh tham số huấn luyện
   - Theo dõi quá trình huấn luyện

## Hướng dẫn sử dụng

### Tiền xử lý MRI

1. Chọn tab "Preprocessing"
2. Nhấn "Select MRI" để chọn file MRI đầu vào
3. Điều chỉnh các tham số tiền xử lý (nếu cần)
4. Nhấn "Preprocess" để bắt đầu tiền xử lý
5. Xem kết quả tiền xử lý trong viewer

### Phân đoạn mô

1. Chọn tab "Segmentation"
2. Chọn MRI đã tiền xử lý (hoặc sử dụng kết quả từ bước tiền xử lý)
3. Chọn phương pháp phân đoạn và vùng giải phẫu
4. Nhấn "Segment" để bắt đầu phân đoạn
5. Xem kết quả phân đoạn trong viewer

### Chuyển đổi MRI sang CT

1. Chọn tab "Conversion"
2. Chọn MRI đã tiền xử lý và kết quả phân đoạn (nếu có)
3. Chọn mô hình chuyển đổi và vùng giải phẫu
4. Nhấn "Convert" để bắt đầu chuyển đổi
5. Xem kết quả CT tổng hợp trong viewer

### Huấn luyện mô hình

1. Chọn tab "Training"
2. Chọn thư mục chứa dữ liệu huấn luyện (MRI và CT thực)
3. Điều chỉnh các tham số huấn luyện
4. Chọn thư mục lưu mô hình
5. Nhấn "Train" để bắt đầu huấn luyện

### Đánh giá kết quả

1. Chọn tab "Evaluation"
2. Chọn CT tổng hợp và CT thực để so sánh
3. Nhấn "Evaluate" để bắt đầu đánh giá
4. Xem kết quả đánh giá chi tiết

## Tùy chỉnh giao diện

Giao diện người dùng hỗ trợ các tùy chọn chủ đề:

- **Light**: Chủ đề sáng (mặc định)
- **Dark**: Chủ đề tối
- **System**: Sử dụng chủ đề hệ thống

Để thay đổi chủ đề:

```bash
python app/main.py gui --theme dark
```

Hoặc thay đổi trong file cấu hình `configs/gui_config.yaml`.

## Xử lý lỗi

Trong trường hợp gặp lỗi:

1. Kiểm tra log chi tiết trong thư mục `logs/`
2. Đảm bảo các phụ thuộc đã được cài đặt đầy đủ
3. Kiểm tra định dạng file đầu vào (DICOM hoặc NIfTI)
4. Điều chỉnh các tham số tiền xử lý hoặc chuyển đổi

## Mở rộng GUI

Để mở rộng giao diện người dùng, hãy xem xét các file:

- `enhanced_gui.py`: Thêm các tính năng mới
- `app/utils/config_utils.py`: Thêm các tham số cấu hình
- `app/core/`: Triển khai các chức năng mới

## API cho nhà phát triển

Nhà phát triển có thể tích hợp các thành phần GUI vào ứng dụng của họ:

```python
from app.gui.enhanced_gui import EnhancedMainWindow
from PySide6.QtWidgets import QApplication

app = QApplication([])
window = EnhancedMainWindow()
window.show()
app.exec()
```

Xem `run.py` để biết thêm chi tiết về cách tích hợp. 