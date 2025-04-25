# Synthetic-CT-from-MRI-for-radiotherapy

Dự án chuyển đổi ảnh MRI thành ảnh CT tổng hợp (synthetic CT hoặc sCT) cho lập kế hoạch xạ trị, nhằm giảm thiểu liều xạ không mong muốn vào cơ thể bệnh nhân.

[English Version](#english-version)

## Tổng quan

Trong lập kế hoạch xạ trị, ảnh CT là cần thiết cho tính toán liều xạ do cung cấp thông tin về mật độ electron của mô. Tuy nhiên, ảnh MRI cung cấp độ tương phản mô mềm tốt hơn để xác định chính xác vị trí khối u. Dự án này nhằm kết hợp ưu điểm của cả hai phương thức chẩn đoán hình ảnh bằng cách tạo ra ảnh CT tổng hợp từ ảnh MRI, giúp:

- Giảm thiểu liều xạ chẩn đoán cho bệnh nhân (không cần chụp CT thêm)
- Cải thiện độ chính xác trong xác định thể tích mục tiêu và cơ quan nguy cấp
- Tối ưu hóa quy trình lập kế hoạch xạ trị

## Tính năng mới triển khai

1. **Pipeline thông minh tích hợp**
   - Triển khai quy trình xử lý tích hợp từ tiền xử lý đến đánh giá
   - Quản lý lỗi và khôi phục thông minh trong quy trình
   - Hỗ trợ theo dõi tiến trình và trực quan hóa phản hồi

2. **Phân đoạn não chuyên biệt**
   - Triển khai thuật toán phân đoạn não dựa trên GMM (Gaussian Mixture Model)
   - Phân biệt chính xác giữa chất xám, chất trắng và dịch não tuỷ
   - Hỗ trợ xử lý hậu kỳ chuyên biệt để cải thiện kết quả phân đoạn

3. **Đánh giá toàn diện**
   - Đánh giá kết quả CT tổng hợp với nhiều thông số (MAE, MSE, PSNR, SSIM)
   - So sánh kết quả theo vùng mô (xương, mô mềm, khoang khí)
   - Trực quan hóa kết quả so sánh giữa CT tổng hợp và CT tham chiếu

4. **Giao diện người dùng nâng cao**
   - Tích hợp quy trình xử lý đầy đủ trong giao diện đồ họa
   - Hiển thị đánh giá chi tiết và trực quan hóa kết quả
   - Quản lý tiến trình và phản hồi thời gian thực

## Quy trình phát triển và lộ trình thực hiện

Dự án được chia thành 5 giai đoạn chính với các mục tiêu và nhiệm vụ cụ thể cho mỗi giai đoạn.

### Giai đoạn 1: Thiết lập cơ sở hạ tầng và tiền xử lý ảnh MRI

**Thời gian**: Tuần 1-3
**Mục tiêu**: Xây dựng cơ sở hạ tầng dự án và hoàn thiện module tiền xử lý ảnh MRI

1. **Khởi tạo dự án**:
   - Thiết lập cấu trúc thư mục dự án
   - Cài đặt các phụ thuộc cơ bản
   - Thiết lập hệ thống quản lý cấu hình

2. **Tiền xử lý ảnh MRI**:
   - Phát triển các thuật toán chuẩn hóa cường độ tín hiệu
   - Triển khai công cụ loại bỏ nhiễu và cải thiện chất lượng ảnh
   - Xây dựng module đăng ký hình ảnh giữa các chuỗi MRI khác nhau
   - Kiểm thử và tối ưu hóa hiệu suất của bước tiền xử lý

3. **Công cụ và tiện ích**:
   - Phát triển các tiện ích xử lý dữ liệu DICOM
   - Tạo công cụ trực quan hóa để kiểm tra kết quả tiền xử lý
   - Xây dựng cấu trúc dữ liệu cho việc xử lý I/O ảnh y tế

### Giai đoạn 2: Phân đoạn mô và cấu trúc giải phẫu

**Thời gian**: Tuần 4-6
**Mục tiêu**: Xây dựng hệ thống phân đoạn mô tự động từ ảnh MRI

1. **Phân đoạn cấu trúc xương**:
   - Phát triển các thuật toán phân đoạn xương tự động
   - Ứng dụng các phương pháp học máy để cải thiện độ chính xác phân đoạn
   - Tối ưu hóa cho các vùng giải phẫu khác nhau (đầu, chậu, ngực)

2. **Phân đoạn mô mềm**:
   - Triển khai phân đoạn cho các loại mô mềm khác nhau
   - Cải thiện tương phản giữa các loại mô
   - Phân loại mô mềm theo mật độ để chuẩn bị cho việc chuyển đổi HU

3. **Phân đoạn khoang khí**:
   - Phát triển thuật toán nhận diện và phân đoạn khoang khí
   - Xử lý các trường hợp đặc biệt (xoang, đường thở...)
   - Khắc phục các thách thức trong phân đoạn khoang khí

4. **Kiểm thử và đánh giá**:
   - Xây dựng các metrics đánh giá độ chính xác phân đoạn
   - So sánh với phân đoạn thủ công bởi chuyên gia
   - Tạo báo cáo đánh giá tự động

### Giai đoạn 3: Phát triển mô hình chuyển đổi MRI-CT

**Thời gian**: Tuần 7-10
**Mục tiêu**: Triển khai và huấn luyện các mô hình chuyển đổi ảnh MRI thành CT tổng hợp

1. **Triển khai mô hình dựa trên Atlas**:
   - Xây dựng cơ sở dữ liệu Atlas cho các vùng giải phẫu
   - Phát triển thuật toán đăng ký và biến đổi dựa trên Atlas
   - Tối ưu hóa cho từng vùng giải phẫu cụ thể

2. **Triển khai mô hình CNN**:
   - Thiết kế kiến trúc mạng CNN cho chuyển đổi MRI-CT
   - Chuẩn bị dữ liệu huấn luyện và kiểm thử
   - Huấn luyện và tinh chỉnh mô hình CNN

3. **Triển khai mô hình GAN**:
   - Phát triển kiến trúc GAN cho việc tạo ảnh CT tổng hợp chất lượng cao
   - Huấn luyện mô hình với các tập dữ liệu cặp MRI-CT
   - Áp dụng các kỹ thuật cải tiến như cGAN, pix2pix để cải thiện chất lượng

4. **Tích hợp phương pháp kết hợp**:
   - Phát triển cơ chế kết hợp kết quả từ nhiều mô hình
   - Tối ưu hóa trọng số kết hợp dựa trên vùng giải phẫu
   - Đánh giá hiệu suất của phương pháp kết hợp so với từng mô hình riêng biệt

5. **Kiểm thử và so sánh các mô hình**:
   - Đánh giá định lượng (MAE, MSE, PSNR, SSIM)
   - Đánh giá định tính (khảo sát từ chuyên gia)
   - So sánh hiệu năng tính toán và yêu cầu tài nguyên

### Giai đoạn 4: Đánh giá và tối ưu hóa kết quả

**Thời gian**: Tuần 11-13
**Mục tiêu**: Đánh giá toàn diện và tối ưu hóa chất lượng CT tổng hợp

1. **Đánh giá tổng thể**:
   - So sánh CT tổng hợp với CT thực trên nhiều metrics
   - Phân tích sai số theo vùng giải phẫu và loại mô
   - Đánh giá tính nhất quán giữa các bệnh nhân

2. **Tối ưu hóa hậu xử lý**:
   - Phát triển các phương pháp hiệu chỉnh sai số tự động
   - Cải thiện chất lượng tái tạo cấu trúc xương phức tạp
   - Làm mịn và điều chỉnh ranh giới mô

3. **Đánh giá trên kế hoạch xạ trị**:
   - Tích hợp CT tổng hợp vào quy trình lập kế hoạch xạ trị
   - So sánh phân bố liều giữa kế hoạch dựa trên CT thực và CT tổng hợp
   - Đánh giá ảnh hưởng của sai số CT tổng hợp đến tính toán liều
   - Phân tích DVH (Dose Volume Histogram) từ cả hai kế hoạch

4. **Kiểm soát chất lượng (QA)**:
   - Phát triển quy trình kiểm tra chất lượng tự động
   - Tạo cơ chế cảnh báo khi có sai lệch lớn
   - Tối ưu hóa các thông số dựa trên kết quả đánh giá

### Giai đoạn 5: Triển khai giao diện người dùng và tài liệu

**Thời gian**: Tuần 14-16
**Mục tiêu**: Hoàn thiện giao diện người dùng và tài liệu cho việc triển khai

1. **Phát triển giao diện người dùng**:
   - Thiết kế và triển khai giao diện đồ họa thân thiện với người dùng
   - Tích hợp tất cả các module vào một hệ thống thống nhất
   - Cải thiện trải nghiệm người dùng dựa trên phản hồi

2. **Tích hợp quy trình đầy đủ**:
   - Kết nối liền mạch tất cả các bước từ tiền xử lý đến đánh giá
   - Tự động hóa quy trình nơi có thể
   - Đảm bảo tính nhất quán giữa các module

3. **Tài liệu và hướng dẫn**:
   - Tạo tài liệu kỹ thuật chi tiết cho nhà phát triển
   - Viết hướng dẫn sử dụng cho người dùng cuối
   - Tạo ví dụ mẫu và hướng dẫn từng bước

4. **Triển khai và kiểm thử cuối cùng**:
   - Triển khai hệ thống hoàn chỉnh trong môi trường thử nghiệm
   - Kiểm thử với người dùng thực tế và thu thập phản hồi
   - Sửa lỗi và cải tiến dựa trên phản hồi

## Tính năng cơ bản

1. **Tiền xử lý ảnh MRI**
   - Chuẩn hóa cường độ tín hiệu
   - Loại bỏ nhiễu và cải thiện chất lượng ảnh
   - Đăng ký hình ảnh (image registration) giữa các chuỗi MRI khác nhau

2. **Chuyển đổi MRI sang CT**
   - Mô hình học sâu (Deep Learning) cho việc chuyển đổi
   - Phương pháp dựa trên Atlas cho các cấu trúc giải phẫu phổ biến
   - Tạo bản đồ số HU (Hounsfield Unit) từ cường độ tín hiệu MRI

3. **Phân đoạn mô tự động**
   - Phân đoạn các cấu trúc xương
   - Phân đoạn mô mềm
   - Phân đoạn khoang khí

4. **Đánh giá kết quả**
   - So sánh MAE (Mean Absolute Error) giữa CT thực và CT tổng hợp
   - Phân tích sai số trên các loại mô khác nhau
   - Công cụ trực quan hóa sự khác biệt

5. **Quản lý dữ liệu bệnh nhân**
   - Nhập và xuất dữ liệu DICOM
   - Bảo mật thông tin bệnh nhân
   - Lưu trữ lịch sử xử lý

## Tính năng nâng cao

1. **Tích hợp đa chuỗi MRI**
   - Kết hợp dữ liệu từ nhiều chuỗi MRI (T1, T2, FLAIR, v.v.)
   - Tối ưu hóa mô hình cho từng vùng giải phẫu cụ thể
   - Tạo CT tổng hợp đa tham số

2. **Tùy chỉnh theo vùng giải phẫu**
   - Mô hình chuyên biệt cho các vùng đầu cổ
   - Mô hình chuyên biệt cho các vùng vùng chậu
   - Mô hình chuyên biệt cho các vùng ngực

3. **Tối ưu hóa lập kế hoạch xạ trị**
   - Tích hợp trực tiếp với hệ thống lập kế hoạch xạ trị (TPS)
   - So sánh phân bố liều giữa kế hoạch dựa trên CT thực và CT tổng hợp
   - Đánh giá ảnh hưởng của sai số CT tổng hợp đến tính toán liều

4. **Công cụ kiểm soát chất lượng (QA)**
   - Kiểm tra độ chính xác của chuyển đổi tự động
   - Cảnh báo khi có sai lệch lớn
   - Tạo báo cáo QA tự động

5. **Học máy nâng cao**
   - Cập nhật và cải tiến mô hình từ dữ liệu mới
   - Sử dụng kỹ thuật GAN (Generative Adversarial Networks) để tạo ảnh CT chất lượng cao
   - Hỗ trợ chuyển giao học tập (transfer learning) cho các bộ dữ liệu nhỏ

6. **Xử lý hậu kỳ thông minh**
   - Hiệu chỉnh sai số tự động
   - Tái tạo chi tiết cấu trúc xương phức tạp
   - Làm mịn và điều chỉnh ranh giới mô

## Cấu trúc dự án

```
synthetic-ct-from-mri/
├── app/                          # Ứng dụng chính
│   ├── main.py                   # Điểm khởi đầu của ứng dụng
│   ├── gui/                      # Giao diện người dùng
│   │   ├── __init__.py           # Định nghĩa module GUI
│   │   ├── main_window.py        # Cửa sổ chính của ứng dụng
│   │   └── run.py                # Script chạy giao diện người dùng
│   ├── core/                     # Các module cốt lõi
│   │   ├── preprocessing/        # Xử lý trước ảnh MRI
│   │   │   ├── __init__.py       # Định nghĩa module
│   │   │   └── preprocess_mri.py # Xử lý ảnh MRI
│   │   ├── segmentation/         # Phân đoạn các cấu trúc giải phẫu
│   │   │   ├── __init__.py       # Định nghĩa module
│   │   │   └── segment_tissues.py # Phân đoạn mô
│   │   ├── conversion/           # Chuyển đổi MRI sang CT
│   │   │   ├── __init__.py       # Định nghĩa module
│   │   │   └── convert_mri_to_ct.py # Chuyển đổi MRI sang CT
│   │   └── evaluation/           # Đánh giá kết quả chuyển đổi
│   │       ├── __init__.py       # Định nghĩa module
│   │       └── evaluate_synthetic_ct.py # Đánh giá kết quả
│   └── utils/                    # Tiện ích chung
│       ├── __init__.py           # Định nghĩa module
│       ├── dicom_utils.py        # Xử lý dữ liệu DICOM
│       ├── visualization.py      # Công cụ trực quan hóa
│       ├── io_utils.py           # Xử lý đầu vào/đầu ra
│       └── config_utils.py       # Quản lý cấu hình
├── data/                         # Thư mục dữ liệu
│   ├── raw/                      # Dữ liệu gốc (MRI và CT tham chiếu)
│   ├── processed/                # Dữ liệu đã xử lý
│   └── results/                  # Kết quả CT tổng hợp
├── models/                       # Lưu trữ mô hình
│   ├── cnn/                      # Mô hình CNN
│   ├── gan/                      # Mô hình GAN
│   └── atlas/                    # Mô hình dựa trên Atlas
├── configs/                      # Tệp cấu hình
│   ├── default_config.yaml       # Cấu hình mặc định
│   ├── model_configs/            # Cấu hình mô hình
│   ├── preprocessing_configs/    # Cấu hình tiền xử lý
│   └── system_configs/           # Cấu hình hệ thống
├── scripts/                      # Tập lệnh hỗ trợ
│   ├── train_gan_model.py        # Huấn luyện mô hình GAN
│   ├── evaluate_model.py         # Đánh giá mô hình
│   └── deployment/               # Triển khai
├── docs/                         # Tài liệu
│   ├── user_manual/              # Hướng dẫn sử dụng
│   ├── developer_docs/           # Tài liệu phát triển
│   └── research_papers/          # Tài liệu nghiên cứu liên quan
├── tests/                        # Bộ kiểm thử
│   ├── unit/                     # Kiểm thử đơn vị
│   └── integration/              # Kiểm thử tích hợp
├── notebooks/                    # Jupyter notebooks cho phân tích
├── requirements.txt              # Các phụ thuộc Python
├── setup.py                      # Cài đặt gói
├── LICENSE                       # Giấy phép
└── README.md                     # Tài liệu giới thiệu
```

## Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python 3.8+
- **Thư viện xử lý ảnh y tế**: SimpleITK, NiBabel
- **Học máy và học sâu**: TensorFlow, scikit-learn
- **Xử lý dữ liệu DICOM**: pydicom
- **Trực quan hóa**: Matplotlib, Plotly
- **Giao diện người dùng**: PySide6
- **Quản lý cấu hình**: PyYAML
- **Xử lý dữ liệu**: NumPy, Pandas
- **Kiểm thử**: pytest

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- CUDA 11.0+ (cho huấn luyện GPU, tùy chọn)
- RAM: ít nhất 8GB (khuyến nghị 16GB cho bộ dữ liệu lớn)
- Ổ cứng: ít nhất 10GB không gian trống
- Hệ điều hành: Windows 10+, macOS 10.14+, hoặc Linux

## Cài đặt và sử dụng

### Cài đặt từ source code

```bash
# Sao chép dự án
git clone https://github.com/yourusername/synthetic-ct-from-mri.git
cd synthetic-ct-from-mri

# Tạo môi trường ảo (tùy chọn nhưng khuyến nghị)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate

# Cài đặt các phụ thuộc
pip install -r requirements.txt

# Cài đặt gói phát triển (tùy chọn)
pip install -e .
```

### Sử dụng giao diện đồ họa (GUI)

```bash
# Chạy ứng dụng với giao diện đồ họa
python app/gui/run.py
```

### Sử dụng dòng lệnh

```bash
# Xử lý một file MRI và tạo synthetic CT
python app/main.py --input path/to/mri.nii.gz --output path/to/output --model gan --region head

# Để xem tất cả các tùy chọn
python app/main.py --help
```

## Ví dụ quy trình hoàn chỉnh

Dưới đây là quy trình đầy đủ từ dữ liệu thô đến ảnh CT tổng hợp và đánh giá:

```bash
# Sử dụng quy trình tích hợp (phương pháp mới)
python app/main.py --mode pipeline --input data/raw/patient001.nii.gz --output results/patient001 --model gan --region head --reference data/raw/patient001_ct.nii.gz
```

Hoặc quy trình từng bước:

```bash
# 1. Tiền xử lý ảnh MRI
python app/main.py --mode preprocess --input data/raw/patient001.nii.gz --output data/processed/patient001_preprocessed.nii.gz

# 2. Phân đoạn mô
python app/main.py --mode segment --input data/processed/patient001_preprocessed.nii.gz --output data/processed/patient001_segmentation.nii.gz --region head

# 3. Chuyển đổi MRI sang CT
python app/main.py --mode convert --input data/processed/patient001_preprocessed.nii.gz --segmentation data/processed/patient001_segmentation.nii.gz --output results/patient001_synthetic_ct.nii.gz --model gan --region head

# 4. Đánh giá kết quả (nếu có CT tham chiếu)
python scripts/evaluate_model.py --mri_path data/processed/patient001_preprocessed.nii.gz --ct_path data/raw/patient001_ct.nii.gz --output_dir results/evaluation/patient001 --model_type gan --region head --mode single
```

Hoặc sử dụng giao diện đồ họa để thực hiện các bước trên một cách trực quan và dễ dàng hơn:

```bash
python app/gui/run.py
```

## Khắc phục sự cố

### Xử lý lỗi thường gặp

- **ImportError**: Kiểm tra xem bạn đã cài đặt đầy đủ các phụ thuộc trong `requirements.txt` chưa.
- **CUDA errors**: Đảm bảo phiên bản CUDA tương thích với phiên bản TensorFlow.
- **Lỗi bộ nhớ**: Giảm kích thước batch hoặc kích thước patch khi xử lý ảnh lớn.
- **Lỗi đọc/ghi file**: Kiểm tra quyền truy cập vào thư mục và định dạng file đầu vào.

### Báo cáo lỗi

Nếu bạn gặp phải lỗi không được liệt kê ở trên, vui lòng tạo issue mới trên GitHub với các thông tin sau:
- Mô tả lỗi
- Bước để tái tạo lỗi
- Thông tin hệ thống (OS, phiên bản Python, GPU, v.v.)
- Logs lỗi (nếu có)

## Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng xem file CONTRIBUTING.md để biết hướng dẫn chi tiết.

1. Fork dự án
2. Tạo nhánh tính năng (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi của bạn (`git commit -m 'Add some AmazingFeature'`)
4. Push lên nhánh (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## Giấy phép

Dự án này được phân phối theo Giấy phép MIT. Xem file LICENSE để biết thêm thông tin.

## Trích dẫn

Nếu bạn sử dụng mã nguồn hoặc phương pháp từ dự án này trong nghiên cứu của mình, vui lòng trích dẫn:

```
@software{SyntheticCTfromMRI,
  author = {Your Name},
  title = {Synthetic-CT-from-MRI-for-radiotherapy},
  url = {https://github.com/yourusername/synthetic-ct-from-mri},
  year = {2023},
}
```

## Liên hệ

Nếu bạn có bất kỳ câu hỏi hoặc gợi ý nào, vui lòng tạo một issue hoặc liên hệ trực tiếp qua email.

## Hướng dẫn thiết lập và kiểm thử

### 1. Cài đặt

```bash
# Clone repository
git clone https://github.com/your-username/Synthetic-CT-from-MRI-for-radiotherapy.git
cd Synthetic-CT-from-MRI-for-radiotherapy

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Tạo mô hình placeholder cho kiểm thử

Dự án sử dụng các mô hình trong thư mục `/models` cho việc chuyển đổi MRI sang CT. Để kiểm thử chức năng với mô hình placeholder, chạy:

```bash
python scripts/create_placeholder_models.py
```

### 3. Kiểm thử chức năng

Để kiểm thử các phương pháp chuyển đổi MRI thành CT:

```bash
python scripts/verify_models.py
```

Script này tạo ảnh MRI và phân đoạn mô giả, sau đó thực hiện chuyển đổi với các phương pháp khác nhau (atlas, cnn, gan) cho ba vùng giải phẫu (head, pelvis, thorax).

Kết quả kiểm thử sẽ được lưu trong thư mục `/test_results`.

### 4. Chạy ứng dụng

```bash
# Chạy giao diện người dùng đồ họa
python app/main.py --mode gui

# Chạy quy trình xử lý đầy đủ từ dòng lệnh
python app/main.py --mode pipeline --input-path <path-to-mri> --output-dir <output-directory>
```

## English Version

Project for converting MRI images to synthetic CT (sCT) for radiotherapy treatment planning, aiming to reduce unwanted radiation dose to patients.

## Setup and Testing Instructions

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-username/Synthetic-CT-from-MRI-for-radiotherapy.git
cd Synthetic-CT-from-MRI-for-radiotherapy

# Install dependencies
pip install -r requirements.txt
```

### 2. Create placeholder models for testing

The project uses models in the `/models` directory for MRI to CT conversion. To test functionality with placeholder models, run:

```bash
python scripts/create_placeholder_models.py
```

### 3. Test functionality

To test the different MRI to CT conversion methods:

```bash
python scripts/verify_models.py
```

This script creates dummy MRI and tissue segmentation, then performs conversion using different methods (atlas, cnn, gan) for three anatomical regions (head, pelvis, thorax).

Test results will be saved in the `/test_results` directory.

### 4. Run the application

```bash
# Run the graphical user interface
python app/main.py --mode gui

# Run the complete processing pipeline from command line
python app/main.py --mode pipeline --input-path <path-to-mri> --output-dir <output-directory>
```

## Project Structure

```
Synthetic-CT-from-MRI-for-radiotherapy/
├── app/                    # Main application code
│   ├── core/               # Core processing modules
│   │   ├── preprocessing/  # MRI preprocessing
│   │   ├── segmentation/   # Tissue segmentation
│   │   ├── conversion/     # MRI to CT conversion
│   │   └── evaluation/     # Results evaluation
│   ├── gui/                # Graphical user interface
│   ├── utils/              # Utility functions
│   └── main.py             # Application entry point
├── configs/                # Configuration files
├── data/                   # Sample data directory
├── models/                 # Pre-trained models
│   ├── atlas/              # Atlas-based models
│   ├── cnn/                # CNN-based models
│   ├── gan/                # GAN-based models
│   └── segmentation/       # Segmentation models
├── scripts/                # Utility scripts
├── tests/                  # Unit and integration tests
└── requirements.txt        # Python dependencies
```

## Features Overview

### Basic Features

1. **MRI Preprocessing**
   - Intensity normalization
   - Denoising and image quality enhancement
   - Registration between different MRI sequences

2. **MRI to CT Conversion**
   - Deep learning models for conversion
   - Atlas-based methods for common anatomical structures
   - Hounsfield Unit (HU) mapping from MRI signal intensity

3. **Automatic Tissue Segmentation**
   - Bone structure segmentation
   - Soft tissue segmentation
   - Air cavity segmentation

4. **Results Evaluation**
   - MAE (Mean Absolute Error) comparison between real and synthetic CT
   - Error analysis on different tissue types
   - Visualization tools for differences

### Advanced Features

1. **Multi-sequence MRI Integration**
   - Combining data from multiple MRI sequences (T1, T2, FLAIR, etc.)
   - Model optimization for specific anatomical regions
   - Multi-parameter synthetic CT generation

2. **Deep Learning Model Implementation**
   - CNN-based models
   - GAN-based models
   - Encoder-decoder architectures

3. **Dose Calculation Integration**
   - DVH (Dose-Volume Histogram) analysis
   - Dose distribution comparison
   - Treatment plan quality metrics

4. **Enhanced User Interface**
   - Integrated full processing pipeline
   - Detailed evaluation display and visualization
   - Real-time progress monitoring

If you have any questions or suggestions, please create an issue or contact us directly via email.