# Synthetic-CT-from-MRI-for-radiotherapy

Dự án chuyển đổi ảnh MRI thành ảnh CT tổng hợp (synthetic CT hoặc sCT) cho lập kế hoạch xạ trị, nhằm giảm thiểu liều xạ không mong muốn vào cơ thể bệnh nhân.

[English Version](#english-version)

## Tổng quan

Trong lập kế hoạch xạ trị, ảnh CT là cần thiết cho tính toán liều xạ do cung cấp thông tin về mật độ electron của mô. Tuy nhiên, ảnh MRI cung cấp độ tương phản mô mềm tốt hơn để xác định chính xác vị trí khối u. Dự án này nhằm kết hợp ưu điểm của cả hai phương thức chẩn đoán hình ảnh bằng cách tạo ra ảnh CT tổng hợp từ ảnh MRI, giúp:

- Giảm thiểu liều xạ chẩn đoán cho bệnh nhân (không cần chụp CT thêm)
- Cải thiện độ chính xác trong xác định thể tích mục tiêu và cơ quan nguy cấp
- Tối ưu hóa quy trình lập kế hoạch xạ trị

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

## Hướng dẫn chi tiết

### Tiền xử lý ảnh MRI

```bash
# Tiền xử lý ảnh MRI riêng biệt
python app/main.py --mode preprocess --input path/to/mri.nii.gz --output path/to/preprocessed
```

### Phân đoạn mô

```bash
# Phân đoạn mô từ ảnh MRI đã tiền xử lý
python app/main.py --mode segment --input path/to/preprocessed/mri.nii.gz --output path/to/segmentation --region head
```

### Chuyển đổi MRI sang CT

```bash
# Chuyển đổi từ MRI đã tiền xử lý sang synthetic CT
python app/main.py --mode convert --input path/to/preprocessed/mri.nii.gz --segmentation path/to/segmentation.nii.gz --output path/to/synthetic_ct.nii.gz --model gan --region head
```

### Đánh giá kết quả

```bash
# Đánh giá synthetic CT với CT tham chiếu
python scripts/evaluate_model.py --mri_path path/to/mri.nii.gz --ct_path path/to/reference_ct.nii.gz --output_dir results/evaluation --model_type gan --region head --mode single

# Đánh giá hàng loạt
python scripts/evaluate_model.py --mri_dir data/test/mri --ct_dir data/test/ct --output_dir results/batch_evaluation --model_type gan --region head --mode batch

# So sánh nhiều mô hình
python scripts/evaluate_model.py --mri_path path/to/mri.nii.gz --ct_path path/to/reference_ct.nii.gz --output_dir results/comparison --region head --mode compare
```

### Huấn luyện mô hình

```bash
# Tạo dataset cho huấn luyện
python scripts/train_gan_model.py --mri_dirs data/raw/mri --ct_dirs data/raw/ct --dataset_dir data/processed/gan_dataset --create_dataset --region head

# Huấn luyện mô hình GAN
python scripts/train_gan_model.py --dataset_dir data/processed/gan_dataset --output_dir models/gan/head --epochs 200 --batch_size 4 --lambda_l1 100 --region head
```

## Ví dụ quy trình hoàn chỉnh

Dưới đây là quy trình đầy đủ từ dữ liệu thô đến ảnh CT tổng hợp và đánh giá:

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

---

# English Version <a name="english-version"></a>

# Synthetic-CT-from-MRI-for-radiotherapy

A project to convert MRI images to synthetic CT images (sCT) for radiotherapy planning, aiming to reduce unwanted radiation dose to patients.

## Overview

In radiotherapy planning, CT images are essential for dose calculation as they provide information about the electron density of tissues. However, MRI provides better soft tissue contrast for accurate tumor localization. This project aims to combine the advantages of both imaging modalities by generating synthetic CT images from MRI, helping to:

- Reduce diagnostic radiation dose to patients (avoiding additional CT scans)
- Improve accuracy in target volume and organs-at-risk delineation
- Optimize the radiotherapy planning workflow

## Basic Features

1. **MRI Preprocessing**
   - Signal intensity normalization
   - Noise reduction and image quality enhancement
   - Image registration between different MRI sequences

2. **MRI to CT Conversion**
   - Deep Learning models for conversion
   - Atlas-based methods for common anatomical structures
   - Generation of HU (Hounsfield Unit) maps from MRI signal intensities

3. **Automatic Tissue Segmentation**
   - Bone structure segmentation
   - Soft tissue segmentation
   - Air cavity segmentation

4. **Result Evaluation**
   - MAE (Mean Absolute Error) comparison between real and synthetic CT
   - Error analysis on different tissue types
   - Difference visualization tools

5. **Patient Data Management**
   - DICOM import and export
   - Patient information security
   - Processing history storage

## Advanced Features

1. **Multi-sequence MRI Integration**
   - Combination of data from multiple MRI sequences (T1, T2, FLAIR, etc.)
   - Model optimization for specific anatomical regions
   - Multi-parameter synthetic CT generation

2. **Anatomical Region Customization**
   - Specialized models for head and neck regions
   - Specialized models for pelvic regions
   - Specialized models for thoracic regions

3. **Radiotherapy Planning Optimization**
   - Direct integration with treatment planning systems (TPS)
   - Dose distribution comparison between real CT and synthetic CT-based plans
   - Assessment of synthetic CT error impacts on dose calculation

4. **Quality Assurance (QA) Tools**
   - Automatic conversion accuracy verification
   - Warning alerts for large discrepancies
   - Automatic QA report generation

5. **Advanced Machine Learning**
   - Model updates and improvements from new data
   - GAN (Generative Adversarial Networks) techniques for high-quality CT image generation
   - Transfer learning support for small datasets

6. **Intelligent Post-processing**
   - Automatic error correction
   - Complex bone structure detail reconstruction
   - Tissue boundary smoothing and adjustment

## Project Structure

```
synthetic-ct-from-mri/
├── app/                          # Main application
│   ├── main.py                   # Application entry point
│   ├── gui/                      # Graphical user interface
│   │   ├── __init__.py           # GUI module definition
│   │   ├── main_window.py        # Main application window
│   │   └── run.py                # GUI runner script
│   ├── core/                     # Core modules
│   │   ├── preprocessing/        # MRI preprocessing
│   │   │   ├── __init__.py       # Module definition
│   │   │   └── preprocess_mri.py # MRI processing
│   │   ├── segmentation/         # Anatomical structure segmentation
│   │   │   ├── __init__.py       # Module definition
│   │   │   └── segment_tissues.py # Tissue segmentation
│   │   ├── conversion/           # MRI to CT conversion
│   │   │   ├── __init__.py       # Module definition
│   │   │   └── convert_mri_to_ct.py # MRI to CT conversion
│   │   └── evaluation/           # Result evaluation
│   │       ├── __init__.py       # Module definition
│   │       └── evaluate_synthetic_ct.py # Synthetic CT evaluation
│   ├── training/                 # Model training module
│   │   ├── __init__.py           # Module definition
│   │   ├── train_cnn.py          # CNN model training
│   │   └── train_gan.py          # GAN model training
│   └── utils/                    # Các tiện ích
│       ├── __init__.py           # Module definition
│       ├── io_utils.py           # Input/output utilities
│       ├── dicom_utils.py        # DICOM handling utilities
│       ├── visualization.py      # Visualization utilities
│       └── config_utils.py       # Configuration utilities
├── configs/                      # Configuration
│   └── default_config.yaml       # Default configuration
├── data/                         # Dữ liệu
│   ├── input/                    # Input data
│   │   ├── mri/                  # MRI data
│   │   └── ct/                   # Reference CT data
│   └── output/                   # Dữ liệu đầu ra
│       ├── preprocessed/         # MRI đã tiền xử lý
│       ├── segmented/            # Kết quả phân đoạn
│       ├── synthetic_ct/         # CT tổng hợp
│       └── evaluation/           # Kết quả đánh giá
├── models/                       # Lưu trữ mô hình
│   ├── cnn/                      # CNN models
│   ├── gan/                      # GAN models
│   └── atlas/                    # Dữ liệu Atlas
├── scripts/                      # Scripts hỗ trợ
│   ├── prepare_dataset.py        # Chuẩn bị dữ liệu
│   ├── train_gan_model.py        # Script huấn luyện GAN
│   └── evaluate_model.py         # Script đánh giá mô hình
├── tests/                        # Kiểm thử
│   ├── __init__.py               # Định nghĩa module tests
│   ├── test_preprocessing.py     # Kiểm thử tiền xử lý
│   ├── test_segmentation.py      # Kiểm thử phân đoạn
│   ├── test_conversion.py        # Kiểm thử chuyển đổi
│   └── test_io_utils.py          # IO utilities tests
├── setup.py                      # Cài đặt gói
└── requirements.txt              # Các phụ thuộc
```

## Installation

```bash
# Clone repository
git clone https://github.com/username/synthetic-ct-from-mri.git
cd synthetic-ct-from-mri

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Usage

### Command-line Interface

```bash
# Tiền xử lý ảnh MRI
synthetic_ct preprocess input_mri.nii.gz preprocessed_mri.nii.gz --bias_correction --denoise --normalize

# Phân đoạn mô
synthetic_ct segment preprocessed_mri.nii.gz segmentation.nii.gz --method deeplearning --region head

# Chuyển đổi MRI sang CT
synthetic_ct convert preprocessed_mri.nii.gz synthetic_ct.nii.gz --method cnn --segmentation segmentation.nii.gz

# Đánh giá kết quả
synthetic_ct evaluate synthetic_ct.nii.gz reference_ct.nii.gz --metrics mae,mse,psnr,ssim --report evaluation_report.pdf

# Hiển thị kết quả
synthetic_ct visualize synthetic_ct.nii.gz --compare reference_ct.nii.gz --output comparison.png
```

### Graphical User Interface (GUI)

```bash
# Launch the GUI
synthetic_ct_gui
```