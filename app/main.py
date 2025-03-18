#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Synthetic CT from MRI for Radiotherapy
Main application entry point
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import traceback

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.io_utils import setup_logging, validate_input_file, ensure_output_dir
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues
from app.core.conversion import convert_mri_to_ct
from app.core.evaluation import evaluate_synthetic_ct
from app.utils.config_utils import load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chuyển đổi ảnh MRI thành ảnh CT tổng hợp cho lập kế hoạch xạ trị"
    )
    
    # Thêm subparsers cho từng chế độ
    subparsers = parser.add_subparsers(dest='mode', help='Chế độ hoạt động')
    
    # Parser chung cho input, output, verbose
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True,
        help="Đường dẫn đến file MRI đầu vào (định dạng DICOM hoặc NIfTI)"
    )
    parent_parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="./data/results",
        help="Đường dẫn đến thư mục/file lưu kết quả"
    )
    parent_parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Hiển thị thông tin chi tiết trong quá trình xử lý"
    )
    parent_parser.add_argument(
        "--config", "-c", 
        type=str, 
        help="Đường dẫn đến file cấu hình tùy chỉnh"
    )
    
    # Chế độ tiền xử lý
    preprocess_parser = subparsers.add_parser('preprocess', parents=[parent_parser], 
                                             help='Tiền xử lý ảnh MRI')
    preprocess_parser.add_argument(
        "--bias-correction", 
        action="store_true",
        help="Áp dụng hiệu chỉnh trường bias N4"
    )
    preprocess_parser.add_argument(
        "--denoise", 
        action="store_true",
        help="Áp dụng bộ lọc giảm nhiễu"
    )
    preprocess_parser.add_argument(
        "--normalize", 
        action="store_true",
        help="Áp dụng chuẩn hóa cường độ"
    )
    
    # Chế độ phân đoạn
    segment_parser = subparsers.add_parser('segment', parents=[parent_parser], 
                                          help='Phân đoạn mô từ ảnh MRI')
    segment_parser.add_argument(
        "--region", "-r", 
        type=str, 
        default="head",
        choices=["head", "pelvis", "thorax"],
        help="Vùng giải phẫu cần phân đoạn"
    )
    segment_parser.add_argument(
        "--method", "-m", 
        type=str, 
        default="deep_learning",
        choices=["deep_learning", "atlas"],
        help="Phương pháp phân đoạn"
    )
    
    # Chế độ chuyển đổi
    convert_parser = subparsers.add_parser('convert', parents=[parent_parser], 
                                          help='Chuyển đổi MRI sang CT tổng hợp')
    convert_parser.add_argument(
        "--model", "-m", 
        type=str, 
        default="gan",
        choices=["cnn", "gan", "atlas"],
        help="Loại mô hình sử dụng cho chuyển đổi"
    )
    convert_parser.add_argument(
        "--region", "-r", 
        type=str, 
        default="head",
        choices=["head", "pelvis", "thorax"],
        help="Vùng giải phẫu cần chuyển đổi"
    )
    convert_parser.add_argument(
        "--segmentation", "-s", 
        type=str, 
        help="Đường dẫn đến file phân đoạn mô (nếu có)"
    )
    
    # Chế độ đánh giá
    evaluate_parser = subparsers.add_parser('evaluate', parents=[parent_parser], 
                                           help='Đánh giá kết quả CT tổng hợp')
    evaluate_parser.add_argument(
        "--reference", "-r", 
        type=str, 
        required=True,
        help="Đường dẫn đến CT thực để so sánh"
    )
    evaluate_parser.add_argument(
        "--metrics", 
        type=str, 
        default="mae,mse,psnr,ssim",
        help="Danh sách các metrics sử dụng để đánh giá (phân cách bởi dấu phẩy)"
    )
    
    # Chế độ end-to-end xử lý toàn bộ quy trình
    pipeline_parser = subparsers.add_parser('pipeline', parents=[parent_parser], 
                                           help='Chạy toàn bộ quy trình từ MRI đến CT tổng hợp')
    pipeline_parser.add_argument(
        "--model", "-m", 
        type=str, 
        default="gan",
        choices=["cnn", "gan", "atlas"],
        help="Loại mô hình sử dụng cho chuyển đổi"
    )
    pipeline_parser.add_argument(
        "--region", "-r", 
        type=str, 
        default="head",
        choices=["head", "pelvis", "thorax"],
        help="Vùng giải phẫu cần chuyển đổi"
    )
    pipeline_parser.add_argument(
        "--reference", 
        type=str, 
        help="Đường dẫn đến CT thực để so sánh (nếu có)"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra nếu không có chế độ nào được chọn
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
        
    return args


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config) if hasattr(args, 'config') and args.config else load_config()
    
    logging.info(f"Chế độ: {args.mode}")
    logging.info(f"File đầu vào: {args.input}")
    logging.info(f"Đường dẫn đầu ra: {args.output}")
    
    try:
        # Kiểm tra file đầu vào
        if not validate_input_file(args.input):
            logging.error(f"File đầu vào không hợp lệ hoặc không tồn tại: {args.input}")
            return 1
            
        # Đảm bảo thư mục đầu ra tồn tại
        ensure_output_dir(args.output)
        
        # Xử lý theo chế độ được chọn
        if args.mode == 'preprocess':
            return handle_preprocess(args, config)
        elif args.mode == 'segment':
            return handle_segment(args, config)
        elif args.mode == 'convert':
            return handle_convert(args, config)
        elif args.mode == 'evaluate':
            return handle_evaluate(args, config)
        elif args.mode == 'pipeline':
            return handle_pipeline(args, config)
        else:
            logging.error(f"Chế độ không được hỗ trợ: {args.mode}")
            return 1
            
    except Exception as e:
        logging.error(f"Lỗi trong quá trình thực hiện: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1


def handle_preprocess(args, config):
    """Xử lý chế độ tiền xử lý."""
    logging.info("Bắt đầu tiền xử lý ảnh MRI")
    
    # Áp dụng tiền xử lý với các tùy chọn từ tham số dòng lệnh
    preprocessed_mri = preprocess_mri(
        args.input,
        apply_bias_field_correction=args.bias_correction,
        apply_denoising=args.denoise,
        apply_normalization=args.normalize,
        config=config
    )
    
    # Lưu kết quả
    preprocessed_mri.save(args.output)
    logging.info(f"Đã lưu ảnh MRI đã tiền xử lý tại: {args.output}")
    return 0


def handle_segment(args, config):
    """Xử lý chế độ phân đoạn."""
    logging.info(f"Bắt đầu phân đoạn mô từ ảnh MRI sử dụng phương pháp {args.method}")
    
    # Phân đoạn mô
    segmentation = segment_tissues(
        args.input,
        method=args.method,
        region=args.region,
        config=config
    )
    
    # Lưu kết quả
    segmentation.save(args.output)
    logging.info(f"Đã lưu kết quả phân đoạn tại: {args.output}")
    return 0


def handle_convert(args, config):
    """Xử lý chế độ chuyển đổi."""
    logging.info(f"Bắt đầu chuyển đổi MRI sang CT tổng hợp sử dụng mô hình {args.model}")
    
    # Chuyển đổi MRI sang CT
    synthetic_ct = convert_mri_to_ct(
        args.input,
        segmentation_path=args.segmentation if hasattr(args, 'segmentation') else None,
        model_type=args.model,
        region=args.region,
        config=config
    )
    
    # Lưu kết quả
    synthetic_ct.save(args.output)
    logging.info(f"Đã lưu CT tổng hợp tại: {args.output}")
    return 0


def handle_evaluate(args, config):
    """Xử lý chế độ đánh giá."""
    logging.info("Bắt đầu đánh giá kết quả CT tổng hợp")
    
    # Chuyển đổi chuỗi metrics thành list
    metrics_list = args.metrics.split(',')
    
    # Đánh giá kết quả
    evaluation_results = evaluate_synthetic_ct(
        args.input,  # CT tổng hợp
        args.reference,  # CT tham chiếu
        metrics=metrics_list,
        config=config
    )
    
    # Lưu và hiển thị kết quả
    output_report = os.path.join(args.output, "evaluation_report.json")
    evaluation_results.save_report(output_report)
    
    logging.info(f"Kết quả đánh giá:")
    for metric, value in evaluation_results.metrics.items():
        logging.info(f"  {metric}: {value}")
    
    logging.info(f"Đã lưu báo cáo đánh giá chi tiết tại: {output_report}")
    return 0


def handle_pipeline(args, config):
    """Xử lý toàn bộ quy trình từ MRI đến CT tổng hợp và đánh giá."""
    logging.info("Bắt đầu quy trình xử lý toàn bộ từ MRI đến CT tổng hợp")
    
    # Tạo các thư mục con trong thư mục đầu ra
    output_base = Path(args.output)
    output_base.mkdir(exist_ok=True, parents=True)
    
    preprocessed_dir = output_base / "preprocessed"
    segmentation_dir = output_base / "segmentation"
    synthetic_ct_dir = output_base / "synthetic_ct"
    evaluation_dir = output_base / "evaluation"
    
    for dir_path in [preprocessed_dir, segmentation_dir, synthetic_ct_dir, evaluation_dir]:
        dir_path.mkdir(exist_ok=True)
    
    try:
        # Step 1: Preprocess MRI
        logging.info("Bước 1: Tiền xử lý ảnh MRI")
        preprocessed_path = preprocessed_dir / f"preprocessed_mri.nii.gz"
        preprocessed_mri = preprocess_mri(
            args.input,
            apply_bias_field_correction=True,
            apply_denoising=True,
            apply_normalization=True,
            config=config
        )
        preprocessed_mri.save(str(preprocessed_path))
        logging.info(f"Đã lưu MRI đã tiền xử lý tại: {preprocessed_path}")
        
        # Step 2: Segment tissues
        logging.info(f"Bước 2: Phân đoạn mô cho vùng {args.region}")
        segmentation_path = segmentation_dir / f"segmentation_{args.region}.nii.gz"
        segmentation = segment_tissues(
            str(preprocessed_path),
            method="deep_learning",
            region=args.region,
            config=config
        )
        segmentation.save(str(segmentation_path))
        logging.info(f"Đã lưu kết quả phân đoạn tại: {segmentation_path}")
        
        # Step 3: Convert MRI to CT
        logging.info(f"Bước 3: Chuyển đổi MRI sang CT sử dụng mô hình {args.model}")
        synthetic_ct_path = synthetic_ct_dir / f"synthetic_ct_{args.model}_{args.region}.nii.gz"
        synthetic_ct = convert_mri_to_ct(
            str(preprocessed_path),
            segmentation_path=str(segmentation_path),
            model_type=args.model,
            region=args.region,
            config=config
        )
        synthetic_ct.save(str(synthetic_ct_path))
        logging.info(f"Đã lưu CT tổng hợp tại: {synthetic_ct_path}")
        
        # Step 4: Evaluate if reference CT is provided
        if hasattr(args, 'reference') and args.reference:
            logging.info("Bước 4: Đánh giá kết quả so với CT tham chiếu")
            evaluation_results = evaluate_synthetic_ct(
                str(synthetic_ct_path),
                args.reference,
                metrics=["mae", "mse", "psnr", "ssim"],
                config=config
            )
            
            # Lưu và hiển thị kết quả
            output_report = evaluation_dir / "evaluation_report.json"
            evaluation_results.save_report(str(output_report))
            
            logging.info(f"Kết quả đánh giá:")
            for metric, value in evaluation_results.metrics.items():
                logging.info(f"  {metric}: {value}")
                
            logging.info(f"Đã lưu báo cáo đánh giá chi tiết tại: {output_report}")
        
        logging.info("Hoàn thành quy trình xử lý toàn bộ từ MRI đến CT tổng hợp")
        return 0
        
    except Exception as e:
        logging.error(f"Lỗi trong quy trình xử lý: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 