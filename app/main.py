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

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.io_utils import setup_logging
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues
from app.core.conversion import convert_mri_to_ct
from app.core.evaluation import evaluate_synthetic_ct


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chuyển đổi ảnh MRI thành ảnh CT tổng hợp cho lập kế hoạch xạ trị"
    )
    
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True,
        help="Đường dẫn đến file MRI đầu vào (định dạng DICOM hoặc NIfTI)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="./data/results",
        help="Đường dẫn đến thư mục lưu kết quả CT tổng hợp"
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default="gan",
        choices=["cnn", "gan", "atlas"],
        help="Loại mô hình sử dụng cho chuyển đổi"
    )
    
    parser.add_argument(
        "--region", "-r", 
        type=str, 
        default="head",
        choices=["head", "pelvis", "thorax"],
        help="Vùng giải phẫu cần chuyển đổi"
    )
    
    parser.add_argument(
        "--evaluate", "-e", 
        action="store_true",
        help="Đánh giá kết quả nếu có CT thực để so sánh"
    )
    
    parser.add_argument(
        "--reference", 
        type=str, 
        help="Đường dẫn đến CT thực để so sánh (nếu có)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Hiển thị thông tin chi tiết trong quá trình xử lý"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logging.info("Bắt đầu chuyển đổi MRI sang CT tổng hợp")
    logging.info(f"File đầu vào: {args.input}")
    logging.info(f"Thư mục đầu ra: {args.output}")
    logging.info(f"Mô hình sử dụng: {args.model}")
    logging.info(f"Vùng giải phẫu: {args.region}")
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Step 1: Preprocess MRI
        logging.info("Bước 1: Tiền xử lý ảnh MRI")
        preprocessed_mri = preprocess_mri(args.input)
        
        # Step 2: Segment tissues
        logging.info("Bước 2: Phân đoạn mô")
        segmented_tissues = segment_tissues(preprocessed_mri, region=args.region)
        
        # Step 3: Convert MRI to CT
        logging.info("Bước 3: Chuyển đổi MRI sang CT")
        synthetic_ct = convert_mri_to_ct(
            preprocessed_mri, 
            segmented_tissues, 
            model_type=args.model,
            region=args.region
        )
        
        # Step 4: Save results
        output_path = os.path.join(args.output, f"synthetic_ct_{args.model}_{args.region}.nii.gz")
        synthetic_ct.save(output_path)
        logging.info(f"Đã lưu CT tổng hợp tại: {output_path}")
        
        # Step 5: Evaluate if requested
        if args.evaluate and args.reference:
            logging.info("Bước 5: Đánh giá kết quả")
            metrics = evaluate_synthetic_ct(synthetic_ct, args.reference)
            logging.info(f"Kết quả đánh giá: {metrics}")
        
        logging.info("Hoàn thành chuyển đổi MRI sang CT tổng hợp")
        return 0
        
    except Exception as e:
        logging.error(f"Lỗi trong quá trình chuyển đổi: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 