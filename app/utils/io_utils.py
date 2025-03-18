#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Utility functions for input/output operations
"""

import os
import logging
import datetime
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(
                    "logs", 
                    f"synthetic_ct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            )
        ]
    )
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)


def load_dicom_series(directory):
    """
    Load a DICOM series from a directory.
    
    Args:
        directory: Path to directory containing DICOM files
        
    Returns:
        SimpleITK image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def load_nifti(file_path):
    """
    Load a NIfTI file.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        SimpleITK image
    """
    return sitk.ReadImage(file_path)


def load_medical_image(path):
    """
    Load a medical image from a file or directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        SimpleITK image
    """
    path = Path(path)
    
    if path.is_dir():
        # Try to load as DICOM series
        return load_dicom_series(str(path))
    else:
        # Try to load as NIfTI
        if path.suffix in ['.nii', '.gz']:
            return load_nifti(str(path))
        else:
            # Try to load as single DICOM file
            return sitk.ReadImage(str(path))


def save_medical_image(image, output_path):
    """
    Save a medical image to a file.
    
    Args:
        image: SimpleITK image to save
        output_path: Path to save the image
    """
    output_path = Path(output_path)
    
    # Create parent directories if they don't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save based on file extension
    if output_path.suffix in ['.nii', '.gz']:
        sitk.WriteImage(image, str(output_path))
    elif output_path.suffix in ['.dcm']:
        sitk.WriteImage(image, str(output_path))
    else:
        # Default to NIfTI
        sitk.WriteImage(image, str(output_path))


def save_as_nifti(image, output_path):
    """
    Save a SimpleITK image as NIfTI.
    
    Args:
        image: SimpleITK image
        output_path: Path to save the NIfTI file
    """
    sitk.WriteImage(image, output_path)


def save_as_dicom(image, output_dir, patient_info=None):
    """
    Save a SimpleITK image as DICOM series.
    
    Args:
        image: SimpleITK image
        output_dir: Directory to save the DICOM series
        patient_info: Dictionary with patient information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert SimpleITK image to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Get image properties
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    
    # Create a base DICOM dataset
    if patient_info is None:
        patient_info = {
            'PatientName': 'ANONYMOUS',
            'PatientID': 'ANONYMOUS',
            'PatientBirthDate': '',
            'PatientSex': '',
            'StudyDescription': 'Synthetic CT',
            'SeriesDescription': 'Synthetic CT from MRI',
        }
    
    # Generate UIDs
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    
    # Save each slice as a separate DICOM file
    for i in range(array.shape[0]):
        # Create a new DICOM dataset for this slice
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = FileDataset(
            os.path.join(output_dir, f'slice_{i:04d}.dcm'),
            {},
            file_meta=file_meta,
            preamble=b'\0' * 128
        )
        
        # Patient information
        ds.PatientName = patient_info.get('PatientName', 'ANONYMOUS')
        ds.PatientID = patient_info.get('PatientID', 'ANONYMOUS')
        ds.PatientBirthDate = patient_info.get('PatientBirthDate', '')
        ds.PatientSex = patient_info.get('PatientSex', '')
        
        # Study information
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        ds.StudyDescription = patient_info.get('StudyDescription', 'Synthetic CT')
        
        # Series information
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = 1
        ds.SeriesDescription = patient_info.get('SeriesDescription', 'Synthetic CT from MRI')
        
        # Image information
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = 'CT'
        
        # CT-specific attributes
        ds.RescaleIntercept = -1024.0
        ds.RescaleSlope = 1.0
        ds.RescaleType = 'HU'
        
        # Image position and orientation
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + i * spacing[2]]
        ds.ImageOrientationPatient = [
            direction[0], direction[1], direction[2],
            direction[3], direction[4], direction[5]
        ]
        
        # Pixel spacing
        ds.PixelSpacing = [spacing[0], spacing[1]]
        ds.SliceThickness = spacing[2]
        
        # Pixel data
        ds.Rows = array.shape[1]
        ds.Columns = array.shape[2]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # Signed
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        
        # Convert HU values to pixel values
        pixel_array = array[i].astype(np.int16)
        ds.PixelData = pixel_array.tobytes()
        
        # Save the DICOM file
        ds.save_as(os.path.join(output_dir, f'slice_{i:04d}.dcm'))


class SyntheticCT:
    """
    Class to represent a synthetic CT image.
    """
    
    def __init__(self, image, metadata=None):
        """
        Initialize a synthetic CT image.
        
        Args:
            image: SimpleITK image
            metadata: Dictionary with metadata
        """
        self.image = image
        if metadata is None:
            metadata = {
                "creation_time": datetime.datetime.now().isoformat()
            }
        self.metadata = metadata
    
    def save(self, output_path):
        """
        Save the synthetic CT image.
        
        Args:
            output_path: Path to save the image
        """
        output_path = Path(output_path)
        
        if output_path.suffix in ['.nii', '.gz']:
            save_as_nifti(self.image, str(output_path))
        elif output_path.is_dir():
            save_as_dicom(self.image, str(output_path), self.metadata.get('patient_info'))
        else:
            # Default to NIfTI
            save_as_nifti(self.image, str(output_path))
    
    def get_array(self):
        """
        Get the image as a numpy array.
        
        Returns:
            Numpy array
        """
        return sitk.GetArrayFromImage(self.image)
    
    def get_metadata(self):
        """
        Get the metadata.
        
        Returns:
            Dictionary with metadata
        """
        return self.metadata
    
    @classmethod
    def load(cls, file_path, metadata=None):
        """
        Load a synthetic CT image from a file.
        
        Args:
            file_path: Path to the image file
            metadata: Optional metadata dictionary
            
        Returns:
            SyntheticCT object
        """
        image = load_medical_image(file_path)
        return cls(image, metadata)


def validate_input_file(file_path):
    """
    Kiểm tra tính hợp lệ của file đầu vào.
    
    Args:
        file_path (str): Đường dẫn đến file cần kiểm tra
        
    Returns:
        bool: True nếu file hợp lệ, False nếu không
    """
    file_path = Path(file_path)
    
    # Kiểm tra sự tồn tại của file
    if not file_path.exists():
        logging.error(f"File không tồn tại: {file_path}")
        return False
        
    # Kiểm tra định dạng file
    if file_path.is_dir():
        # Nếu là thư mục, giả định là chuỗi DICOM
        dicom_files = list(file_path.glob('*.dcm'))
        if not dicom_files:
            logging.error(f"Không tìm thấy file DICOM trong thư mục: {file_path}")
            return False
    else:
        # Nếu là file, kiểm tra phần mở rộng
        valid_extensions = ['.nii', '.nii.gz', '.dcm', '.mha', '.mhd', '.nrrd']
        if not any(str(file_path).lower().endswith(ext) for ext in valid_extensions):
            logging.error(f"Định dạng file không được hỗ trợ: {file_path}")
            logging.error(f"Các định dạng được hỗ trợ: {', '.join(valid_extensions)}")
            return False
    
    return True


def ensure_output_dir(output_path):
    """
    Đảm bảo thư mục đầu ra tồn tại.
    
    Args:
        output_path (str): Đường dẫn đến thư mục hoặc file đầu ra
        
    Returns:
        Path: Đối tượng Path của thư mục đầu ra
    """
    output_path = Path(output_path)
    
    # Nếu output_path là file, lấy thư mục cha
    if output_path.suffix:
        output_dir = output_path.parent
    else:
        output_dir = output_path
        
    # Tạo thư mục nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def load_multi_sequence_mri(paths, sequence_types=None):
    """
    Tải nhiều chuỗi MRI và đăng ký chúng để sẵn sàng cho xử lý.
    
    Args:
        paths (list): Danh sách các đường dẫn đến các file MRI
        sequence_types (list, optional): Danh sách các loại chuỗi MRI tương ứng với paths
            (ví dụ: ['T1', 'T2', 'FLAIR']). Mặc định là None.
            
    Returns:
        MultiSequenceMRI: Đối tượng chứa nhiều chuỗi MRI đã được đăng ký
    """
    if not paths:
        raise ValueError("Danh sách đường dẫn không được để trống")
        
    if sequence_types and len(sequence_types) != len(paths):
        raise ValueError("Số lượng loại chuỗi phải bằng số lượng đường dẫn")
        
    # Tải các chuỗi MRI
    mri_sequences = {}
    reference_image = None
    
    for i, path in enumerate(paths):
        # Xác định tên chuỗi
        seq_name = sequence_types[i] if sequence_types else f"sequence_{i+1}"
        
        logging.info(f"Tải chuỗi MRI {seq_name} từ {path}")
        image = load_medical_image(path)
        
        if reference_image is None:
            reference_image = image
            mri_sequences[seq_name] = image
        else:
            # Đăng ký ảnh với ảnh tham chiếu
            logging.info(f"Đăng ký chuỗi {seq_name} với chuỗi tham chiếu")
            registered_image = register_image(image, reference_image)
            mri_sequences[seq_name] = registered_image
    
    return MultiSequenceMRI(mri_sequences)


class MultiSequenceMRI:
    """Lớp đại diện cho nhiều chuỗi MRI đã được đăng ký."""
    
    def __init__(self, sequences):
        """
        Khởi tạo đối tượng MultiSequenceMRI.
        
        Args:
            sequences (dict): Dictionary các chuỗi MRI, với key là tên chuỗi
        """
        self.sequences = sequences
        self.reference_name = next(iter(sequences.keys()))
        
    def get_sequence(self, name):
        """Lấy một chuỗi cụ thể theo tên."""
        if name not in self.sequences:
            raise ValueError(f"Chuỗi '{name}' không tồn tại")
        return self.sequences[name]
        
    def get_reference(self):
        """Lấy chuỗi tham chiếu."""
        return self.sequences[self.reference_name]
        
    def get_all_sequences(self):
        """Lấy tất cả các chuỗi."""
        return self.sequences
        
    def get_sequence_names(self):
        """Lấy danh sách tên các chuỗi."""
        return list(self.sequences.keys())
        
    def save(self, output_dir):
        """Lưu tất cả các chuỗi vào thư mục."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, image in self.sequences.items():
            output_path = output_dir / f"{name}.nii.gz"
            save_medical_image(image, str(output_path))
            
        return output_dir 


def register_image(moving_image, fixed_image, transform_type='rigid'):
    """
    Đăng ký (căn chỉnh) ảnh di động với ảnh cố định.
    
    Args:
        moving_image: Ảnh cần căn chỉnh
        fixed_image: Ảnh tham chiếu cố định
        transform_type (str): Loại biến đổi ('rigid', 'affine', hoặc 'bspline')
        
    Returns:
        Ảnh đã được căn chỉnh
    """
    import SimpleITK as sitk
    
    logging.info(f"Đăng ký ảnh sử dụng phương pháp {transform_type}")
    
    # Tạo đối tượng registration
    registration_method = sitk.ImageRegistrationMethod()
    
    # Thiết lập phép đo tương tự
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    # Thiết lập bộ nội suy
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Thiết lập bộ tối ưu hóa
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Thiết lập loại biến đổi
    if transform_type == 'rigid':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == 'affine':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == 'bspline':
        mesh_size = [10, 10, 10]
        initial_transform = sitk.BSplineTransformInitializer(
            fixed_image, 
            mesh_size
        )
    else:
        raise ValueError(f"Loại biến đổi không được hỗ trợ: {transform_type}")
    
    registration_method.SetInitialTransform(initial_transform)
    
    # Thực hiện đăng ký
    try:
        final_transform = registration_method.Execute(fixed_image, moving_image)
        logging.info("Đăng ký ảnh thành công")
    except Exception as e:
        logging.error(f"Lỗi trong quá trình đăng ký ảnh: {str(e)}")
        logging.warning("Sử dụng biến đổi ban đầu")
        final_transform = initial_transform
    
    # Áp dụng biến đổi
    registered_image = sitk.Resample(
        moving_image, 
        fixed_image, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_image.GetPixelID()
    )
    
    return registered_image 