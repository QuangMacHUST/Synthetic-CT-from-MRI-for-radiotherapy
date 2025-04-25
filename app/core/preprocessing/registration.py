#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for MRI image registration techniques.

This module provides various methods for registering MRI images,
which is a critical step in the preprocessing pipeline for synthetic CT generation,
especially when dealing with multiple MRI sequences or multi-modal registration.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Union, Optional, Tuple, List

# Set up logger
logger = logging.getLogger(__name__)

def register_images(fixed_image: Union[sitk.Image, np.ndarray], 
                   moving_image: Union[sitk.Image, np.ndarray],
                   method: str = 'rigid',
                   fixed_mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                   moving_mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                   params: Optional[Dict[str, Any]] = None) -> Tuple[Union[sitk.Image, np.ndarray], sitk.Transform]:
    """
    Register a moving image to a fixed image.
    
    Args:
        fixed_image: Fixed (reference) image as SimpleITK image or numpy array
        moving_image: Moving image to be registered as SimpleITK image or numpy array
        method: Registration method ('rigid', 'affine', 'bspline', 'demons')
        fixed_mask: Optional mask for the fixed image to focus registration
        moving_mask: Optional mask for the moving image to focus registration
        params: Additional parameters for the registration method
    
    Returns:
        Tuple of (registered_image, transform)
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Convert numpy arrays to SimpleITK images if needed
    fixed_is_numpy = isinstance(fixed_image, np.ndarray)
    moving_is_numpy = isinstance(moving_image, np.ndarray)
    
    if fixed_is_numpy:
        fixed_sitk = sitk.GetImageFromArray(fixed_image)
    else:
        fixed_sitk = fixed_image
    
    if moving_is_numpy:
        moving_sitk = sitk.GetImageFromArray(moving_image)
        # Transfer properties from fixed image if available
        if not fixed_is_numpy:
            moving_sitk.SetSpacing(fixed_sitk.GetSpacing())
            moving_sitk.SetOrigin(fixed_sitk.GetOrigin())
            moving_sitk.SetDirection(fixed_sitk.GetDirection())
    else:
        moving_sitk = moving_image
    
    # Convert masks to SimpleITK if provided
    if fixed_mask is not None:
        if isinstance(fixed_mask, np.ndarray):
            fixed_mask_sitk = sitk.GetImageFromArray(fixed_mask.astype(np.uint8))
            # Transfer properties from fixed image
            fixed_mask_sitk.SetSpacing(fixed_sitk.GetSpacing())
            fixed_mask_sitk.SetOrigin(fixed_sitk.GetOrigin())
            fixed_mask_sitk.SetDirection(fixed_sitk.GetDirection())
        else:
            fixed_mask_sitk = fixed_mask
    else:
        fixed_mask_sitk = None
    
    if moving_mask is not None:
        if isinstance(moving_mask, np.ndarray):
            moving_mask_sitk = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
            # Transfer properties from moving image
            moving_mask_sitk.SetSpacing(moving_sitk.GetSpacing())
            moving_mask_sitk.SetOrigin(moving_sitk.GetOrigin())
            moving_mask_sitk.SetDirection(moving_sitk.GetDirection())
        else:
            moving_mask_sitk = moving_mask
    else:
        moving_mask_sitk = None
    
    # Apply appropriate registration method
    if method.lower() == 'rigid':
        registered_image, transform = rigid_registration(fixed_sitk, moving_sitk, fixed_mask_sitk, moving_mask_sitk, params)
    elif method.lower() == 'affine':
        registered_image, transform = affine_registration(fixed_sitk, moving_sitk, fixed_mask_sitk, moving_mask_sitk, params)
    elif method.lower() == 'bspline':
        registered_image, transform = bspline_registration(fixed_sitk, moving_sitk, fixed_mask_sitk, moving_mask_sitk, params)
    elif method.lower() == 'demons':
        registered_image, transform = demons_registration(fixed_sitk, moving_sitk, fixed_mask_sitk, moving_mask_sitk, params)
    else:
        logger.warning(f"Unknown registration method '{method}'. Using rigid registration.")
        registered_image, transform = rigid_registration(fixed_sitk, moving_sitk, fixed_mask_sitk, moving_mask_sitk, params)
    
    # Convert result back to numpy if input was numpy
    if fixed_is_numpy and moving_is_numpy:
        return sitk.GetArrayFromImage(registered_image), transform
    else:
        return registered_image, transform

def initialize_registration(fixed_image: sitk.Image, 
                           moving_image: sitk.Image, 
                           fixed_mask: Optional[sitk.Image] = None,
                           moving_mask: Optional[sitk.Image] = None,
                           params: Optional[Dict[str, Any]] = None) -> Tuple[sitk.ImageRegistrationMethod, Optional[sitk.Transform]]:
    """
    Initialize a registration method with common parameters.
    
    Args:
        fixed_image: Fixed (reference) SimpleITK image
        moving_image: Moving SimpleITK image
        fixed_mask: Optional mask for the fixed image
        moving_mask: Optional mask for the moving image
        params: Additional parameters for registration initialization
    
    Returns:
        Tuple of (registration_method, initial_transform)
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Extract common parameters
    use_histogram_matching = params.get('use_histogram_matching', True)
    sampling_percentage = params.get('sampling_percentage', 0.01)
    sampling_strategy = params.get('sampling_strategy', 'RANDOM')
    learning_rate = params.get('learning_rate', 1.0)
    number_of_iterations = params.get('number_of_iterations', 100)
    convergence_min_value = params.get('convergence_min_value', 1e-6)
    convergence_window_size = params.get('convergence_window_size', 10)
    shrink_factors = params.get('shrink_factors', [4, 2, 1])
    smoothing_sigmas = params.get('smoothing_sigmas', [2, 1, 0])
    
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set common parameters
    if fixed_mask is not None:
        registration_method.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration_method.SetMetricMovingMask(moving_mask)
    
    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Set optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=learning_rate,
        numberOfIterations=number_of_iterations,
        convergenceMinimumValue=convergence_min_value,
        convergenceWindowSize=convergence_window_size
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Set up the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Optional histogram matching
    if use_histogram_matching:
        registration_method.SetMetricUseHistogramMatching(True)
    
    # Set up sampling strategy
    if sampling_strategy.upper() == 'RANDOM':
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    elif sampling_strategy.upper() == 'REGULAR':
        registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    else:
        registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    
    registration_method.SetMetricSamplingPercentage(sampling_percentage)
    
    # Initialize with center of mass if specified
    initial_transform = None
    initial_method = params.get('initial_transform', 'NONE')
    
    if initial_method.upper() == 'CENTER':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif initial_method.upper() == 'MOMENTS':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
    
    return registration_method, initial_transform

def rigid_registration(fixed_image: sitk.Image, 
                      moving_image: sitk.Image,
                      fixed_mask: Optional[sitk.Image] = None,
                      moving_mask: Optional[sitk.Image] = None,
                      params: Optional[Dict[str, Any]] = None) -> Tuple[sitk.Image, sitk.Transform]:
    """
    Perform rigid registration between two images.
    
    Args:
        fixed_image: Fixed (reference) SimpleITK image
        moving_image: Moving SimpleITK image
        fixed_mask: Optional mask for the fixed image
        moving_mask: Optional mask for the moving image
        params: Additional parameters for rigid registration
    
    Returns:
        Tuple of (registered_image, transform)
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Initialize registration method
    registration_method, initial_transform = initialize_registration(
        fixed_image, moving_image, fixed_mask, moving_mask, params
    )
    
    # Set up the metric
    metric = params.get('metric', 'MI')
    if metric.upper() == 'MI':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric.upper() == 'CORRELATION':
        registration_method.SetMetricAsCorrelation()
    elif metric.upper() == 'MEAN_SQUARES':
        registration_method.SetMetricAsMeanSquares()
    else:
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # Set up the transform
    dimension = fixed_image.GetDimension()
    if dimension == 2:
        transform = sitk.Euler2DTransform()
    else:
        transform = sitk.Euler3DTransform()
    
    # Initialize transform if provided
    if initial_transform is not None:
        transform.SetParameters(initial_transform.GetParameters())
        transform.SetFixedParameters(initial_transform.GetFixedParameters())
    
    registration_method.SetInitialTransform(transform, inPlace=True)
    
    # Execute the registration
    try:
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        # Apply the transform to the moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
        
        registered_image = resampler.Execute(moving_image)
        
        logger.info("Rigid registration completed successfully.")
        
        return registered_image, final_transform
    except Exception as e:
        logger.error(f"Rigid registration failed: {str(e)}")
        logger.warning("Returning original moving image.")
        return moving_image, transform

def affine_registration(fixed_image: sitk.Image, 
                       moving_image: sitk.Image,
                       fixed_mask: Optional[sitk.Image] = None,
                       moving_mask: Optional[sitk.Image] = None,
                       params: Optional[Dict[str, Any]] = None) -> Tuple[sitk.Image, sitk.Transform]:
    """
    Perform affine registration between two images.
    
    Args:
        fixed_image: Fixed (reference) SimpleITK image
        moving_image: Moving SimpleITK image
        fixed_mask: Optional mask for the fixed image
        moving_mask: Optional mask for the moving image
        params: Additional parameters for affine registration
    
    Returns:
        Tuple of (registered_image, transform)
    """
    # Default parameters
    if params is None:
        params = {}
    
    # First perform rigid registration for better initialization
    if params.get('use_rigid_initialization', True):
        _, rigid_transform = rigid_registration(
            fixed_image, moving_image, fixed_mask, moving_mask, params
        )
        
        # Apply rigid transform to moving image for better initialization
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(rigid_transform)
        
        moving_image_rigid = resampler.Execute(moving_image)
    else:
        moving_image_rigid = moving_image
        rigid_transform = None
    
    # Initialize registration method
    registration_method, initial_transform = initialize_registration(
        fixed_image, moving_image_rigid, fixed_mask, moving_mask, params
    )
    
    # Set up the metric
    metric = params.get('metric', 'MI')
    if metric.upper() == 'MI':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric.upper() == 'CORRELATION':
        registration_method.SetMetricAsCorrelation()
    elif metric.upper() == 'MEAN_SQUARES':
        registration_method.SetMetricAsMeanSquares()
    else:
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # Set up the transform
    dimension = fixed_image.GetDimension()
    if dimension == 2:
        transform = sitk.AffineTransform(2)
    else:
        transform = sitk.AffineTransform(3)
    
    # Initialize with rigid transform if available
    if rigid_transform is not None:
        # Convert rigid to affine
        affine_transform = convert_rigid_to_affine(rigid_transform)
        transform.SetParameters(affine_transform.GetParameters())
        transform.SetFixedParameters(affine_transform.GetFixedParameters())
    elif initial_transform is not None:
        # Convert initial transform to affine
        affine_transform = convert_rigid_to_affine(initial_transform)
        transform.SetParameters(affine_transform.GetParameters())
        transform.SetFixedParameters(affine_transform.GetFixedParameters())
    
    registration_method.SetInitialTransform(transform, inPlace=True)
    
    # Execute the registration
    try:
        final_transform = registration_method.Execute(fixed_image, moving_image_rigid)
        
        # Apply the transform to the original moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        # If rigid was used for initialization, combine transforms
        if rigid_transform is not None and params.get('combine_transforms', True):
            final_composite_transform = sitk.CompositeTransform(fixed_image.GetDimension())
            final_composite_transform.AddTransform(final_transform)
            final_composite_transform.AddTransform(rigid_transform)
            resampler.SetTransform(final_composite_transform)
            final_transform = final_composite_transform
        else:
            resampler.SetTransform(final_transform)
        
        registered_image = resampler.Execute(moving_image)
        
        logger.info("Affine registration completed successfully.")
        
        return registered_image, final_transform
    except Exception as e:
        logger.error(f"Affine registration failed: {str(e)}")
        logger.warning("Returning original moving image.")
        return moving_image, transform

def bspline_registration(fixed_image: sitk.Image, 
                        moving_image: sitk.Image,
                        fixed_mask: Optional[sitk.Image] = None,
                        moving_mask: Optional[sitk.Image] = None,
                        params: Optional[Dict[str, Any]] = None) -> Tuple[sitk.Image, sitk.Transform]:
    """
    Perform B-spline registration between two images.
    
    Args:
        fixed_image: Fixed (reference) SimpleITK image
        moving_image: Moving SimpleITK image
        fixed_mask: Optional mask for the fixed image
        moving_mask: Optional mask for the moving image
        params: Additional parameters for B-spline registration
    
    Returns:
        Tuple of (registered_image, transform)
    """
    # Default parameters
    if params is None:
        params = {}
    
    # First perform affine registration for better initialization
    if params.get('use_affine_initialization', True):
        _, affine_transform = affine_registration(
            fixed_image, moving_image, fixed_mask, moving_mask, params
        )
        
        # Apply affine transform to moving image for better initialization
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(affine_transform)
        
        moving_image_affine = resampler.Execute(moving_image)
    else:
        moving_image_affine = moving_image
        affine_transform = None
    
    # Initialize registration method
    registration_method, _ = initialize_registration(
        fixed_image, moving_image_affine, fixed_mask, moving_mask, params
    )
    
    # Set up the metric
    metric = params.get('metric', 'MI')
    if metric.upper() == 'MI':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric.upper() == 'CORRELATION':
        registration_method.SetMetricAsCorrelation()
    elif metric.upper() == 'MEAN_SQUARES':
        registration_method.SetMetricAsMeanSquares()
    else:
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    # Set up the B-spline transform
    mesh_size = params.get('mesh_size', [8, 8, 8])
    if fixed_image.GetDimension() == 2:
        mesh_size = mesh_size[:2]
    
    transform_domain_mesh_size = mesh_size
    transform_domain_physical_dimensions = [
        size * spacing for size, spacing in zip(
            fixed_image.GetSize(), fixed_image.GetSpacing()
        )
    ]
    transform_domain_origin = fixed_image.GetOrigin()
    transform_domain_direction = fixed_image.GetDirection()
    
    transform = sitk.BSplineTransformInitializer(
        fixed_image, 
        transform_domain_mesh_size,
        order=3
    )
    
    registration_method.SetInitialTransform(transform, inPlace=True)
    
    # Execute the registration
    try:
        final_transform = registration_method.Execute(fixed_image, moving_image_affine)
        
        # Apply the transform to the original moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        # If affine was used for initialization, combine transforms
        if affine_transform is not None and params.get('combine_transforms', True):
            final_composite_transform = sitk.CompositeTransform(fixed_image.GetDimension())
            final_composite_transform.AddTransform(final_transform)
            final_composite_transform.AddTransform(affine_transform)
            resampler.SetTransform(final_composite_transform)
            final_transform = final_composite_transform
        else:
            resampler.SetTransform(final_transform)
        
        registered_image = resampler.Execute(moving_image)
        
        logger.info("B-spline registration completed successfully.")
        
        return registered_image, final_transform
    except Exception as e:
        logger.error(f"B-spline registration failed: {str(e)}")
        logger.warning("Returning original moving image.")
        return moving_image, transform

def demons_registration(fixed_image: sitk.Image, 
                       moving_image: sitk.Image,
                       fixed_mask: Optional[sitk.Image] = None,
                       moving_mask: Optional[sitk.Image] = None,
                       params: Optional[Dict[str, Any]] = None) -> Tuple[sitk.Image, sitk.Transform]:
    """
    Perform demons registration between two images.
    
    Args:
        fixed_image: Fixed (reference) SimpleITK image
        moving_image: Moving SimpleITK image
        fixed_mask: Optional mask for the fixed image
        moving_mask: Optional mask for the moving image
        params: Additional parameters for demons registration
    
    Returns:
        Tuple of (registered_image, transform)
    """
    # Default parameters
    if params is None:
        params = {}
    
    # First perform affine registration for better initialization
    if params.get('use_affine_initialization', True):
        _, affine_transform = affine_registration(
            fixed_image, moving_image, fixed_mask, moving_mask, params
        )
        
        # Apply affine transform to moving image for better initialization
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(affine_transform)
        
        moving_image_affine = resampler.Execute(moving_image)
    else:
        moving_image_affine = moving_image
        affine_transform = None
    
    # Extract demons parameters
    num_iterations = params.get('number_of_iterations', [50, 30, 20])
    smooth_sigmas = params.get('smooth_sigmas', [3, 2, 1])
    demons_type = params.get('demons_type', 'SYMMETRIC')
    
    # Use histogram matching for better results
    if params.get('use_histogram_matching', True):
        moving_image_affine = sitk.HistogramMatching(
            moving_image_affine, 
            fixed_image, 
            numberOfHistogramLevels=1024, 
            numberOfMatchPoints=7
        )
    
    # Select demons filter based on type
    if demons_type.upper() == 'SYMMETRIC':
        demons_filter = sitk.SymmetricForcesDemonsRegistrationFilter()
    elif demons_type.upper() == 'DIFFEOMORPHIC':
        demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()
    elif demons_type.upper() == 'FAST_SYMMETRIC':
        demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    else:
        logger.warning(f"Unknown demons type '{demons_type}'. Using symmetric demons.")
        demons_filter = sitk.SymmetricForcesDemonsRegistrationFilter()
    
    # Set up the demons filter
    demons_filter.SetNumberOfIterations(num_iterations)
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(smooth_sigmas)
    
    # Execute the registration
    try:
        displacement_field = demons_filter.Execute(fixed_image, moving_image_affine)
        
        # Get the transform
        transform = sitk.DisplacementFieldTransform(displacement_field)
        
        # Apply the transform to the original moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        # If affine was used for initialization, combine transforms
        if affine_transform is not None and params.get('combine_transforms', True):
            final_composite_transform = sitk.CompositeTransform(fixed_image.GetDimension())
            final_composite_transform.AddTransform(transform)
            final_composite_transform.AddTransform(affine_transform)
            resampler.SetTransform(final_composite_transform)
            final_transform = final_composite_transform
        else:
            resampler.SetTransform(transform)
            final_transform = transform
        
        registered_image = resampler.Execute(moving_image)
        
        logger.info("Demons registration completed successfully.")
        
        return registered_image, final_transform
    except Exception as e:
        logger.error(f"Demons registration failed: {str(e)}")
        logger.warning("Returning original moving image.")
        return moving_image, sitk.Transform()

def convert_rigid_to_affine(transform):
    """Convert a rigid transform to an affine transform."""
    dimension = transform.GetDimension()
    
    if dimension == 2:
        # For 2D transform
        affine_transform = sitk.AffineTransform(2)
    else:
        # For 3D transform
        affine_transform = sitk.AffineTransform(3)
    
    # Copy fixed parameters (center of rotation)
    affine_transform.SetFixedParameters(transform.GetFixedParameters())
    
    # For Euler3D transform, convert the parameters to an affine matrix,
    # then use that to set the parameters of the affine transform
    if isinstance(transform, (sitk.Euler3DTransform, sitk.Euler2DTransform)):
        affine_params = []
        
        if dimension == 2:
            # For 2D, we need to extract the rotation matrix and translation
            # from the Euler transform and set them in the affine transform
            # The affine parameters are [m11, m12, m21, m22, t1, t2]
            rotation = transform.GetMatrix()
            translation = transform.GetTranslation()
            
            # Fill parameters in the right order
            affine_params = [rotation[0], rotation[1], rotation[2], rotation[3], 
                           translation[0], translation[1]]
        else:
            # For 3D, we need to extract the rotation matrix and translation
            # The affine parameters are [m11, m12, m13, m21, ..., m33, t1, t2, t3]
            rotation = transform.GetMatrix()
            translation = transform.GetTranslation()
            
            # Fill parameters in the right order
            for i in range(9):
                affine_params.append(rotation[i])
            
            for i in range(3):
                affine_params.append(translation[i])
        
        affine_transform.SetParameters(affine_params)
    
    return affine_transform

def apply_transform(image: Union[sitk.Image, np.ndarray],
                  transform: sitk.Transform,
                  reference_image: Optional[Union[sitk.Image, np.ndarray]] = None,
                  interpolator: str = 'linear',
                  default_pixel_value: float = 0.0) -> Union[sitk.Image, np.ndarray]:
    """
    Apply a transform to an image.
    
    Args:
        image: Image to transform as SimpleITK image or numpy array
        transform: Transform to apply
        reference_image: Reference image for resampling (default: use input image)
        interpolator: Interpolation method ('linear', 'nearest', 'bspline', 'gaussian')
        default_pixel_value: Value for pixels outside the image domain
    
    Returns:
        Transformed image in the same format as input
    """
    # Convert numpy arrays to SimpleITK images if needed
    is_numpy = isinstance(image, np.ndarray)
    
    if is_numpy:
        sitk_image = sitk.GetImageFromArray(image)
        if reference_image is not None and isinstance(reference_image, np.ndarray):
            reference_sitk = sitk.GetImageFromArray(reference_image)
        else:
            reference_sitk = reference_image
    else:
        sitk_image = image
        reference_sitk = reference_image
    
    # Set reference image if not provided
    if reference_sitk is None:
        reference_sitk = sitk_image
    
    # Set interpolator
    if interpolator.lower() == 'linear':
        interp = sitk.sitkLinear
    elif interpolator.lower() == 'nearest':
        interp = sitk.sitkNearestNeighbor
    elif interpolator.lower() == 'bspline':
        interp = sitk.sitkBSpline
    elif interpolator.lower() == 'gaussian':
        interp = sitk.sitkGaussian
    else:
        logger.warning(f"Unknown interpolator '{interpolator}'. Using linear interpolation.")
        interp = sitk.sitkLinear
    
    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_sitk)
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(transform)
    
    transformed_image = resampler.Execute(sitk_image)
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        transformed_image = sitk.GetArrayFromImage(transformed_image)
    
    return transformed_image 