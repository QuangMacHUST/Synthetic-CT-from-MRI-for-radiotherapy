#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Server for synthetic CT generation.
"""

import os
import logging
import tempfile
from typing import Optional, Dict, Any, List, Union

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, File, UploadFile, Form, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.utils.config_utils import ConfigManager, get_config, get_region_params
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues
from app.core import convert_to_ct
from app.core.evaluation import evaluate_synthetic_ct
from app.utils.io_utils import SyntheticCT, MultiSequenceMRI, save_medical_image, load_medical_image

# Create FastAPI app
app = FastAPI(
    title="Synthetic CT Generator API",
    description="API for generating synthetic CT images from MRI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
api_key = None
model_cache: Dict[str, Any] = {}
logger = logging.getLogger(__name__)
API_KEY_NAME = "X-API-Key"

# API Key security
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate API Key."""
    if api_key and api_key_header != api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return api_key_header


class ConversionRequest(BaseModel):
    """Conversion request model."""
    region: str = "brain"
    method: str = "gan"
    preprocess: bool = True
    output_format: str = "nifti"


class StatusResponse(BaseModel):
    """Status response model."""
    status: str
    message: str
    task_id: Optional[str] = None


def initialize_models(model: str, region: str, config: Optional[ConfigManager] = None) -> None:
    """
    Initialize models for conversion.
    
    Args:
        model: Conversion model to use
        region: Anatomical region
        config: Configuration manager
    """
    global model_cache
    
    # Get configuration
    if config is None:
        config = get_config()
    
    # Get region-specific parameters
    region_params = get_region_params(region)
    
    # Initialize models based on conversion method
    if model == "atlas":
        # Load atlas data
        atlas_path = region_params.get("atlas_path")
        if atlas_path and os.path.exists(atlas_path):
            model_cache["atlas"] = {"path": atlas_path, "data": None}
            logger.info(f"Atlas initialized: {atlas_path}")
        else:
            logger.warning(f"Atlas not found: {atlas_path}")
    
    elif model == "cnn":
        # Load CNN model
        model_path = region_params.get("model_path")
        if model_path and os.path.exists(model_path):
            model_cache["cnn"] = {"path": model_path, "model": None}
            logger.info(f"CNN model initialized: {model_path}")
        else:
            logger.warning(f"CNN model not found: {model_path}")
    
    elif model == "gan":
        # Load GAN model
        generator_path = region_params.get("generator_path")
        if generator_path and os.path.exists(generator_path):
            model_cache["gan"] = {"path": generator_path, "model": None}
            logger.info(f"GAN model initialized: {generator_path}")
        else:
            logger.warning(f"GAN model not found: {generator_path}")
    
    else:
        logger.warning(f"Unknown model type: {model}")


@app.get("/")
async def root():
    """Get API server info."""
    return {
        "name": "Synthetic CT Generator API",
        "version": "1.0.0",
        "description": "API for generating synthetic CT images from MRI",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/convert", response_model=StatusResponse, dependencies=[Depends(get_api_key)])
async def convert_mri(
    background_tasks: BackgroundTasks,
    mri_file: UploadFile = File(...),
    params: ConversionRequest = Depends()
):
    """Convert MRI to synthetic CT."""
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            mri_path = os.path.join(temp_dir, mri_file.filename)
            with open(mri_path, "wb") as f:
                f.write(await mri_file.read())
            
            # Process in background
            task_id = f"task_{os.path.basename(mri_path)}"
            background_tasks.add_task(
                process_mri_to_ct,
                mri_path=mri_path,
                region=params.region,
                method=params.method,
                preprocess=params.preprocess,
                output_format=params.output_format,
                task_id=task_id
            )
            
            return {
                "status": "submitted",
                "message": "Conversion task submitted successfully",
                "task_id": task_id
            }
    except Exception as e:
        logger.error(f"Error processing conversion request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing conversion request: {str(e)}"
        )


async def process_mri_to_ct(
    mri_path: str,
    region: str,
    method: str,
    preprocess: bool,
    output_format: str,
    task_id: str
):
    """Process MRI to CT conversion as a background task."""
    try:
        # Load MRI data
        mri_data = load_medical_image(mri_path)
        
        # Preprocess MRI data
        if preprocess:
            mri_data = preprocess_mri(mri_data, region=region)
        
        # Segment tissues
        segmentation = segment_tissues(mri_data, region=region)
        
        # Convert to CT
        synthetic_ct = convert_to_ct(mri_data, segmentation, method=method, region=region)
        
        # Save CT data
        output_dir = os.path.join(get_config().get("app", "output_dir", default="output"))
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{task_id}_synthetic_ct")
        if output_format == "dicom":
            output_path = save_medical_image(synthetic_ct, output_path, format="dicom")
        else:
            output_path = save_medical_image(synthetic_ct, output_path, format="nifti")
        
        logger.info(f"Synthetic CT saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing MRI to CT conversion: {str(e)}")


def start_server(
    model: str,
    region: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    api_key_param: Optional[str] = None,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
    config: Optional[ConfigManager] = None
) -> None:
    """
    Start API server.
    
    Args:
        model: Conversion model to use
        region: Anatomical region
        host: Host to run server on
        port: Port to run server on
        workers: Number of worker processes
        api_key_param: API key for authentication
        ssl_cert: Path to SSL certificate
        ssl_key: Path to SSL key
        config: Configuration manager
    """
    global api_key
    api_key = api_key_param
    
    # Initialize models
    initialize_models(model, region, config)
    
    # Start server
    uvicorn.run(
        "app.deployment.api_server:app",
        host=host,
        port=port,
        workers=workers,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key
    )