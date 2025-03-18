#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Logging utilities for MRI to CT conversion application.
"""

import os
import sys
import logging
import datetime
from pathlib import Path


def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (optional)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    
    # Create logs directory if log_file is not specified
    if log_file is None:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Default log file with timestamp
        log_file = logs_dir / f"synthetic_ct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Add file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Suppress overly verbose logs from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Log environment info
    logger.info("Logging initialized")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name):
    """
    Get logger with specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_exception(logger, e, context=""):
    """
    Log exception with traceback.
    
    Args:
        logger: Logger instance
        e: Exception
        context: Context information
    """
    import traceback
    
    if context:
        logger.error(f"{context}: {str(e)}")
    else:
        logger.error(str(e))
        
    logger.debug(traceback.format_exc()) 