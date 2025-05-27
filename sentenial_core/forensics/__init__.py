#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentenial-X Forensics Module
----------------------------
Core digital forensics capabilities for incident response and threat hunting.

This package provides digital forensics tools for memory analysis, disk forensics,
and artifact collection with chain-of-custody preservation.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Module metadata
__version__ = '0.3.2'
__author__ = 'Erik G.'
__license__ = 'Proprietary'
__maintainer__ = 'Sentenial-X Team'

# Configure module logger
logger = logging.getLogger('SentenialX.forensics')

# Import submodules to make them available when importing the package
try:
    from .memory import MemoryDump, ProcessAnalyzer
    from .disk import DiskImage, FileSystemAnalyzer
    from .network import PacketCapture, NetworkTrafficAnalyzer
    from .timeline import EventTimeline, TimelineBuilder
except ImportError as e:
    logger.warning(f"Some forensics components couldn't be imported: {e}")

# Forensics module configuration
DEFAULT_CONFIG = {
    'evidence_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evidence'),
    'hash_algorithms': ['sha256', 'md5'],
    'compress_artifacts': True,
    'preserve_timestamps': True,
    'chain_of_custody': True,
}

# Ensure evidence directory exists
os.makedirs(DEFAULT_CONFIG['evidence_dir'], exist_ok=True)

def get_runtime_info() -> Dict[str, str]:
    """
    Get runtime environment information for forensic context.
    
    Returns:
        Dict containing platform, Python version and timestamp information
    """
    import platform
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'module_version': __version__,
    }

def generate_case_id() -> str:
    """
    Generate a unique case identifier for forensic investigations.
    
    Returns:
        String containing timestamped case ID
    """
    import uuid
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    return f"SX-{timestamp}-{unique_id}"

def init_investigation(case_name: str, analyst: str = None) -> Dict[str, str]:
    """
    Initialize a new forensic investigation with proper case management.
    
    Args:
        case_name: Human-readable name for the investigation
        analyst: Name of the analyst (defaults to system user)
        
    Returns:
        Dictionary with case information including paths and IDs
    """
    case_id = generate_case_id()
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    if not analyst:
        try:
            import getpass
            analyst = getpass.getuser()
        except Exception:
            analyst = "unknown"
    
    case_dir = os.path.join(DEFAULT_CONFIG['evidence_dir'], case_id)
    os.makedirs(case_dir, exist_ok=True)
    
    # Create case index file
    with open(os.path.join(case_dir, 'case.info'), 'w') as f:
        f.write(f"Case ID: {case_id}\n")
        f.write(f"Name: {case_name}\n")
        f.write(f"Created: {timestamp}\n")
        f.write(f"Analyst: {analyst}\n")
        f.write(f"Status: Active\n")
    
    logger.info(f"Investigation initialized: {case_id} - {case_name}")
    
    # Return case details
    return {
        'case_id': case_id,
        'name': case_name,
        'timestamp': timestamp,
        'analyst': analyst,
        'evidence_path': case_dir,
    }

# Expose key functionality at package level
__all__ = [
    'MemoryDump',
    'DiskImage',
    'PacketCapture',
    'EventTimeline',
    'get_runtime_info',
    'generate_case_id',
    'init_investigation',
    'DEFAULT_CONFIG',
    '__version__',
]