#!/usr/bin/env python3
"""
file_manager.py
================
FITS File Upload and Management System
Handles FITS file uploads, validation, and storage for the multi-agent system
"""

import os
import uuid
import shutil
import logging
import hashlib
import mimetypes
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from astropy.io import fits
    import numpy as np
except ImportError:
    print("❌ Please install dependencies first:")
    print("   pip install astropy numpy")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FITSFileInfo:
    """Complete FITS file information"""
    file_id: str
    original_filename: str
    file_path: str
    file_size: int
    upload_time: datetime
    is_valid: bool
    validation_error: Optional[str] = None
    
    # FITS specific metadata
    n_hdus: int = 0
    has_data: bool = False
    data_shapes: List[tuple] = field(default_factory=list)
    header_info: Dict[str, Any] = field(default_factory=dict)
    column_info: Dict[str, List[str]] = field(default_factory=dict)
    
    # Analysis readiness
    lightcurve_ready: bool = False
    recommended_column: Optional[str] = None
    data_quality_score: float = 0.0


@dataclass
class FileManagerStats:
    """File manager statistics"""
    total_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    total_size_mb: float = 0.0
    lightcurve_ready_files: int = 0
    avg_quality_score: float = 0.0


class FITSFileManager:
    """
    🗂️ Advanced FITS File Upload and Management System
    
    Features:
    - Secure file upload with validation
    - Comprehensive FITS format analysis
    - Lightcurve data detection
    - Data quality assessment
    - Automatic cleanup
    - Error recovery
    """
    
    def __init__(self, 
                 upload_dir: str = "storage/uploads",
                 max_file_size: int = 500 * 1024 * 1024,  # 500MB
                 allowed_extensions: List[str] = None):
        
        self.upload_dir = Path(upload_dir)
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or ['.fits', '.fit', '.fts']
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # File tracking
        self.uploaded_files: Dict[str, FITSFileInfo] = {}
        self.upload_locks: Dict[str, bool] = {}  # Prevent concurrent uploads
        
        # Statistics
        self.stats = FileManagerStats()
        
        logger.info(f"FITS File Manager initialized")
        logger.info(f"Upload directory: {self.upload_dir}")
        logger.info(f"Max file size: {self.max_file_size / (1024*1024):.1f} MB")
    
    async def upload_fits_file(self, 
                             file_data: bytes, 
                             filename: str,
                             user_id: Optional[str] = None,
                             validate_immediately: bool = True) -> FITSFileInfo:
        """
        Upload and process FITS file
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            user_id: Optional user identifier
            validate_immediately: Whether to validate on upload
            
        Returns:
            FITSFileInfo with complete metadata
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Basic validation
        await self._validate_upload_request(file_data, filename)
        
        # Check for concurrent uploads
        if filename in self.upload_locks:
            raise ValueError(f"File {filename} is already being uploaded")
        
        self.upload_locks[filename] = True
        
        try:
            # Create secure file path
            file_extension = self._get_secure_extension(filename)
            safe_filename = f"{file_id}{file_extension}"
            file_path = self.upload_dir / safe_filename
            
            # Write file atomically (write to temp, then move)
            temp_path = file_path.with_suffix(f"{file_extension}.tmp")
            
            try:
                with open(temp_path, 'wb') as f:
                    f.write(file_data)
                
                # Move to final location
                temp_path.rename(file_path)
                
            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise
            
            # Create file info object
            file_info = FITSFileInfo(
                file_id=file_id,
                original_filename=filename,
                file_path=str(file_path),
                file_size=len(file_data),
                upload_time=datetime.now(),
                is_valid=False
            )
            
            # Validate FITS format and extract metadata
            if validate_immediately:
                file_info = await self._comprehensive_fits_validation(file_info)
            else:
                file_info.is_valid = True  # Assume valid for deferred validation
            
            # Store file info
            self.uploaded_files[file_id] = file_info
            self._update_stats()
            
            logger.info(f"File uploaded successfully: {filename} → {file_id}")
            logger.info(f"Validation: {'✅ PASS' if file_info.is_valid else '❌ FAIL'}")
            
            return file_info
            
        except Exception as e:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            
            logger.error(f"File upload failed: {filename} - {str(e)}")
            raise
        
        finally:
            # Release upload lock
            self.upload_locks.pop(filename, None)
    
    async def upload_from_path(self, 
                             source_path: str, 
                             user_id: Optional[str] = None) -> FITSFileInfo:
        """Upload FITS file from local filesystem path"""
        
        source_path = Path(source_path)
        
        # Validate source file
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if not source_path.is_file():
            raise ValueError(f"Path is not a file: {source_path}")
        
        # Check file size before reading
        file_size = source_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Read file data
        try:
            with open(source_path, 'rb') as f:
                file_data = f.read()
        except Exception as e:
            raise IOError(f"Failed to read source file: {str(e)}")
        
        return await self.upload_fits_file(
            file_data=file_data,
            filename=source_path.name,
            user_id=user_id,
            validate_immediately=True
        )
    
    async def _validate_upload_request(self, file_data: bytes, filename: str):
        """Validate upload request before processing"""
        
        # Check file size
        if len(file_data) > self.max_file_size:
            raise ValueError(
                f"File too large: {len(file_data)} bytes "
                f"(max: {self.max_file_size / (1024*1024):.1f} MB)"
            )
        
        # Check filename
        if not filename or len(filename) > 255:
            raise ValueError("Invalid filename")
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise ValueError(
                f"Invalid file extension: {file_ext}. "
                f"Allowed: {', '.join(self.allowed_extensions)}"
            )
        
        # Check for empty file
        if len(file_data) == 0:
            raise ValueError("Empty file not allowed")
        
        # Basic FITS header check (first 8 bytes should be "SIMPLE  ")
        if len(file_data) >= 8:
            header_start = file_data[:8].decode('ascii', errors='ignore')
            if not header_start.startswith('SIMPLE'):
                logger.warning(f"File {filename} doesn't start with FITS header")
    
    async def _comprehensive_fits_validation(self, file_info: FITSFileInfo) -> FITSFileInfo:
        """Comprehensive FITS validation and metadata extraction"""
        
        try:
            with fits.open(file_info.file_path, memmap=False) as hdul:
                
                # Basic structure validation
                n_hdus = len(hdul)
                if n_hdus == 0:
                    raise ValueError("Empty FITS file - no HDUs found")
                
                file_info.n_hdus = n_hdus
                
                # Analyze each HDU
                data_shapes = []
                column_info = {}
                has_data = False
                lightcurve_candidates = []
                
                for i, hdu in enumerate(hdul):
                    hdu_analysis = self._analyze_hdu(hdu, i)
                    
                    if hdu_analysis['has_data']:
                        has_data = True
                        data_shapes.append(hdu_analysis['shape'])
                        
                        # Check for lightcurve data
                        if hdu_analysis['is_table'] and hdu_analysis['columns']:
                            column_info[f"HDU_{i}"] = hdu_analysis['columns']
                            
                            # Look for time-series columns
                            time_columns = self._find_time_columns(hdu_analysis['columns'])
                            rate_columns = self._find_rate_columns(hdu_analysis['columns'])
                            
                            if time_columns and rate_columns:
                                lightcurve_candidates.append({
                                    'hdu_index': i,
                                    'time_column': time_columns[0],
                                    'rate_column': rate_columns[0],
                                    'n_rows': hdu_analysis['n_rows']
                                })
                
                # Extract header information
                primary_header = hdul[0].header
                header_info = self._extract_header_info(primary_header)
                
                # Assess data quality and lightcurve readiness
                quality_score, recommended_column = self._assess_data_quality(
                    hdul, lightcurve_candidates
                )
                
                # Update file info
                file_info.is_valid = True
                file_info.has_data = has_data
                file_info.data_shapes = data_shapes
                file_info.header_info = header_info
                file_info.column_info = column_info
                file_info.lightcurve_ready = len(lightcurve_candidates) > 0
                file_info.recommended_column = recommended_column
                file_info.data_quality_score = quality_score
                
                logger.info(f"FITS validation successful: {file_info.file_id}")
                logger.info(f"  HDUs: {n_hdus}, Data: {has_data}, "
                           f"Lightcurve ready: {file_info.lightcurve_ready}")
                
                if lightcurve_candidates:
                    logger.info(f"  Found {len(lightcurve_candidates)} lightcurve candidates")
                    logger.info(f"  Recommended column: {recommended_column}")
                
        except Exception as e:
            file_info.is_valid = False
            file_info.validation_error = str(e)
            logger.error(f"FITS validation failed: {file_info.file_id} - {str(e)}")
        
        return file_info
    
    def _analyze_hdu(self, hdu, index: int) -> Dict[str, Any]:
        """Analyze individual HDU"""
        
        analysis = {
            'index': index,
            'type': type(hdu).__name__,
            'has_data': hdu.data is not None,
            'shape': None,
            'is_table': False,
            'columns': [],
            'n_rows': 0
        }
        
        if hdu.data is not None:
            analysis['shape'] = hdu.data.shape
            
            # Check if it's a table
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                analysis['is_table'] = True
                analysis['columns'] = list(hdu.columns.names)
                analysis['n_rows'] = len(hdu.data)
        
        return analysis
    
    def _find_time_columns(self, columns: List[str]) -> List[str]:
        """Find potential time columns"""
        time_keywords = ['time', 't', 'mjd', 'utc', 'seconds', 'sec']
        time_columns = []
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                time_columns.append(col)
        
        return time_columns
    
    def _find_rate_columns(self, columns: List[str]) -> List[str]:
        """Find potential rate/count columns"""
        rate_keywords = ['rate', 'count', 'flux', 'intensity', 'signal']
        rate_columns = []
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in rate_keywords):
                rate_columns.append(col)
        
        return rate_columns
    
    def _extract_header_info(self, header) -> Dict[str, Any]:
        """Extract key header information"""
        
        info = {}
        
        # Standard FITS keywords
        standard_keys = [
            'OBJECT', 'OBSERVER', 'DATE-OBS', 'DATE', 'INSTRUME', 
            'TELESCOP', 'FILTER', 'EXPTIME', 'OBSTIME'
        ]
        
        for key in standard_keys:
            if key in header:
                info[key] = str(header[key]).strip()
        
        # Special handling for filename (often in comments)
        if 'XDAL0' in header:
            xdal0 = str(header['XDAL0'])
            parts = xdal0.split()
            if parts:
                info['FILENAME'] = parts[0]
        
        return info
    
    def _assess_data_quality(self, hdul, lightcurve_candidates: List[Dict]) -> tuple:
        """Assess data quality and recommend best column"""
        
        if not lightcurve_candidates:
            return 0.0, None
        
        best_score = 0.0
        best_column = None
        
        for candidate in lightcurve_candidates:
            hdu = hdul[candidate['hdu_index']]
            rate_column = candidate['rate_column']
            
            try:
                data = hdu.data[rate_column]
                
                # Calculate quality metrics
                n_points = len(data)
                n_finite = np.sum(np.isfinite(data))
                n_positive = np.sum(data > 0)
                
                # Quality score based on data characteristics
                completeness = n_finite / n_points if n_points > 0 else 0
                positivity = n_positive / n_finite if n_finite > 0 else 0
                size_factor = min(n_points / 1000, 1.0)  # Prefer larger datasets
                
                score = (completeness * 0.4 + positivity * 0.3 + size_factor * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_column = rate_column
                    
            except Exception:
                continue
        
        return best_score, best_column
    
    def _get_secure_extension(self, filename: str) -> str:
        """Get secure file extension"""
        ext = Path(filename).suffix.lower()
        return ext if ext in self.allowed_extensions else '.fits'
    
    def _update_stats(self):
        """Update internal statistics"""
        files = list(self.uploaded_files.values())
        
        self.stats.total_files = len(files)
        self.stats.valid_files = sum(1 for f in files if f.is_valid)
        self.stats.invalid_files = self.stats.total_files - self.stats.valid_files
        self.stats.total_size_mb = sum(f.file_size for f in files) / (1024 * 1024)
        self.stats.lightcurve_ready_files = sum(1 for f in files if f.lightcurve_ready)
        
        if self.stats.valid_files > 0:
            valid_files = [f for f in files if f.is_valid]
            self.stats.avg_quality_score = sum(f.data_quality_score for f in valid_files) / len(valid_files)
    
    # ========================================
    # Public Interface Methods
    # ========================================
    
    def get_file_info(self, file_id: str) -> Optional[FITSFileInfo]:
        """Get complete file information"""
        return self.uploaded_files.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file system path"""
        file_info = self.get_file_info(file_id)
        return file_info.file_path if file_info else None
    
    def list_files(self, 
                   valid_only: bool = True,
                   lightcurve_ready_only: bool = False) -> List[FITSFileInfo]:
        """List uploaded files with filtering"""
        
        files = list(self.uploaded_files.values())
        
        if valid_only:
            files = [f for f in files if f.is_valid]
        
        if lightcurve_ready_only:
            files = [f for f in files if f.lightcurve_ready]
        
        # Sort by upload time (newest first)
        files.sort(key=lambda f: f.upload_time, reverse=True)
        
        return files
    
    def delete_file(self, file_id: str) -> bool:
        """Delete file and cleanup"""
        file_info = self.get_file_info(file_id)
        if not file_info:
            return False
        
        try:
            # Remove file from disk
            file_path = Path(file_info.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Remove from tracking
            del self.uploaded_files[file_id]
            self._update_stats()
            
            logger.info(f"File deleted: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {str(e)}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up old files"""
        current_time = datetime.now()
        deleted_count = 0
        
        for file_id, file_info in list(self.uploaded_files.items()):
            age_hours = (current_time - file_info.upload_time).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                if self.delete_file(file_id):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old files")
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "files": {
                "total": self.stats.total_files,
                "valid": self.stats.valid_files,
                "invalid": self.stats.invalid_files,
                "lightcurve_ready": self.stats.lightcurve_ready_files
            },
            "storage": {
                "total_size_mb": round(self.stats.total_size_mb, 2),
                "upload_dir": str(self.upload_dir),
                "max_file_size_mb": self.max_file_size / (1024 * 1024)
            },
            "quality": {
                "avg_quality_score": round(self.stats.avg_quality_score, 3),
                "success_rate": (self.stats.valid_files / max(self.stats.total_files, 1))
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """System health check"""
        
        health = {
            "status": "healthy",
            "upload_dir_exists": self.upload_dir.exists(),
            "upload_dir_writable": os.access(self.upload_dir, os.W_OK),
            "active_uploads": len(self.upload_locks),
            "total_files": len(self.uploaded_files)
        }
        
        # Check disk space (if available)
        try:
            stat = os.statvfs(self.upload_dir)
            free_bytes = stat.f_frsize * stat.f_bavail
            health["free_space_gb"] = free_bytes / (1024**3)
        except:
            health["free_space_gb"] = "unknown"
        
        # Overall health status
        if not health["upload_dir_exists"] or not health["upload_dir_writable"]:
            health["status"] = "unhealthy"
        
        return health


# ========================================
# Factory Function
# ========================================

def create_file_manager(**kwargs) -> FITSFileManager:
    """Factory function to create file manager"""
    return FITSFileManager(**kwargs)


# ========================================
# Testing and Demo Functions
# ========================================

async def test_file_manager():
    """Interactive test of file manager"""
    print("🧪 FITS File Manager Test")
    print("=" * 50)
    
    # Create file manager
    manager = create_file_manager()
    
    # Get FITS file path from user
    while True:
        fits_path = input("\n📁 Enter path to your FITS file (or 'quit'): ").strip()
        
        if fits_path.lower() == 'quit':
            break
        
        if not fits_path or not os.path.exists(fits_path):
            print("❌ File not found. Please provide a valid path.")
            continue
        
        try:
            print(f"\n🚀 Uploading: {fits_path}")
            start_time = datetime.now()
            
            # Upload file
            file_info = await manager.upload_from_path(fits_path)
            
            upload_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            print(f"\n✅ Upload completed in {upload_time:.2f}s")
            print(f"📋 File Information:")
            print(f"   File ID: {file_info.file_id}")
            print(f"   Original: {file_info.original_filename}")
            print(f"   Size: {file_info.file_size / 1024:.1f} KB")
            print(f"   Valid: {'✅ YES' if file_info.is_valid else '❌ NO'}")
            
            if file_info.validation_error:
                print(f"   Error: {file_info.validation_error}")
                continue
            
            print(f"   HDUs: {file_info.n_hdus}")
            print(f"   Has Data: {'✅ YES' if file_info.has_data else '❌ NO'}")
            print(f"   Lightcurve Ready: {'✅ YES' if file_info.lightcurve_ready else '❌ NO'}")
            print(f"   Quality Score: {file_info.data_quality_score:.2f}")
            
            if file_info.recommended_column:
                print(f"   Recommended Column: {file_info.recommended_column}")
            
            if file_info.data_shapes:
                print(f"   Data Shapes: {file_info.data_shapes}")
            
            if file_info.column_info:
                print(f"   Columns: {file_info.column_info}")
            
            if file_info.header_info:
                print(f"   Header Info: {file_info.header_info}")
            
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
    
    # Show final statistics
    print(f"\n📊 Final Statistics:")
    stats = manager.get_stats()
    for category, data in stats.items():
        print(f"\n{category.upper()}:")
        for key, value in data.items():
            print(f"   {key}: {value}")
    
    # Health check
    print(f"\n🏥 Health Check:")
    health = manager.health_check()
    for key, value in health.items():
        status = "✅" if (isinstance(value, bool) and value) or value == "healthy" else "❌" if isinstance(value, bool) else "ℹ️"
        print(f"   {key}: {status} {value}")
    
    return manager


if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("🗂️ FITS File Upload & Management System")
        print("=" * 60)
        
        try:
            # Run interactive test
            manager = await test_file_manager()
            
            print(f"\n🎉 File manager test completed!")
            print(f"📁 {len(manager.uploaded_files)} files uploaded")
            
        except KeyboardInterrupt:
            print("\n👋 Test interrupted")
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())