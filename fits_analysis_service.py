#!/usr/bin/env python3
"""
fits_analysis_service.py
========================
Real FITS Analysis Service - Complete Implementation
Integrates with File Manager and uses actual functions from apitools.txt
"""

import os
import uuid
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import FITS analysis tools
from fits_analysis_tools import (
    # FITS loading
    load_fits_data, get_fits_header,
    # Analysis functions
    calculate_statistics, compute_psd, bin_psd,
    fit_power_law, fit_bending_power_law,
    # Model functions
    power_law_fn, bending_power_law_fn,
    # Plotting functions
    plot_psd_figure, plot_power_law_with_residual_figure,
    plot_bending_power_law_with_residual_figure
)

from file_manager import FITSFileManager, FITSFileInfo, create_file_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Comprehensive analysis result container"""
    # Basic info
    analysis_type: str
    success: bool
    processing_time: float
    
    # Results and parameters
    results: Dict[str, Any] = field(default_factory=dict)
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    
    # Files and plots
    plot_url: Optional[str] = None
    plot_path: Optional[str] = None
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    fits_info: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    quality_score: float = 0.0
    reliability: str = "unknown"  # "high", "medium", "low", "unknown"


@dataclass
class AnalysisServiceStats:
    """Analysis service statistics"""
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    
    # Analysis type breakdown
    statistics_count: int = 0
    psd_count: int = 0
    power_law_count: int = 0
    bending_power_law_count: int = 0
    
    # Quality metrics
    avg_quality_score: float = 0.0
    plots_generated: int = 0


class FITSAnalysisService:
    """
    🔬 Real FITS Analysis Service
    
    Complete analysis service using actual functions from apitools.txt:
    - Statistical analysis of lightcurve data
    - Power Spectral Density computation with customizable parameters
    - Power law fitting with two-step optimization
    - Bending power law fitting with break frequency detection
    - High-quality plot generation with residuals
    - Comprehensive error handling and quality assessment
    """
    
    def __init__(self, 
                 file_manager: Optional[FITSFileManager] = None,
                 plots_dir: str = "storage/plots"):
        
        self.file_manager = file_manager or create_file_manager()
        self.plots_dir = Path(plots_dir)
        
        # Create plot subdirectories
        self.plot_subdirs = {
            "psd": self.plots_dir / "psd",
            "power_law": self.plots_dir / "power_law", 
            "bending_power_law": self.plots_dir / "bending_power_law"
        }
        
        # Ensure all directories exist
        for subdir in self.plot_subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        # Analysis statistics
        self.stats = AnalysisServiceStats()
        
        # Analysis configuration
        self.default_configs = {
            "statistics": {
                "metrics": ["mean", "median", "std", "min", "max", "count"]
            },
            "psd": {
                "low_freq": 1e-5,
                "high_freq": 0.05,
                "bins": 3500
            },
            "fitting_power_law": {
                "low_freq": 1e-5,
                "high_freq": 0.05,
                "bins": 3500,
                "noise_bound_percent": 0.7,
                "A0": 1.0,
                "b0": 1.0,
                "A_min": 0.0,
                "A_max": 1e38,
                "b_min": 0.1,
                "b_max": 3.0,
                "maxfev": 1000000
            },
            "fitting_bending_power_law": {
                "low_freq": 1e-5,
                "high_freq": 0.05,
                "bins": 3500,
                "noise_bound_percent": 0.7,
                "A0": 10.0,
                "fb0": 0.01,  # Auto-detect
                "sh0": 1.0,
                "A_min": 0.0,
                "A_max": 1e38,
                "fb_min": 1e-5,
                "fb_max": 0.1,
                "sh_min": 0.3,
                "sh_max": 3.0,
                "maxfev": 1000000
            }
        }
        
        logger.info("FITS Analysis Service initialized")
        logger.info(f"Plots directory: {self.plots_dir}")
        logger.info(f"File manager: {type(self.file_manager).__name__}")
    
    # ========================================
    # Main Analysis Interface
    # ========================================
    
    async def execute_analysis(self, 
                             fits_id: str,
                             analysis_type: str,
                             parameters: Optional[Dict[str, Any]] = None,
                             column: str = "Rate") -> AnalysisResult:
        """
        Execute comprehensive analysis on FITS file
        
        Args:
            fits_id: File ID from file manager
            analysis_type: Type of analysis ("statistics", "psd", "fitting_power_law", "fitting_bending_power_law")
            parameters: Analysis parameters (uses defaults if None)
            column: Column name for table data extraction
            
        Returns:
            AnalysisResult with complete results and metadata
        """
        start_time = datetime.now()
        self.stats.total_analyses += 1
        
        # Validate analysis type
        valid_types = ["statistics", "psd", "fitting_power_law", "fitting_bending_power_law"]
        if analysis_type not in valid_types:
            return self._create_error_result(
                analysis_type, f"Invalid analysis type. Must be one of: {valid_types}", 0.0
            )
        
        # Update type-specific statistics
        if analysis_type == "statistics":
            self.stats.statistics_count += 1
        elif analysis_type == "psd":
            self.stats.psd_count += 1
        elif analysis_type == "fitting_power_law":
            self.stats.power_law_count += 1
        elif analysis_type == "fitting_bending_power_law":
            self.stats.bending_power_law_count += 1
        
        try:
            # Get file information
            file_info = self.file_manager.get_file_info(fits_id)
            if not file_info:
                raise ValueError(f"FITS file not found: {fits_id}")
            
            if not file_info.is_valid:
                raise ValueError(f"FITS file is invalid: {file_info.validation_error}")
            
            file_path = file_info.file_path
            
            # Merge parameters with defaults
            final_params = self._prepare_parameters(analysis_type, parameters)
            
            # Route to appropriate analysis method
            if analysis_type == "statistics":
                result = await self._execute_statistics_analysis(file_path, final_params, column)
            elif analysis_type == "psd":
                result = await self._execute_psd_analysis(file_path, final_params, column)
            elif analysis_type == "fitting_power_law":
                result = await self._execute_power_law_analysis(file_path, final_params, column)
            elif analysis_type == "fitting_bending_power_law":
                result = await self._execute_bending_power_law_analysis(file_path, final_params, column)
            
            # Add timing and metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            # Enhance metadata
            result.metadata.update({
                "fits_id": fits_id,
                "analysis_time": start_time.isoformat(),
                "column_used": column,
                "service_version": "1.0",
                "parameter_source": "user_provided" if parameters else "defaults"
            })
            
            # Add FITS file information
            result.fits_info = {
                "original_filename": file_info.original_filename,
                "file_size": file_info.file_size,
                "data_quality_score": file_info.data_quality_score,
                "lightcurve_ready": file_info.lightcurve_ready,
                "recommended_column": file_info.recommended_column
            }
            
            # Assess result quality
            result.quality_score = self._assess_result_quality(result, file_info)
            result.reliability = self._determine_reliability(result.quality_score)
            
            # Update statistics
            if result.success:
                self.stats.successful_analyses += 1
                if result.plot_url:
                    self.stats.plots_generated += 1
            else:
                self.stats.failed_analyses += 1
            
            self.stats.total_processing_time += processing_time
            self.stats.avg_processing_time = (
                self.stats.total_processing_time / self.stats.total_analyses
            )
            
            # Update average quality score
            if result.success and result.quality_score > 0:
                current_avg = self.stats.avg_quality_score
                total_success = self.stats.successful_analyses
                self.stats.avg_quality_score = (
                    (current_avg * (total_success - 1) + result.quality_score) / total_success
                )
            
            logger.info(f"Analysis {analysis_type} completed in {processing_time:.3f}s")
            logger.info(f"Success: {result.success}, Quality: {result.quality_score:.2f}")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats.failed_analyses += 1
            self.stats.total_processing_time += processing_time
            
            logger.error(f"Analysis {analysis_type} failed: {str(e)}")
            
            return self._create_error_result(analysis_type, str(e), processing_time)
    
    # ========================================
    # Individual Analysis Methods
    # ========================================
    
    async def _execute_statistics_analysis(self, 
                                          file_path: str, 
                                          params: Dict[str, Any],
                                          column: str) -> AnalysisResult:
        """Execute statistical analysis using real functions"""
        
        try:
            # Load FITS data
            rate = load_fits_data(file_path, column=column)
            header_info = get_fits_header(file_path)
            filename = header_info.get("filename", os.path.basename(file_path))
            
            # Get metrics to calculate
            metrics = params.get("metrics", self.default_configs["statistics"]["metrics"])
            
            # Calculate real statistics
            stats = calculate_statistics(rate, metrics)
            
            # Additional data quality metrics
            n_points = len(rate)
            n_finite = np.sum(np.isfinite(rate))
            n_positive = np.sum(rate > 0)
            data_range = [float(np.min(rate)), float(np.max(rate))]
            
            # Create result
            result = AnalysisResult(
                analysis_type="statistics",
                success=True,
                processing_time=0.0,  # Will be set later
                results={
                    "statistics": stats,
                    "data_quality": {
                        "n_points": n_points,
                        "n_finite": n_finite,
                        "n_positive": n_positive,
                        "completeness": n_finite / n_points if n_points > 0 else 0.0,
                        "positivity": n_positive / n_finite if n_finite > 0 else 0.0
                    },
                    "data_range": data_range,
                    "file_name": filename
                },
                parameters_used=params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Statistics analysis failed: {str(e)}")
            raise
    
    async def _execute_psd_analysis(self, 
                                   file_path: str, 
                                   params: Dict[str, Any],
                                   column: str) -> AnalysisResult:
        """Execute PSD computation using real functions"""
        
        try:
            # Load FITS data
            rate = load_fits_data(file_path, column=column)
            header_info = get_fits_header(file_path)
            filename = header_info.get("filename", os.path.basename(file_path))
            
            # Get PSD parameters
            low_freq = params.get("low_freq", 1e-5)
            high_freq = params.get("high_freq", 0.05)
            bins = params.get("bins", 3500)
            
            # Compute real PSD
            freqs, psd = compute_psd(rate)
            
            # Validate frequency bounds
            min_freq = float(np.min(freqs))
            max_freq = float(np.max(freqs))
            validated_low_freq = max(low_freq, min_freq)
            validated_high_freq = min(high_freq, max_freq)
            
            # Bin PSD
            x, y = bin_psd(freqs, psd, validated_low_freq, validated_high_freq, bins)
            
            # Generate plot
            fig = plot_psd_figure(x, y, title=f"Power Spectral Density - [{filename}]")
            
            # Save plot
            plot_id = str(uuid.uuid4())
            plot_path = self.plot_subdirs["psd"] / f"psd_{plot_id}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Create result
            result = AnalysisResult(
                analysis_type="psd",
                success=True,
                processing_time=0.0,
                results={
                    "frequencies": x.tolist(),
                    "psd_values": y.tolist(),
                    "file_name": filename,
                    "n_points": len(x),
                    "freq_range": {
                        "requested": [low_freq, high_freq],
                        "actual": [float(x[0]), float(x[-1])],
                        "data_limits": [min_freq, max_freq]
                    },
                    "bins": {
                        "requested": bins,
                        "actual": len(x)
                    }
                },
                parameters_used=params,
                plot_url=f"/storage/plots/psd/psd_{plot_id}.png",
                plot_path=str(plot_path)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"PSD analysis failed: {str(e)}")
            raise
    
    async def _execute_power_law_analysis(self, 
                                         file_path: str, 
                                         params: Dict[str, Any],
                                         column: str) -> AnalysisResult:
        """Execute power law fitting using real functions"""
        
        try:
            # Load data and compute PSD first
            rate = load_fits_data(file_path, column=column)
            header_info = get_fits_header(file_path)
            filename = header_info.get("filename", os.path.basename(file_path))
            
            # Get parameters with defaults
            low_freq = params.get("low_freq", 1e-5)
            high_freq = params.get("high_freq", 0.05)
            bins = params.get("bins", 3500)
            noise_bound_percent = params.get("noise_bound_percent", 0.7)
            
            # Initial parameters
            initial_params = {
                'A': params.get('A0', 1.0),
                'b': params.get('b0', 1.0)
            }
            
            # Parameter bounds
            param_bounds = {
                'A': (params.get('A_min', 0.0), params.get('A_max', 1e38)),
                'b': (params.get('b_min', 0.1), params.get('b_max', 3.0))
            }
            
            # Compute PSD
            freqs, psd = compute_psd(rate)
            
            # Validate frequency bounds  
            min_freq = float(np.min(freqs))
            max_freq = float(np.max(freqs))
            validated_low_freq = max(low_freq, min_freq)
            validated_high_freq = min(high_freq, max_freq)
            
            # Bin PSD
            x, y = bin_psd(freqs, psd, validated_low_freq, validated_high_freq, bins)
            
            # Fit power law using real function
            A, b, n = fit_power_law(
                x, y,
                noise_bound_percent=noise_bound_percent,
                initial_params=initial_params,
                param_bounds=param_bounds,
                maxfev=params.get('maxfev', 1000000)
            )
            
            # Calculate goodness of fit
            y_fit = power_law_fn(x, A, b, n)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Calculate reduced chi-squared
            chi_squared = np.sum(((y - y_fit) / np.sqrt(np.abs(y_fit))) ** 2)
            reduced_chi_squared = chi_squared / (len(x) - 3)  # 3 parameters
            
            # Generate plot
            fig = plot_power_law_with_residual_figure(
                x, y, A, b, n, title=f"Power Law Fit - [{filename}]"
            )
            
            # Save plot
            plot_id = str(uuid.uuid4())
            plot_path = self.plot_subdirs["power_law"] / f"power_law_{plot_id}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Create result
            result = AnalysisResult(
                analysis_type="fitting_power_law",
                success=True,
                processing_time=0.0,
                results={
                    "fitted_parameters": {
                        "A": float(A), 
                        "b": float(b), 
                        "n": float(n)
                    },
                    "goodness_of_fit": {
                        "r_squared": float(r_squared),
                        "chi_squared": float(chi_squared),
                        "reduced_chi_squared": float(reduced_chi_squared)
                    },
                    "file_name": filename,
                    "fit_info": {
                        "low_freq": float(validated_low_freq),
                        "high_freq": float(validated_high_freq),
                        "bins": int(bins),
                        "noise_bound_percent": float(noise_bound_percent),
                        "n_data_points": len(x)
                    },
                    "initial_parameters": initial_params,
                    "parameter_bounds": {
                        k: [float(v[0]), "unbounded" if np.isinf(v[1]) else float(v[1])]
                        for k, v in param_bounds.items()
                    }
                },
                parameters_used=params,
                plot_url=f"/storage/plots/power_law/power_law_{plot_id}.png",
                plot_path=str(plot_path)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Power law fitting failed: {str(e)}")
            raise
    
    async def _execute_bending_power_law_analysis(self, 
                                                 file_path: str, 
                                                 params: Dict[str, Any],
                                                 column: str) -> AnalysisResult:
        """Execute bending power law fitting using real functions"""
        
        try:
            # Load data and compute PSD
            rate = load_fits_data(file_path, column=column)
            header_info = get_fits_header(file_path)
            filename = header_info.get("filename", os.path.basename(file_path))
            
            # Parameters
            low_freq = params.get("low_freq", 1e-5)
            high_freq = params.get("high_freq", 0.05)
            bins = params.get("bins", 3500)
            noise_bound_percent = params.get("noise_bound_percent", 0.7)
            
            # Initial parameters for bending power law
            initial_params = {
                'A': params.get('A0', 10.0),
                'sh': params.get('sh0', 1.0)
            }
            if params.get('fb0'):
                initial_params['fb'] = params['fb0']
            
            # Parameter bounds
            param_bounds = {
                'A': (params.get('A_min', 0.0), params.get('A_max', 1e38)),
                'sh': (params.get('sh_min', 0.3), params.get('sh_max', 3.0))
            }
            if params.get('fb_min') or params.get('fb_max'):
                param_bounds['fb'] = (
                    params.get('fb_min', low_freq),
                    params.get('fb_max', high_freq)
                )
            
            # Compute PSD and fit
            freqs, psd = compute_psd(rate)
            min_freq = float(np.min(freqs))
            max_freq = float(np.max(freqs))
            validated_low_freq = max(low_freq, min_freq)
            validated_high_freq = min(high_freq, max_freq)
            x, y = bin_psd(freqs, psd, validated_low_freq, validated_high_freq, bins)
            
            # Fit bending power law
            A, fb, sh, n = fit_bending_power_law(
                x, y,
                noise_bound_percent=noise_bound_percent,
                initial_params=initial_params,
                param_bounds=param_bounds,
                maxfev=params.get('maxfev', 1000000)
            )
            
            # Calculate goodness of fit
            y_fit = bending_power_law_fn(x, A, fb, sh, n)
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Calculate reduced chi-squared
            chi_squared = np.sum(((y - y_fit) / np.sqrt(np.abs(y_fit))) ** 2)
            reduced_chi_squared = chi_squared / (len(x) - 4)  # 4 parameters
            
            # Generate plot
            fig = plot_bending_power_law_with_residual_figure(
                x, y, A, fb, sh, n, title=f"Bending Power Law Fit - [{filename}]"
            )
            
            # Save plot
            plot_id = str(uuid.uuid4())
            plot_path = self.plot_subdirs["bending_power_law"] / f"bending_power_law_{plot_id}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Create result
            result = AnalysisResult(
                analysis_type="fitting_bending_power_law",
                success=True,
                processing_time=0.0,
                results={
                    "fitted_parameters": {
                        "A": float(A),
                        "fb": float(fb), 
                        "sh": float(sh),
                        "n": float(n)
                    },
                    "goodness_of_fit": {
                        "r_squared": float(r_squared),
                        "chi_squared": float(chi_squared),
                        "reduced_chi_squared": float(reduced_chi_squared)
                    },
                    "file_name": filename,
                    "fit_info": {
                        "low_freq": float(validated_low_freq),
                        "high_freq": float(validated_high_freq),
                        "bins": int(bins),
                        "noise_bound_percent": float(noise_bound_percent),
                        "n_data_points": len(x)
                    },
                    "initial_parameters": initial_params,
                    "parameter_bounds": {
                        k: [float(v[0]), "unbounded" if np.isinf(v[1]) else float(v[1])]
                        for k, v in param_bounds.items()
                    }
                },
                parameters_used=params,
                plot_url=f"/storage/plots/bending_power_law/bending_power_law_{plot_id}.png",
                plot_path=str(plot_path)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Bending power law fitting failed: {str(e)}")
            raise
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _prepare_parameters(self, analysis_type: str, user_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge user parameters with defaults"""
        defaults = self.default_configs.get(analysis_type, {}).copy()
        
        if user_params:
            # Merge user parameters, validating types
            for key, value in user_params.items():
                if key in defaults:
                    # Type validation
                    expected_type = type(defaults[key])
                    if not isinstance(value, expected_type):
                        try:
                            # Try to convert
                            defaults[key] = expected_type(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid parameter {key}: {value}. Using default: {defaults[key]}")
                    else:
                        defaults[key] = value
                else:
                    # Add new parameter
                    defaults[key] = value
        
        return defaults
    
    def _assess_result_quality(self, result: AnalysisResult, file_info: FITSFileInfo) -> float:
        """Assess the quality of analysis results"""
        
        quality_score = 0.0
        
        # Base score from file quality
        quality_score += file_info.data_quality_score * 0.3
        
        # Analysis-specific quality assessment
        if result.analysis_type == "statistics":
            # Statistics quality based on data completeness
            data_quality = result.results.get("data_quality", {})
            completeness = data_quality.get("completeness", 0.0)
            positivity = data_quality.get("positivity", 0.0)
            quality_score += (completeness * 0.4 + positivity * 0.3)
            
        elif result.analysis_type == "psd":
            # PSD quality based on frequency coverage and data points
            n_points = result.results.get("n_points", 0)
            freq_range = result.results.get("freq_range", {})
            actual_range = freq_range.get("actual", [0, 0])
            coverage = (actual_range[1] - actual_range[0]) / actual_range[1] if actual_range[1] > 0 else 0
            points_factor = min(n_points / 1000, 1.0)  # Prefer more data points
            quality_score += (coverage * 0.4 + points_factor * 0.3)
            
        elif result.analysis_type.startswith("fitting_"):
            # Fitting quality based on goodness of fit
            goodness = result.results.get("goodness_of_fit", {})
            r_squared = goodness.get("r_squared", 0.0)
            reduced_chi_squared = goodness.get("reduced_chi_squared", float('inf'))
            
            # R-squared contribution (higher is better)
            r_squared_score = max(0, min(1, r_squared)) * 0.4
            
            # Chi-squared contribution (closer to 1 is better)
            if reduced_chi_squared < float('inf'):
                chi_score = max(0, 1 - abs(reduced_chi_squared - 1)) * 0.3
            else:
                chi_score = 0.0
            
            quality_score += r_squared_score + chi_score
        
        return min(1.0, max(0.0, quality_score))  # Clamp to [0, 1]
    
    def _determine_reliability(self, quality_score: float) -> str:
        """Determine reliability level based on quality score"""
        if quality_score >= 0.8:
            return "high"
        elif quality_score >= 0.6:
            return "medium"
        elif quality_score >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def _create_error_result(self, analysis_type: str, error_msg: str, processing_time: float) -> AnalysisResult:
        """Create error result"""
        return AnalysisResult(
            analysis_type=analysis_type,
            success=False,
            processing_time=processing_time,
            error=error_msg,
            results={},
            parameters_used={},
            quality_score=0.0,
            reliability="unknown"
        )
    
    # ========================================
    # Public Interface Methods
    # ========================================
    
    def get_default_parameters(self, analysis_type: str) -> Dict[str, Any]:
        """Get default parameters for analysis type"""
        return self.default_configs.get(analysis_type, {}).copy()
    
    def list_analysis_types(self) -> List[str]:
        """Get list of supported analysis types"""
        return list(self.default_configs.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        total = max(self.stats.total_analyses, 1)
        
        return {
            "performance": {
                "total_analyses": self.stats.total_analyses,
                "successful_analyses": self.stats.successful_analyses,
                "failed_analyses": self.stats.failed_analyses,
                "success_rate": self.stats.successful_analyses / total,
                "avg_processing_time": round(self.stats.avg_processing_time, 3),
                "total_processing_time": round(self.stats.total_processing_time, 2)
            },
            "analysis_breakdown": {
                "statistics": self.stats.statistics_count,
                "psd": self.stats.psd_count,
                "power_law": self.stats.power_law_count,
                "bending_power_law": self.stats.bending_power_law_count
            },
            "quality": {
                "avg_quality_score": round(self.stats.avg_quality_score, 3),
                "plots_generated": self.stats.plots_generated
            },
            "storage": {
                "plots_directory": str(self.plots_dir),
                "plot_subdirectories": {k: str(v) for k, v in self.plot_subdirs.items()}
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Service health check"""
        
        health = {
            "status": "healthy",
            "plots_dir_exists": self.plots_dir.exists(),
            "plots_dir_writable": os.access(self.plots_dir, os.W_OK),
            "file_manager_status": "connected" if self.file_manager else "disconnected",
            "analysis_types_available": len(self.default_configs)
        }
        
        # Check plot subdirectories
        for name, path in self.plot_subdirs.items():
            health[f"{name}_dir_ready"] = path.exists() and os.access(path, os.W_OK)
        
        # Overall status
        if not all([
            health["plots_dir_exists"],
            health["plots_dir_writable"],
            health["file_manager_status"] == "connected"
        ]):
            health["status"] = "unhealthy"
        
        return health
    
    async def cleanup_old_plots(self, max_age_hours: int = 24) -> int:
        """Clean up old plot files"""
        current_time = datetime.now()
        deleted_count = 0
        
        for subdir in self.plot_subdirs.values():
            if not subdir.exists():
                continue
                
            for plot_file in subdir.glob("*.png"):
                try:
                    file_age = current_time - datetime.fromtimestamp(plot_file.stat().st_mtime)
                    if file_age.total_seconds() / 3600 > max_age_hours:
                        plot_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old plot {plot_file}: {str(e)}")
        
        logger.info(f"Cleaned up {deleted_count} old plot files")
        return deleted_count


# ========================================
# Factory Function
# ========================================

def create_analysis_service(file_manager: Optional[FITSFileManager] = None) -> FITSAnalysisService:
    """Factory function to create analysis service"""
    return FITSAnalysisService(file_manager=file_manager)


# ========================================
# Testing and Demo Functions  
# ========================================

async def test_analysis_service():
    """Comprehensive test of the analysis service"""
    print("🔬 FITS Analysis Service Test")
    print("=" * 60)
    
    # Create services
    file_manager = create_file_manager()
    analysis_service = create_analysis_service(file_manager)
    
    # Check for uploaded files
    files = file_manager.list_files()
    if not files:
        print("❌ No FITS files available.")
        print("Please upload a FITS file first using the file manager.")
        return False
    
    print(f"📁 Found {len(files)} FITS files:")
    for i, file_info in enumerate(files):
        status = "✅" if file_info.lightcurve_ready else "⚠️"
        print(f"   {i+1}. {status} {file_info.original_filename} (ID: {file_info.file_id[:8]}...)")
        if file_info.recommended_column:
            print(f"      Recommended column: {file_info.recommended_column}")
    
    # Select file for testing
    if len(files) == 1:
        test_file = files[0]
        print(f"\n🧪 Testing with: {test_file.original_filename}")
    else:
        while True:
            try:
                choice = input(f"\nSelect file (1-{len(files)}): ").strip()
                if choice.lower() == 'quit':
                    return False
                file_index = int(choice) - 1
                if 0 <= file_index < len(files):
                    test_file = files[file_index]
                    break
                print(f"Invalid choice. Please enter 1-{len(files)}")
            except KeyboardInterrupt:
                return False
            except ValueError:
                print("Please enter a valid number")
    
    # Determine column to use
    column = test_file.recommended_column or "Rate"
    print(f"Using column: {column}")
    
    # Test different analysis types
    test_cases = [
        {
            "name": "Statistical Analysis",
            "type": "statistics",
            "params": {"metrics": ["mean", "std", "median", "min", "max", "count"]},
            "expected_keys": ["statistics", "data_quality"]
        },
        {
            "name": "Power Spectral Density",
            "type": "psd", 
            "params": {"low_freq": 1e-5, "high_freq": 0.05, "bins": 2000},
            "expected_keys": ["frequencies", "psd_values", "freq_range"]
        },
        {
            "name": "Power Law Fitting",
            "type": "fitting_power_law",
            "params": {
                "low_freq": 1e-4, "high_freq": 0.01, "bins": 1500,
                "A0": 2.0, "b0": 1.5, "noise_bound_percent": 0.8
            },
            "expected_keys": ["fitted_parameters", "goodness_of_fit"]
        },
        {
            "name": "Bending Power Law Fitting", 
            "type": "fitting_bending_power_law",
            "params": {
                "low_freq": 1e-4, "high_freq": 0.01, "bins": 1500,
                "A0": 15.0, "sh0": 1.2, "noise_bound_percent": 0.8
            },
            "expected_keys": ["fitted_parameters", "goodness_of_fit"]
        }
    ]
    
    successful_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Type: {test_case['type']}")
        print(f"Parameters: {test_case['params']}")
        print(f"{'='*70}")
        
        try:
            start_time = datetime.now()
            
            # Execute analysis
            result = await analysis_service.execute_analysis(
                fits_id=test_file.file_id,
                analysis_type=test_case['type'],
                parameters=test_case['params'],
                column=column
            )
            
            test_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            print(f"✅ Success: {result.success}")
            print(f"⏱️  Processing Time: {result.processing_time:.3f}s")
            print(f"🎯 Quality Score: {result.quality_score:.3f}")
            print(f"🔒 Reliability: {result.reliability}")
            
            if result.success:
                print(f"\n📊 Results:")
                
                # Check expected keys
                missing_keys = []
                for key in test_case['expected_keys']:
                    if key not in result.results:
                        missing_keys.append(key)
                    else:
                        value = result.results[key]
                        if isinstance(value, dict):
                            print(f"   {key}: {list(value.keys())}")
                        elif isinstance(value, list):
                            print(f"   {key}: [{len(value)} items]")
                        else:
                            print(f"   {key}: {value}")
                
                if missing_keys:
                    print(f"   ⚠️  Missing keys: {missing_keys}")
                else:
                    print(f"   ✅ All expected keys present")
                
                # Plot information
                if result.plot_url:
                    print(f"📈 Plot: {result.plot_url}")
                    print(f"📁 File: {result.plot_path}")
                
                # Analysis-specific details
                if test_case['type'] == "statistics":
                    stats = result.results.get('statistics', {})
                    print(f"   Key Statistics: mean={stats.get('mean', 'N/A'):.3f}, "
                          f"std={stats.get('std', 'N/A'):.3f}")
                
                elif test_case['type'] == "psd":
                    freq_range = result.results.get('freq_range', {})
                    actual = freq_range.get('actual', [0, 0])
                    print(f"   Frequency Range: {actual[0]:.2e} - {actual[1]:.2e} Hz")
                    print(f"   Data Points: {result.results.get('n_points', 'N/A')}")
                
                elif test_case['type'].startswith('fitting_'):
                    params = result.results.get('fitted_parameters', {})
                    goodness = result.results.get('goodness_of_fit', {})
                    print(f"   Fitted Parameters: {params}")
                    print(f"   R-squared: {goodness.get('r_squared', 'N/A'):.4f}")
                    print(f"   Reduced χ²: {goodness.get('reduced_chi_squared', 'N/A'):.4f}")
                
                successful_tests += 1
                
            else:
                print(f"❌ Analysis failed: {result.error}")
                if result.warnings:
                    print(f"⚠️  Warnings: {result.warnings}")
        
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Final summary
    print(f"\n{'='*70}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful Tests: {successful_tests}/{len(test_cases)}")
    print(f"📊 Success Rate: {successful_tests/len(test_cases)*100:.1f}%")
    
    # Service statistics
    stats = analysis_service.get_stats()
    print(f"\n📈 Service Statistics:")
    for category, data in stats.items():
        print(f"\n{category.upper()}:")
        for key, value in data.items():
            print(f"   {key}: {value}")
    
    # Health check
    health = analysis_service.health_check()
    print(f"\n🏥 Health Check:")
    for key, value in health.items():
        status = "✅" if (isinstance(value, bool) and value) or value == "healthy" or value == "connected" else "❌" if isinstance(value, bool) else "ℹ️"
        print(f"   {key}: {status} {value}")
    
    if successful_tests == len(test_cases):
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ FITS Analysis Service is fully functional!")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Check the errors above.")
        return False


async def demo_analysis_workflow():
    """Demo complete analysis workflow"""
    print("🚀 FITS Analysis Workflow Demo")
    print("=" * 60)
    
    # Step 1: File upload
    print("Step 1: Upload FITS file")
    file_manager = create_file_manager()
    
    fits_path = input("Enter path to FITS file: ").strip()
    if not fits_path or not os.path.exists(fits_path):
        print("❌ File not found")
        return
    
    try:
        file_info = await file_manager.upload_from_path(fits_path)
        print(f"✅ Uploaded: {file_info.original_filename}")
        print(f"   Valid: {file_info.is_valid}")
        print(f"   Lightcurve ready: {file_info.lightcurve_ready}")
        print(f"   Recommended column: {file_info.recommended_column}")
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return
    
    # Step 2: Create analysis service
    print(f"\nStep 2: Initialize analysis service")
    analysis_service = create_analysis_service(file_manager)
    print(f"✅ Analysis service ready")
    
    # Step 3: Run complete analysis pipeline
    print(f"\nStep 3: Execute analysis pipeline")
    
    column = file_info.recommended_column or "Rate"
    analyses = ["statistics", "psd", "fitting_power_law", "fitting_bending_power_law"]
    results = {}
    
    for analysis_type in analyses:
        print(f"\n🔬 Running {analysis_type}...")
        try:
            result = await analysis_service.execute_analysis(
                fits_id=file_info.file_id,
                analysis_type=analysis_type,
                column=column
            )
            
            results[analysis_type] = result
            
            if result.success:
                print(f"   ✅ Success (Quality: {result.quality_score:.2f})")
                if result.plot_url:
                    print(f"   📈 Plot: {result.plot_url}")
            else:
                print(f"   ❌ Failed: {result.error}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Step 4: Summary report
    print(f"\n{'='*60}")
    print("📋 ANALYSIS SUMMARY REPORT")
    print(f"{'='*60}")
    
    successful_analyses = [k for k, v in results.items() if v.success]
    print(f"File: {file_info.original_filename}")
    print(f"Column used: {column}")
    print(f"Successful analyses: {len(successful_analyses)}/{len(analyses)}")
    print(f"Overall quality: {np.mean([r.quality_score for r in results.values() if r.success]):.2f}")
    
    for analysis_type, result in results.items():
        if result.success:
            print(f"\n{analysis_type.upper()}:")
            print(f"   Quality: {result.quality_score:.2f} ({result.reliability})")
            print(f"   Time: {result.processing_time:.2f}s")
            
            if analysis_type == "statistics":
                stats = result.results.get('statistics', {})
                print(f"   Mean: {stats.get('mean', 'N/A'):.3f}")
                print(f"   Std: {stats.get('std', 'N/A'):.3f}")
                
            elif analysis_type.startswith('fitting_'):
                goodness = result.results.get('goodness_of_fit', {})
                print(f"   R²: {goodness.get('r_squared', 'N/A'):.4f}")
    
    print(f"\n🎉 Demo completed!")


if __name__ == "__main__":
    async def main():
        print("🔬 FITS Analysis Service")
        print("=" * 40)
        
        choice = input(
            "Choose option:\n"
            "1. Test analysis service\n"
            "2. Demo complete workflow\n"
            "3. Quick validation\n"
            "Enter choice (1-3): "
        ).strip()
        
        try:
            if choice == "1":
                await test_analysis_service()
            elif choice == "2":
                await demo_analysis_workflow()
            elif choice == "3":
                # Quick validation with file manager test
                print("\n🚀 Quick Validation")
                print("Running file manager test first...")
                
                # Import and run file manager test
                from file_manager import test_file_manager
                manager = await test_file_manager()
                
                if manager and manager.uploaded_files:
                    print("\n🔬 Testing analysis service...")
                    service = create_analysis_service(manager)
                    
                    files = manager.list_files()
                    test_file = files[0]
                    
                    # Quick statistics test
                    result = await service.execute_analysis(
                        fits_id=test_file.file_id,
                        analysis_type="statistics"
                    )
                    
                    print(f"✅ Analysis test: {'PASS' if result.success else 'FAIL'}")
                    if result.success:
                        print(f"   Quality: {result.quality_score:.2f}")
                        print(f"   Time: {result.processing_time:.3f}s")
                
            else:
                print("Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted")
        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(main())