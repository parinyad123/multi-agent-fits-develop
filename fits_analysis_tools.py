#!/usr/bin/env python3
"""
fits_analysis_tools.py
===========================
FITS Analysis Tools - Organized from apitools.txt
=================================================
Breaking down the tools from apitools.txt into organized modules
"""

# app/tools/fits/loader.py
import os
import logging
import numpy as np
from astropy.io import fits
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_fits_data(path: str, hdu_index: int = 1, column: str = 'Rate') -> np.ndarray:
    """
    Load data from a FITS file.
    
    Args:
        path: Path to the FITS file
        hdu_index: Index of the HDU to load data from (default: 1)
        column: Column name to extract (for table HDUs)
        
    Returns:
        NumPy array with the extracted data
    """
    try:
        with fits.open(path) as hdul:
            if hdu_index >= len(hdul):
                raise ValueError(f"HDU index: {hdu_index} out of range (file has {len(hdul)} HDUs)")
            
            hdu = hdul[hdu_index]
            
            if hdu.data is None:
                raise ValueError(f"HDU {hdu_index} has no data")
            
            # Extract data based on HDU type
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                # Case-insensitive column matching
                available_columns = hdu.columns.names
                
                # Try exact match first
                if column in available_columns:
                    data = hdu.data[column]
                else:
                    # Try case-insensitive match
                    column_lower = column.lower()
                    matching_columns = [col for col in available_columns if col.lower() == column_lower]
                    
                    if matching_columns:
                        data = hdu.data[matching_columns[0]]
                    else:
                        # No match found
                        raise ValueError(f"Column '{column}' not found in HDU {hdu_index}. Available columns: {', '.join(available_columns)}")
            else:
                # Image data
                data = hdu.data
            
            # Convert to numpy array and clean
            array_data = np.array(data)
            clean_data = np.nan_to_num(array_data, nan=0.0)
            
            # Ensure non-negative values for time series
            return np.where(clean_data < 0, 0, clean_data)

    except Exception as e:
        logger.error(f"Error loading FITS file {path}: {str(e)}")
        raise


def get_fits_header(path: str) -> dict:
    """
    Extract header information from a FITS file.
    
    Args:
        path: Path to the FITS file
        
    Returns:
        Dictionary with header information
    """
    try:
        with fits.open(path) as hdul:
            primary_header = hdul[0].header
            
            # Extract the XDAL0 value if it exists
            xdal0 = primary_header.get('XDAL0', '')
            
            # If XDAL0 exists, parse the filename
            filename = ''
            if xdal0:
                # XDAL0 format example: '0792180601_PN_source_lc_300_1000eV.fits 2024-04-20T15:08:06.000 Cre&'
                parts = xdal0.split()
                if parts:
                    filename = parts[0]  # Get the first part which should be the filename
            
            return {
                "filename": filename or os.path.basename(path),
                "xdal0": xdal0,
                "header": {k: str(v) for k, v in primary_header.items()}
            }
    except Exception as e:
        logger.error(f"Error extracting header from FITS file {path}: {str(e)}")
        return {
            "filename": os.path.basename(path), 
            "error": str(e)
        }


# ========================================
# app/tools/analysis/statistics.py
# ========================================

from typing import List, Dict, Any

def calculate_statistics(data: np.ndarray, metrics: List[str]) -> Dict[str, Any]:
    """
    Calculate statistical metrics from time series data.
    
    Args:
        data: Time series data array
        metrics: List of statistical metrics to calculate
        
    Returns: 
        Dictionary mapping metrics name to their calculated values
    """
    stats: Dict[str, Any] = {}
    
    # Calculate basic statistics as requested
    if "mean" in metrics:
        stats["mean"] = float(np.mean(data))
    if "median" in metrics:
        stats["median"] = float(np.median(data))
    if "min" in metrics:
        stats["min"] = float(np.min(data))
    if "max" in metrics:
        stats["max"] = float(np.max(data))
    if "std" in metrics:
        stats["std"] = float(np.std(data))
    if "count" in metrics:
        stats["count"] = int(len(data))
    
    # Handle percentile calculation (e.g. percentile_25, percentile_90)
    for metric in metrics:
        if metric.startswith("percentile_"):
            try:
                # Extract the percentile value from the metric name
                p = float(metric.split("_")[1])
                # Calculate the specified percentile
                stats[metric] = float(np.percentile(data, p))
            except ValueError:
                # Skip invalid percentile specification
                continue
    
    return stats


# ========================================
# app/tools/analysis/psd.py
# ========================================

from scipy.stats import binned_statistic
from typing import Tuple

def compute_psd(rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Power Spectral Density (PSD) from time series.
    
    Args: 
        rate: Time series data array
        
    Returns:
        Tuple containing
            - freqs: Array of frequencies (Hz)
            - psd: Array of corresponding power values
    """
    N = len(rate)
    dt = 1.0  # Assumed time step between samples
    
    # Calculate FFT of the entire spectrum 
    fft_vals = np.fft.fft(rate)[:N//2+1]
    
    # Compute PSD by squaring the magnitude of FFT values and multiplying by 2
    psd = 2.0 * (np.abs(fft_vals) ** 2)
    
    # Calculate the corresponding frequencies
    freqs = np.fft.fftfreq(N, dt)[:N//2+1]
    
    # Remove DC component (first value) and Nyquist frequency (last value if it exists)
    return freqs[1:-1], psd[1:-1]


def bin_psd(freqs: np.ndarray,
            psd: np.ndarray,
            low_freq: float = 1e-5,
            high_freq: float = 0.05,
            bins: int = 3500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin PSD values into specified frequency ranges.
    
    Args:
        freqs: Array of frequencies
        psd: Array of PSD values
        low_freq: Minimum frequency to include (default: 1e-5 Hz)
        high_freq: Maximum frequency to include (default: 0.05 Hz)
        bins: Number of frequency bins to use (default: 3500)
        
    Returns:
        Tuple containing:
            - centers: Array of bin center frequencies
            - psd_binned: Array of mean PSD values in each bin
    """
    total_points = len(freqs)
    
    # Check if bins exceed the number of data points and log a warning
    if bins >= total_points:
        logger.warning(
            f"Number of bins ({bins}) >= number of data points ({total_points}). "
            "Reducing bins to avoid empty bins or NaNs in PSD."
        )
    
    # Automatically reduce bins to prevent exceeding the number of points
    adjusted_bins = min(bins, total_points)
    
    # Create bin edges linearly spaced between low_freq and high_freq
    bin_edges = np.linspace(low_freq, high_freq, adjusted_bins + 1)
    
    # Perform binning: compute mean PSD value in each bin
    psd_binned, _, _ = binned_statistic(freqs, psd, statistic='mean', bins=bin_edges)
    
    # Calculate bin centers (midpoint between edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return centers, psd_binned


# ========================================
# app/tools/analysis/fitting.py
# ========================================

from scipy.optimize import curve_fit
from typing import Tuple, Dict, Any, Optional

def power_law_fn(f: np.ndarray, A: float, b: float, n: float) -> np.ndarray:
    """
    Power law model function: PSD(f) = A/f^b + n
    
    Args:
        f: Frequency array
        A: Amplitude
        b: Power law index 
        n: Frequency-independent noise level
        
    Returns:
        Power spectral density values according to the model
    """
    return (A / (f ** b)) + n


def bending_power_law_fn(f: np.ndarray, A: float, fb: float, sh: float, n: float) -> np.ndarray:
    """
    Bending power law model function: PSD(f) = A / [f(1+(f/fb)^(sh-1))] + n 
    
    Args: 
        f: Frequency array
        A: Amplitude
        fb: Break frequency
        sh: Shape parameter controlling the sharpness of the transition
        n: Frequency-independent noise level
        
    Returns:
         Power spectral density values according to the model
    """
    return (A / (f * (1 + (f / fb) ** (sh - 1)))) + n


def fit_power_law(x: np.ndarray, 
                  y: np.ndarray, 
                  noise_bound_percent: float = 0.7,
                  maxfev: int = 1000000,
                  initial_params: Optional[Dict[str, float]] = None,
                  param_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[float, float, float]:
    """
    Fit a power law model to frequency data.
    
    Args:
        x: Frequency array
        y: Power spectral density array
        noise_bound_percent: Controls the allowed range for the noise parameter
        maxfev: Maximum number of function evaluations
        initial_params: Optional dictionary with custom initial parameter guesses
        param_bounds: Optional dictionary with custom parameter bounds
        
    Returns:
        Tuple of fitted parameters: (A, b, n)
    """
    try:
        # Default values
        initial_params = initial_params or {}
        param_bounds = param_bounds or {}
        
        # Estimate the noise level from the high-frequency end of the spectrum
        n0 = float(np.mean(y[-10:]))
        
        # Initial guesses for parameters
        A0 = initial_params.get('A', 1.0)
        b0 = initial_params.get('b', 1.0)
        
        p0 = [A0, b0, n0]
        
        # Set bounds for the noise parameter based on noise_bound_percent
        n_bound_min = n0 * (1 - noise_bound_percent)
        n_bound_max = n0 * (1 + noise_bound_percent)
        
        # Set bounds for all parameters
        A_bounds = param_bounds.get('A', (0, 15))
        b_bounds = param_bounds.get('b', (0.1, 3.0))
        
        bounds = ([A_bounds[0], b_bounds[0], n_bound_min], 
                  [A_bounds[1], b_bounds[1], n_bound_max])
        
        # First step of curve fitting
        popt1, _ = curve_fit(power_law_fn, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        
        # Extract parameters from first fit
        _, _, n1 = popt1
        
        # Refine noise bounds for second step using the same noise_bound_percent
        n_bound_min_2 = n1 * (1 - noise_bound_percent)
        n_bound_max_2 = n1 * (1 + noise_bound_percent)
        
        # Second step with refined bounds
        bounds2 = ([A_bounds[0], b_bounds[0], n_bound_min_2], 
                   [A_bounds[1], b_bounds[1], n_bound_max_2])
        
        # Perform second fit using first fit result as initial guess
        popt2, _ = curve_fit(power_law_fn, x, y, p0=popt1, bounds=bounds2, maxfev=maxfev)
        
        # Log the fit results
        logger.info(f"Power law fit results: A={popt2[0]:.3e}, b={popt2[1]:.3f}, n={popt2[2]:.3e}")
        
        # Return the fitted parameters from second step
        return tuple(popt2)
        
    except Exception as e:
        logger.error(f"Error fitting power law: {str(e)}")
        # Return default values in case of fitting error
        return (1.0, 1.0, n0)


def fit_bending_power_law(x: np.ndarray, 
                          y: np.ndarray, 
                          noise_bound_percent: float = 0.7,
                          maxfev: int = 1000000,
                          initial_params: Optional[Dict[str, float]] = None,
                          param_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[float, float, float, float]:
    """
    Fit a bending power law model to frequency data.
    
    Args:
        x: Frequency array
        y: PSD values array
        noise_bound_percent: Controls the allowed range for the noise parameter
        maxfev: Maximum number of function evaluations
        initial_params: Optional dictionary with custom initial parameter guesses
        param_bounds: Optional dictionary with custom parameter bounds
        
    Returns:
        Tuple of fitting parameters: (A, fb, sh, n)
    """
    try: 
        # Default values
        initial_params = initial_params or {}
        param_bounds = param_bounds or {}
        
        # Estimate the noise level from the high-frequency end of the spectrum
        n0 = float(np.mean(y[-10:]))
        
        # Estimate the break frequency (fb) as the frequency where PSD drops to 1/10 of max,
        # or if that's not found, use the middle of the frequency range
        default_fb0 = x[np.argmin(np.abs(y - max(y) / 10))] if np.any(y < max(y) / 10) else x[len(x)//2]
        
        # Set Initial guesses for parameters
        A0 = initial_params.get('A', 10.0)
        fb0 = initial_params.get('fb', default_fb0)
        sh0 = initial_params.get('sh', 1.0)
        
        p0 = [A0, fb0, sh0, n0]
        
        # Set bounds for the noise parameter based on noise_bound_percent
        n_bound_min = n0 * (1 - noise_bound_percent)
        n_bound_max = n0 * (1 + noise_bound_percent)
        
        # Set bound for all parameters
        A_bounds = param_bounds.get('A', (0, 1e38))
        fb_bounds = param_bounds.get('fb', (x[0], x[-1]))
        sh_bounds = param_bounds.get('sh', (0.3, 3.0))
        
        # Handle the case of infinity in bounds
        A_min = A_bounds[0] if not np.isinf(A_bounds[0]) else 0.0
        A_max = A_bounds[1] if not np.isinf(A_bounds[1]) else 1e38
        fb_min = fb_bounds[0] if not np.isinf(fb_bounds[0]) else x[0]
        fb_max = fb_bounds[1] if not np.isinf(fb_bounds[1]) else x[-1]
        sh_min = sh_bounds[0] if not np.isinf(sh_bounds[0]) else 0.3
        sh_max = sh_bounds[1] if not np.isinf(sh_bounds[1]) else 3.0
        
        bounds = ([A_min, fb_min, sh_min, n_bound_min], 
                  [A_max, fb_max, sh_max, n_bound_max])
        
        # First step of curve fitting
        popt1, _ = curve_fit(bending_power_law_fn, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        
        # Extract parameters from first fit
        _, _, _, n1 = popt1
        
        # Refine noise bounds for second step
        n_bound_min_2 = n1 * (1 - noise_bound_percent)
        n_bound_max_2 = n1 * (1 + noise_bound_percent)
        
        # Second step with refined bounds
        bounds2 = ([A_min, fb_min, sh_min, n_bound_min_2], 
                   [A_max, fb_max, sh_max, n_bound_max_2])
        
        # Perform second fit using first fit result as initial guess
        popt2, _ = curve_fit(bending_power_law_fn, x, y, p0=popt1, bounds=bounds2, maxfev=maxfev)
        
        # Log the fit result
        logger.info(f"Bending power law fit results: A={popt2[0]:.3e}, fb={popt2[1]:.3f}, sh={popt2[2]:.3f}, n={popt2[3]:.3e}")
        
        # Return the fitted parameters from second step
        return tuple(popt2)
    
    except RuntimeError as e:
        logger.error(f"Curve fitting failed: {str(e)}")
        raise Exception(f"Curve fitting failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error fitting bending power law: {str(e)}")
        raise Exception(f"Error fitting bending power law: {str(e)}")


# ========================================
# app/tools/visualization/plots.py
# ========================================

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_psd_figure(x: np.ndarray, y: np.ndarray, title: str = "Power Spectral Density") -> Figure:
    """
    Create a figure displaying a Power Spectral Density (PSD) plot.
    
    Args:
        x: Frequency array
        y: PSD values array
        title: Plot title
        
    Returns:
        Matplotlib Figure object containing the PSD plot
    """
    # Create figure and axes with specified size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the PSD data with logarithmic x and y axes
    ax.loglog(x, y)
    
    # Add axis labels
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    
    # Add title
    ax.set_title(title)
    
    # Add grid for better readability on log-log plot
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    
    # Adjust layout to prevent labels from being cut off
    fig.tight_layout()
    
    return fig


def plot_power_law_with_residual_figure(x: np.ndarray, 
                                        y: np.ndarray, 
                                        A: float, 
                                        b: float, 
                                        n: float,
                                        title: str = "Power Law Fit",
                                        yscale_range=None) -> Figure:
    """
    Create a figure showing data with a power law model fit and residuals.
    
    Args:
        x: Frequency array
        y: PSD values array
        A: Amplitude parameter from the fit
        b: Power law index from the fit
        n: Frequency-independent noise level from the fit
        title: Plot title
        yscale_range: Optional tuple of (ymin, ymax) to override automatic y-axis limits
        
    Returns:
        Matplotlib Figure object containing the data, fit, and residuals
    """
    # Calculate model components
    y_fit = power_law_fn(x, A, b, n)       # Full model (signal + noise)
    y_model = power_law_fn(x, A, b, 0)     # Signal component only
    y_noise = np.ones_like(x) * n          # Noise component only
    
    # Calculate residuals as ratio of data to model (multiplicative residuals)
    residuals = y / y_fit
    
    # Create figure with two subplots (main plot and residuals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                 gridspec_kw={'height_ratios': [3, 1]}, 
                                 sharex=True)
    
    # Plot data and model components on main plot
    ax1.plot(x, y, color='black', label='Original PSD')
    ax1.plot(x, y_fit, '--', color='red', label='PSD + Noise')
    ax1.plot(x, y_model, '-', color='blue', label=rf'$P(f)=A f^{{-b}}+n$')
    ax1.plot(x, y_noise, ':', color='green', label='Noise')
    
    # Configure main plot
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title(f'{title}: \nA={A:.2e}, b={b:.2f}, n={n:.2e}')
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Set y-axis limits either automatically or from provided range
    if yscale_range is None:
        # Auto-calculate y limits, excluding zeros and providing some margin
        y_min = np.min(y[y > 0])
        y_max = np.max(y)
        y_lower = y_min * 0.5  # 50% below min(y)
        y_upper = y_max * 1.2  # 20% above max(y)
        
        ax1.set_ylim(y_lower, y_upper)
    else:
        ax1.set_ylim(yscale_range)
    
    # Plot residuals
    ax2.plot(x, residuals, '.', color='purple', markersize=4)
    ax2.axhline(1, color='gray', linestyle='--', linewidth=1)  # Reference line at 1.0
    
    # Configure residual plot
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Residual (PSD/Model)')
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_bending_power_law_with_residual_figure(x: np.ndarray, 
                                                y: np.ndarray,
                                                A: float,
                                                fb: float,
                                                sh: float,
                                                n: float,
                                                title: str = "Bending Power Law Fit",
                                                yscale_range=None) -> Figure:
    """
    Create a figure showing data with a bending power law model fit and residuals.
    
    Args:
        x: Frequency array
        y: PSD values array
        A: Amplitude parameter from the fit
        fb: Break frequency from the fit
        sh: Shape parameter from the fit
        n: Frequency-independent noise level from the fit
        title: Plot title
        yscale_range: Optional tuple of (ymin, ymax) to override automatic y-axis limits
        
    Returns:
        Matplotlib Figure object containing the data, fit, and residuals
    """
    # Calculate model components
    y_fit = bending_power_law_fn(x, A, fb, sh, n)  # Full model (signal + noise)
    y_model = bending_power_law_fn(x, A, fb, sh, 0)  # Signal component only
    y_noise = np.ones_like(x) * n  # Noise component only
    
    # Calculate residuals as ratio of data to model (multiplicative residuals)
    residuals = y / y_fit
    
    # Create figure with two subplots (main plot and residuals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 sharex=True)
    
    # Plot data and model components on main plot
    ax1.plot(x, y, color='black', label='Original PSD')
    ax1.plot(x, y_fit, '--', color='red', label='PSD + Noise')
    ax1.plot(x, y_model, '-', color='blue', 
            label=rf'$P(f)=\frac{{A}}{{f[1+(f/f_b)^{{{sh}-1}}]}}')
    ax1.plot(x, y_noise, ':', color='green', label='Noise')
    
    # Configure main plot
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title(f'{title}: \nA={A:.2e}, fb={fb:.2e}, sh={sh:.2f}, n={n:.2e}')
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Set y-axis limits either automatically or from provided range
    if yscale_range is None:
        # Auto-calculate y limits, excluding zeros and providing some margin
        y_min = np.min(y[y > 0])
        y_max = np.max(y)
        y_lower = y_min * 0.5  # 50% below min(y)
        y_upper = y_max * 1.2  # 20% above max(y)
        
        ax1.set_ylim(y_lower, y_upper)
    else:
        ax1.set_ylim(yscale_range)
    
    # Plot residuals
    ax2.plot(x, residuals, '.', color='purple', markersize=4)
    ax2.axhline(1, color='gray', linestyle='--', linewidth=1)  # Reference line at 1.0
    
    # Configure residual plot
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Residual (PSD/Model)')
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig