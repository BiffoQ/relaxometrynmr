import os
import warnings
import nmrglue as ng
import numpy as np
import pandas as pd
from mrsimulator import signal_processor as sp
from scipy.optimize import curve_fit
from scipy.integrate import simpson, trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def _parse_fwhm_hz(fwhm):
    """
    Parse a line-broadening FWHM into a plain float in Hz. Accepts the same
    inputs `process_spectrum`'s mrsimulator apodization accepts (a bare
    number, or a string like `'200 Hz'`/`'1.2 kHz'`); 0/None/'' means no
    broadening.
    """
    if fwhm in (None, "", 0):
        return 0.0
    if isinstance(fwhm, (int, float)):
        return float(fwhm)
    from astropy import units as u
    return float(u.Quantity(fwhm).to(u.Hz).value)


def _lorentzian_kernel(n_points, gamma_points):
    """
    Normalized (unit-area) discrete Lorentzian convolution kernel of the
    given HWHM (`gamma_points`, in samples), centered on `n_points`
    samples. Convolving a spectrum with this is the frequency-domain
    equivalent of multiplying its (unavailable, already-FFT'd) FID by an
    exponential decay window - the same relationship `gaussian_filter1d`
    exploits for the Gaussian case in `process_processed_spectrum`.
    """
    n = np.arange(n_points) - n_points // 2
    kernel = gamma_points / (np.pi * (n ** 2 + gamma_points ** 2))
    return kernel / kernel.sum()


class _Coordinates:
    def __init__(self, value):
        self.value = value


class _Dimension:
    def __init__(self, value):
        self.coordinates = _Coordinates(value)


class _DependentVariable:
    def __init__(self, data):
        self.components = [data]


class _PhasedSpectrum:
    """
    Minimal CSDM-like wrapper around an already Fourier-transformed spectrum
    (e.g. Bruker pdata), exposing the same `.dimensions[0].coordinates.value`
    / `.dependent_variables[0].components[0]` interface as a CSDM dataset.
    This lets processed (pdata) and raw spectra flow through the same
    downstream code (`integrate_spectrum_region`, plotting, API handlers)
    without callers needing to special-case the source.
    """

    def __init__(self, ppm, data):
        self.dimensions = [_Dimension(ppm)]
        self.dependent_variables = [_DependentVariable(data)]


class T1Functions:

    """
    A comprehensive class for processing and analyzing T1 relaxation NMR data from Bruker format.
    This class provides tools for data conversion, processing, visualization, and fitting of T1 relaxation data.
    It supports various fitting models including mono-, bi-, and tri-exponential functions, as well as
    stretched exponentials for complex relaxation behavior analysis.
    """

    def __init__(self, file_path):

        """
        Initialize T1Functions with a path to Bruker NMR data.

        Args:
            file_path (str):
            Path to the Bruker NMR data directory containing the FID and acquisition
            parameters.
            The directory should contain standard Bruker files (fid, acqu, acqus, etc.)
        """
        self.file_path = file_path

    def read_and_convert_bruker_data(self, save_nmrpipe=True):

        """
        Read and convert Bruker NMR data to NMRPipe and CSDM formats.
        This function handles the complex
        process of importing raw Bruker data,
        interpreting the acquisition parameters, and converting
        the data into formats suitable for further analysis.
        It automatically detects and loads the
        variable delay list (vdlist, vplist, or vclist) used in the T1 experiment.

        Args:
            save_nmrpipe (bool): Whether to save the converted NMRPipe data to disk.
            NMRPipe format is widely used in the NMR community and
            can be processed with various
                                third-party tools.

        Returns:
            tuple: A tuple containing three elements:
                - list of 1D spectra: Each element
                    is a single FID from the pseudo-2D dataset
                - variable delay list: The time delays used in the T1 experiment
                - CSDM dataset: The complete dataset in CSDM format for advanced processing

        Raises:
            FileNotFoundError: If no variable delay list file
                                (vdlist, vplist, or vclist) is found
                                    in the Bruker data directory
        """
        # Read Bruker data
        dic, data = ng.bruker.read(self.file_path)
        u = ng.bruker.guess_udic(dic, data)

        # Create the converter object and initialize with Bruker data
        C = ng.convert.converter()
        C.from_bruker(dic, data, u)

        # Optionally save NMRPipe formatted data
        if save_nmrpipe:
            ng.pipe.write(self.file_path + "2d_pipe.fid", *C.to_pipe(), overwrite=True)

        # Convert to CSDM format
        csdm_ds = C.to_csdm()
        dim1, dim2 = csdm_ds.shape

        # Extract 1D spectra from the 2D dataset
        spectra_1d = [csdm_ds[:, i] for i in range(dim2)]

        possible_files = ["vdlist", "vplist", "vclist"]

        for filename in possible_files:

            try:
                vd_list = np.loadtxt(self.file_path + filename)
                break
            except OSError:

                if filename == possible_files[-1]:
                    raise FileNotFoundError("No vdlist, vplist, or vclist file found.")
                continue

        # Bruker's raw pseudo-2D acquisition dimension (TD1) can include a
        # padded/extra plane, or the vdlist can be truncated relative to it,
        # leaving spectra_1d and vd_list with different lengths. Truncate
        # both to the shorter one so downstream code (e.g. /integrate) never
        # indexes past the end of vd_list.
        n = min(len(spectra_1d), len(vd_list))
        if len(spectra_1d) != len(vd_list):
            warnings.warn(
                f"Number of spectra ({len(spectra_1d)}) does not match the "
                f"length of the variable delay list ({len(vd_list)}); "
                f"truncating both to {n}."
            )
            spectra_1d = spectra_1d[:n]
            vd_list = vd_list[:n]

        return spectra_1d, vd_list, csdm_ds

    def _is_pdata_path(self, path):
        """
        Determine whether `path` points to Bruker processed data (pdata)
        rather than a raw experiment directory containing fid/ser.

        Returns True if `path` itself is a procno folder (contains 1r/2rr),
        or if it is an experiment folder that only exposes a pdata subfolder
        and no raw fid/ser file.
        """
        if os.path.isfile(os.path.join(path, "1r")) or os.path.isfile(os.path.join(path, "2rr")):
            return True

        has_raw = os.path.isfile(os.path.join(path, "fid")) or os.path.isfile(os.path.join(path, "ser"))
        if has_raw:
            return False

        return os.path.isdir(os.path.join(path, "pdata"))

    def _resolve_pdata_path(self, proc_no=1):
        """
        Resolve `self.file_path` to a concrete pdata/<proc_no> directory,
        whether `self.file_path` already points at that directory or at
        the experiment root above it.
        """
        path = self.file_path.rstrip("/\\")

        if os.path.isfile(os.path.join(path, "1r")) or os.path.isfile(os.path.join(path, "2rr")):
            return path

        return os.path.join(path, "pdata", str(proc_no))

    def _experiment_root(self, pdata_path):
        """Given a .../<expno>/pdata/<procno> path, return the <expno> directory."""
        return os.path.dirname(os.path.dirname(pdata_path.rstrip("/\\")))

    def read_processed_bruker_data(self, proc_no=1, empty_atol=1e-10):
        """
        Read a pseudo-2D relaxation series directly from Bruker processed
        data (pdata) instead of the raw FID/SER file.

        This is needed when only processed data is available (e.g. data
        acquired on instruments/setups where the raw time-domain data was
        not exported or is inaccessible), such as relaxation series recorded
        at Warwick. Because Bruker's automatic processing (phasing, etc.) is
        not always well optimized, the returned spectra are frequency-domain
        and complex, so they can still be run through
        `zero_order_phasing`/`first_order_phasing` for manual correction.

        Bruker pseudo-2D pdata arrays are frequently padded with trailing
        placeholder rows (e.g. TD1 rounded up to a convenient size beyond the
        number of delays actually listed in vdlist/vplist/vclist). Any row
        whose real (…rr/…r) OR imaginary (…ii/…i) component is entirely zero
        is treated as an empty placeholder and dropped, and the delay list is
        truncated to match, so callers never see spectra with no signal and
        no corresponding delay.

        Args:
            proc_no (int): Processing number to read, i.e. the pdata/<proc_no>
                            subfolder. Defaults to 1.
            empty_atol (float): Absolute tolerance used to decide a component
                                 is all-zero (and thus empty).

        Returns:
            tuple: (spectra, vd_list, ppm, sw)
                - spectra: list of 1D complex frequency-domain spectra, one
                  per non-empty point in the pseudo-2D series
                - vd_list: variable delay list (T1 relaxation time points),
                  truncated to match `spectra`
                - ppm: ppm scale shared by all spectra
                - sw: spectral width (Hz) of the direct dimension, needed to
                  convert a Gaussian FWHM (Hz) into an equivalent smoothing
                  width when line-broadening these already Fourier-transformed
                  spectra (see `process_processed_spectrum`)

        Raises:
            FileNotFoundError: If no vdlist, vplist, or vclist file is found
                                in the experiment directory.
        """
        pdata_path = self._resolve_pdata_path(proc_no)

        dic, (real, imag) = ng.bruker.read_pdata(pdata_path, all_components=True)
        udic = ng.bruker.guess_udic(dic, real)
        uc = ng.fileiobase.uc_from_udic(udic, dim=1)
        ppm = uc.ppm_scale()
        sw = udic[1]["sw"]

        exp_root = self._experiment_root(pdata_path)
        possible_files = ["vdlist", "vplist", "vclist"]
        vd_list = None
        for filename in possible_files:
            try:
                vd_list = np.loadtxt(os.path.join(exp_root, filename))
                break
            except OSError:
                continue
        if vd_list is None:
            raise FileNotFoundError("No vdlist, vplist, or vclist file found.")
        vd_list = np.atleast_1d(vd_list)

        non_empty = ~(
            np.all(np.isclose(real, 0, atol=empty_atol), axis=-1)
            | np.all(np.isclose(imag, 0, atol=empty_atol), axis=-1)
        )
        real, imag = real[non_empty], imag[non_empty]

        n = min(real.shape[0], vd_list.shape[0])
        spectra = [real[i] + 1j * imag[i] for i in range(n)]
        vd_list = vd_list[:n]

        return spectra, vd_list, ppm, sw

    def load_relaxation_series(self, proc_no=1, save_nmrpipe=True):
        """
        Load a pseudo-2D relaxation series, auto-detecting whether
        `self.file_path` points to raw Bruker data (fid/ser) or to
        processed data (pdata) and dispatching accordingly.

        Use this as the single entry point when it is unknown in advance
        (e.g. datasets pooled from multiple instruments/sites) whether raw
        FIDs will be available. Raw data goes through the standard
        `read_and_convert_bruker_data` pipeline (apodization, zero-filling,
        FFT, phasing). Processed data goes through
        `read_processed_bruker_data`, skipping the FFT step since Bruker
        has already performed it; only phase correction is typically still
        needed there.

        Args:
            proc_no (int): pdata processing number to use if processed data
                            is detected. Ignored for raw data.
            save_nmrpipe (bool): Whether to save NMRPipe output if raw data
                                  is detected. Ignored for processed data.

        Returns:
            tuple: (source, spectra, vd_list, extra)
                - source (str): "raw" or "processed", indicating which path
                  was taken
                - spectra: list of 1D spectra (CSDM datasets for "raw",
                  complex ndarrays for "processed")
                - vd_list: variable delay list
                - extra: the CSDM dataset for "raw", or (ppm, sw) for
                  "processed"
        """
        if self._is_pdata_path(self.file_path):
            spectra, vd_list, ppm, sw = self.read_processed_bruker_data(proc_no=proc_no)
            return "processed", spectra, vd_list, (ppm, sw)

        spectra_1d, vd_list, csdm_ds = self.read_and_convert_bruker_data(save_nmrpipe=save_nmrpipe)
        return "raw", spectra_1d, vd_list, csdm_ds

    def process_processed_spectrum(self, spectrum, ppm, sw, fwhm, ph0, ph1, window_type='gaussian'):
        """
        Phase-correct (and optionally line-broaden) a spectrum read from
        Bruker processed data (pdata).

        Unlike `process_spectrum`, no zero-filling or FFT is applied here:
        Bruker's own processing already performed that step, and there is no
        time-domain FID left to re-apodize the same way. If `fwhm` is
        nonzero, extra broadening is instead applied directly in the
        frequency domain as a convolution - by the Fourier convolution
        theorem this is equivalent to multiplying the (unavailable) FID by
        the same window `process_spectrum` uses, without needing to
        reconstruct a pseudo-FID via inverse FFT. `window_type='gaussian'`
        convolves with a Gaussian (matches Gaussian multiplication of the
        FID); `window_type='lorentzian'` convolves with a Lorentzian
        (matches exponential multiplication of the FID, i.e. classic
        Lorentzian line broadening). The result exposes the same
        `dimensions[0].coordinates.value` / `dependent_variables[0].components[0]`
        interface as `process_spectrum`'s CSDM output, so processed and raw
        spectra can be passed to `integrate_spectrum_region` and other
        downstream code interchangeably.

        Args:
            spectrum (ndarray): Complex frequency-domain spectrum, as returned
                by `read_processed_bruker_data` / `load_relaxation_series`.
            ppm (ndarray): ppm scale shared by all spectra in the series,
                returned alongside `spectrum` by the same call.
            sw (float): Spectral width (Hz) of the direct dimension, returned
                alongside `ppm`; used to convert `fwhm` into an equivalent
                smoothing width in points.
            fwhm (float or str): Extra line broadening to apply, e.g. `200`
                or `'200 Hz'`. 0 (or falsy) applies none.
            ph0 (float): Zero-order phase correction in degrees.
            ph1 (float): First-order phase correction factor.
            window_type (str): 'gaussian' or 'lorentzian' - which window
                function's frequency-domain equivalent to apply.

        Returns:
            _PhasedSpectrum: phase-corrected (and optionally broadened) spectrum.
        """
        fwhm_hz = _parse_fwhm_hz(fwhm)
        if fwhm_hz > 0:
            hz_per_point = sw / spectrum.shape[0]
            if window_type == 'lorentzian':
                gamma_points = (fwhm_hz / hz_per_point) / 2.0
                kernel = _lorentzian_kernel(spectrum.shape[0], gamma_points)
                spectrum = (
                    fftconvolve(spectrum.real, kernel, mode='same')
                    + 1j * fftconvolve(spectrum.imag, kernel, mode='same')
                )
            else:
                sigma_points = (fwhm_hz / hz_per_point) / 2.354820045030949
                spectrum = (
                    gaussian_filter1d(spectrum.real, sigma_points)
                    + 1j * gaussian_filter1d(spectrum.imag, sigma_points)
                )

        phased = self.zero_order_phasing(spectrum, ph0)
        phased = self.first_order_phasing(phased, ph1)
        return _PhasedSpectrum(ppm, phased)

    def zero_fill(self, data, new_len):
        """
        Zero-fill NMR data to extend its length,
        improving spectral resolution in the frequency domain.
        Zero-filling is a crucial preprocessing step that increases the
        digital resolution of the spectrum
        by extending the FID with zeros.
        This does not add any new information but provides interpolation
        in the frequency domain, resulting in smoother spectral lines.

        Args:
            data (ndarray): Input NMR data array, typically a time-domain FID
            new_len (int): Desired length after zero-filling.
            Should be greater than the original
                          data length, typically a power of 2 for efficient FFT processing

        Returns:
            ndarray: Zero-filled data array with length new_len.
            If new_len is less than or equal
                    to the current length, returns the original data unchanged
        """
        current_len = data.shape[0]
        if new_len <= current_len:
            return data
        zeros_to_add = new_len - current_len
        return np.pad(data, (0, zeros_to_add), 'constant')

    def zero_order_phasing(self, data, ph0):
        """
        Apply zero-order phase correction to NMR data.
        Zero-order phasing applies a constant phase
        adjustment across the entire spectrum,
        correcting for the receiver phase offset during
        signal acquisition.
        This is essential for obtaining pure absorption mode spectra and is
        typically the first step in phase correction.

        Args:
            data (ndarray): Complex input NMR data array in either time or frequency domain
            ph0 (float): Phase angle in degrees. The phase correction is applied uniformly
                        across the entire spectrum. Typical values range from -180° to +180°

        Returns:
            ndarray: Phase-corrected complex data array.
            The correction is applied by multiplying
                    the data by exp(i*φ), where φ is the phase angle in radians
        """
        phase = np.deg2rad(ph0)

        phased_data = data * np.exp(1j * phase)

        return phased_data

    def first_order_phasing(self, data, ph1):
        """
        Apply first-order phase correction to NMR data.
        First-order phasing applies a frequency-dependent
        phase correction that varies linearly across the spectrum.
        This corrects for delays between
        excitation and detection, digital filtering effects,
        and other instrumental factors that can
        cause frequency-dependent phase errors.

        Args:
            data (ndarray): Complex input NMR data array, typically in the frequency domain
            ph1 (float): First-order phase correction factor.
            This determines the slope of the
                        phase correction across the spectrum.
                        The actual phase correction at each
                        point is ph1 * frequency

        Returns:
            ndarray: Phase-corrected complex data array with
            frequency-dependent phase adjustment
        """
        n = data.shape[0]
        ppm = np.linspace(-n//2, n//2, n)

        phase = np.deg2rad(ph1*ppm)

        phased_data = data * np.exp(1j * phase)

        return phased_data

    def process_spectrum(self, spectrum, fwhm, zero_fill_factor, ph0, ph1, window_type='gaussian'):
        """
        Process NMR spectrum with a comprehensive set of
        standard NMR data processing steps.
        This function applies, in order:
        1. Apodization (Gaussian or Lorentzian window) for line broadening
           and S/N improvement
        2. Zero-filling for increased digital resolution
        3. Fourier transformation to convert from time to frequency domain
        4. Phase corrections (both zero- and first-order)
        5. Conversion to ppm scale for chemical shift referencing

        Gaussian apodization (`window_type='gaussian'`) multiplies the FID by
        a Gaussian decay - it has no long tail, so it improves resolution
        (narrows lines, suppresses truncation wiggles) at some cost in S/N,
        and does not distort an already-Lorentzian lineshape's tails. Best
        for resolving overlapping/crowded peaks, e.g. many ssNMR spectra with
        several close chemical environments.
        Lorentzian (exponential) apodization (`window_type='lorentzian'`)
        multiplies the FID by an exponential decay, matched to the natural
        homogeneous linewidth (T2*-limited decay) - it maximizes
        signal-to-noise for an already-Lorentzian peak (matched filter) but
        broadens the line and adds Lorentzian tails. Best for boosting S/N on
        well-resolved, individually narrow lines where extra resolution
        isn't needed.

        Args:
            spectrum (ndarray): Input time-domain NMR spectrum (FID)
            fwhm (float): Full width at half maximum for apodization in Hz.
                         Controls the trade-off between resolution and signal-to-noise
            zero_fill_factor (int): Factor for zero filling, typically 2-4 for moderate
                                   resolution enhancement
            ph0 (float): Zero-order phase correction in degrees
            ph1 (float): First-order phase correction factor
            window_type (str): 'gaussian' (resolution-enhancing, default) or
                         'lorentzian' (S/N-enhancing, matched filter)

        Returns:
            ndarray: Fully processed frequency-domain spectrum referenced to ppm scale
        """
        # Apply line broadening and Fourier transform
        apodization = sp.apodization.Exponential(FWHM=fwhm) if window_type == 'lorentzian' else sp.apodization.Gaussian(FWHM=fwhm)
        ft = sp.SignalProcessor(operations=[apodization, sp.FFT()])

        # Apply zero filling
        spectrum = self.zero_fill(spectrum, zero_fill_factor * spectrum.shape[0])

        # Apply first order phasing
        exp_spectrum = self.zero_order_phasing(spectrum, ph0)

        # Apply operations from the signal processor
        exp_spectrum = ft.apply_operations(dataset=exp_spectrum)

        # Apply second order phasing
        exp_spectrum = self.first_order_phasing(exp_spectrum, ph1)

        # Convert to ppm
        exp_spectrum.dimensions[0].to("ppm", "nmr_frequency_ratio")

        return exp_spectrum

    def integrate_spectrum_region(self, exp_spectrum, ppm_start, ppm_end):
        """
        Calculate integrated intensity of a spectral region using multiple
        numerical integration
        methods.
        This function provides robust integration by comparing different numerical
        integration techniques
        (trapezoid and Simpson's rules) and estimating the uncertainty
        in the integration.
        This is particularly useful for quantitative NMR analysis and
        relaxation measurements.

        Args:
            exp_spectrum (ndarray): Input frequency-domain NMR spectrum
            ppm_start (float): Starting chemical shift in ppm for the integration region
            ppm_end (float): Ending chemical shift in ppm for the integration region

        Returns:
            tuple: A comprehensive set of integration results:
                - trapezoid integration value
                - Simpson's rule integration value
                - x coordinates of the integrated region
                - y coordinates of the integrated region
                - estimated integration uncertainty (difference between methods)
        """
        # Convert the ppm range to indices
        ppm_scale = exp_spectrum.dimensions[0].coordinates.value

        # Create a mask for the region of interest
        region_mask = (ppm_scale >= ppm_start) & (ppm_scale <= ppm_end)

        # Extract the region of interest
        x_region = ppm_scale[region_mask]
        y_real = exp_spectrum.dependent_variables[0].components[0].real
        y_region = y_real[region_mask]

        # Calculate integrated intensity using trapezoid and Simpson's rule methods
        integrated_intensity_trapz = trapezoid(y=y_region, x=x_region)
        integrated_intensity_simps = simpson(y=y_region, x=x_region)
        # integrated_intensity_romb = romb(y=y_region, dx=x_region[1]-x_region[0])
        integrated_uncertainty = abs(integrated_intensity_trapz - integrated_intensity_simps)

        return integrated_intensity_trapz, integrated_intensity_simps, x_region, y_region, integrated_uncertainty

    def plot_spectra_and_zoomed_regions(self, exp_spectra, x_regions, y_regions, xlim1, xlim2):
        """
        Create publication-quality plots of
        NMR spectra with both full view and zoomed regions.
        This visualization function creates a two-panel figure showing:
        1. The full spectrum for context
        2. A zoomed view of specific regions of interest

        The zoomed regions can be highlighted for emphasis, making it easy to focus on
        specific spectral features while maintaining the context of the full spectrum.

        Args:
            exp_spectra (list): List of processed NMR spectra to display
            x_regions (list): X coordinates for each region to be highlighted
            y_regions (list): Y coordinates for each highlighted region
            xlim1 (float): Lower chemical shift limit for the zoomed view (in ppm)
            xlim2 (float): Upper chemical shift limit for the zoomed view (in ppm)

        Returns:
            list: Maximum intensities from each spectrum, useful for normalization
                 and comparison between spectra
        """
        intensities = []


        for i, exp_spectrum in enumerate(exp_spectra):
            # if i == 0:
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={"projection": "csdm"})

            # Read via the shared dimensions/dependent_variables interface (same
            # one integrate_spectrum_region uses) rather than `.real`, so this
            # works for both actual CSDM datasets and the lightweight
            # _PhasedSpectrum wrapper used for processed (pdata) spectra.
            x = exp_spectrum.dimensions[0].coordinates.value
            y = exp_spectrum.dependent_variables[0].components[0].real

            ax[0].plot(x, y)
            ax[0].set_title(f"Full Spectrum {i+1}")
            ax[0].set_xlabel("ppm")
            ax[0].invert_xaxis()
            ax[1].plot(x, y, label="real")
            ax[1].fill_between(x_regions[i], y_regions[i], color='red', alpha=0.5)
            ax[1].set_title(f"Zoomed Spectrum {i+1}")
            ax[1].set_xlabel("ppm")
            ax[1].invert_xaxis()
            ax[1].set_xlim(xlim1, xlim2) #make this modular by passing x_lim as a parameter

            intensity = np.abs(exp_spectrum.dependent_variables[0].components[0].max())
            intensities.append(intensity)

        plt.tight_layout()
        plt.legend()
        plt.show()

        return intensities


    def lorentzian_fit(self, name, x_regions, y_regions,
                       plot_filepath='fit_plot.svg',
                       params_filepath='fit_params.csv',
                       data_filepath='fit_data.csv'):
        """
        Fit a Lorentzian to spectral data and save the plot, fit parameters, and raw data.

        Args:
            name (str): Sample name used in the output data filename.
            x_regions (list): X data (ppm values) from integrate_spectrum_region.
            y_regions (list): Y data (intensities) from integrate_spectrum_region.
            plot_filepath (str): Directory (or prefix) for the output SVG plot.
            params_filepath (str): Directory (or prefix) for the fitted parameters CSV.
            data_filepath (str): Directory (or prefix) for the raw+fitted data CSV.

        Returns:
            dict: Optimised parameters and their uncertainties
                  (keys: A, x0, gamma, offset, fwhm_hz).
        """
        x = np.array(x_regions).astype(np.float64).flatten()
        y = np.array(y_regions).astype(np.float64).flatten()

        def lorentzian(x, A, x0, gamma, offset):
            return (A / np.pi) * (gamma / (gamma**2 + (x - x0)**2)) + offset

        p0 = [np.max(y), x[np.argmax(y)], 7.19, np.min(y)]
        params, covariance = curve_fit(lorentzian, x, y, p0=p0)
        A_opt, x0_opt, gamma_opt, offset_opt = params
        A_err, x0_err, gamma_err, offset_err = np.sqrt(np.diag(covariance))

        half_height = 0.5 * A_opt / (gamma_opt * np.pi) + offset_opt
        fwhm = gamma_opt * 2
        y_fitted = lorentzian(x, *params)

        fig, ax = plt.subplots()
        ax.scatter(x, y, color='blue', alpha=0.5, label='Data')
        ax.plot(x, y_fitted, color='red', linestyle='-', label='Fit')
        ax.axvline(x0_opt, color='grey', linestyle='-')
        ax.hlines(half_height, x0_opt - fwhm / 2, x0_opt + fwhm / 2, color='black')
        ax.invert_xaxis()
        ax.set_xlabel('$^{17}$O chemical shift (ppm)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_filepath + 'lorentzian_fit.svg', bbox_inches='tight',
                    transparent=True, format='svg')
        plt.show()

        fwhm_hz = gamma_opt * 67.8 * 2
        fwhm_hz_err = 2 * gamma_err * 67.8 * 2
        params_df = pd.DataFrame({
            'Parameter': ['A', 'x0', 'gamma', 'offset', 'fwhm (Hz)'],
            'Value': [A_opt, x0_opt, gamma_opt, offset_opt, fwhm_hz],
            'Error': [A_err, x0_err, gamma_err, offset_err, fwhm_hz_err],
        })
        params_df.to_csv(params_filepath + 'fitted_fwhm_params.csv', index=False)

        pd.DataFrame({'x_regions': x, 'y_regions': y, 'y_fitted': y_fitted}).to_csv(
            data_filepath + name + '.csv', index=False
        )

        print(f'A      = {A_opt:.4f} +/- {A_err:.4f}')
        print(f'x0     = {x0_opt:.4f} +/- {x0_err:.4f}')
        print(f'gamma  = {gamma_opt:.4f} +/- {gamma_err:.4f}')
        print(f'offset = {offset_opt:.4f} +/- {offset_err:.4f}')
        print(f'fwhm   = {fwhm_hz:.4f} Hz +/- {fwhm_hz_err:.4f} Hz')

        return dict(A=A_opt, A_err=A_err, x0=x0_opt, x0_err=x0_err,
                    gamma=gamma_opt, gamma_err=gamma_err,
                    offset=offset_opt, offset_err=offset_err,
                    fwhm_hz=fwhm_hz, fwhm_hz_err=fwhm_hz_err)

    def mono_satrec_func(self, t, M0, T1, A, B):
        """
        Single-component saturation recovery function for T1 fitting.
        This model describes
        the simplest case of longitudinal relaxation where a single population of spins
        returns to equilibrium following an exponential recovery. It follows the equation:
        M(t) = A*M0*(1 - exp(-t/T1)) + B

        This model is appropriate for systems with a single well-defined relaxation process,
        such as pure liquids or mobile species in solution.

        Args:
            t (ndarray): Time points of the relaxation curve
            M0 (float): Equilibrium magnetization, representing the fully relaxed signal
            T1 (float): Spin-lattice relaxation time constant
            A (float): Scaling factor to account for experimental conditions
            B (float): Baseline offset to account for instrumental effects

        Returns:
            ndarray: Calculated magnetization values at each time point
        """
        return A*M0 * (1 - np.exp(-t / T1)) + B

    def di_satrec_func(self, t, M0, T1, A, M1, T2):
        """
        Two-component saturation recovery function for T1 fitting.
        This model describes systems
        with two distinct populations of spins, each with its own relaxation time constant.
        The function follows the equation:
        M(t) = A*[M0*(1 - exp(-t/T1)) + M1*(1 - exp(-t/T2))]

        This model is useful for heterogeneous systems, such as:
        - Different chemical environments in solids
        - Multiple phases in materials
        - Systems with distinct mobility regions

        Args:
            t (ndarray): Time points of the relaxation curve
            M0, M1 (float): Equilibrium magnetizations for each component
            T1, T2 (float): Relaxation time constants for each component
            A (float): Overall scaling factor

        Returns:
            ndarray: Combined magnetization values from both components
        """
        return A* ( (M0 * (1 - np.exp(-t / T1))) + (M1 * (1 - np.exp(-t / T2))) )

    def tri_satrec_func(self, t, M0, T1, A, M1, T2, M2, T3):
        """
        Three-component saturation recovery function for T1 fitting.
        This model handles complex
        systems with three distinct relaxation processes.
        The function follows the equation:
        M(t) = A*[M0*(1 - exp(-t/T1)) + M1*(1 - exp(-t/T2)) + M2*(1 - exp(-t/T3)^2)]

        This sophisticated model is applicable to:
        - Complex heterogeneous materials
        - Multi-phase systems
        - Materials with distinct domains of different mobilities
        - Systems with both surface and bulk relaxation processes

        Args:
            t (ndarray): Time points of the relaxation curve
            M0, M1, M2 (float): Equilibrium magnetizations for each component
            T1, T2, T3 (float): Relaxation time constants for each component
            A (float): Overall scaling factor

        Returns:
            ndarray: Combined magnetization values from all three components
        """
        return A* ( (M0 * (1 - np.exp(-t / T1))) + (M1 * (1 - np.exp(-t / T2))) +
                   (M2 * (1 - np.exp(-t / T3))**2) )

    def stretch_t1_exponential(self, t, T1_star, c):
        """
        Stretched exponential function for non-standard T1 relaxation behavior.
        This model
        accounts for systems with a continuous distribution of relaxation times,
        following
        the Kohlrausch-Williams-Watts (KWW) function:
        M(t) = 1 - exp(-(t/T1_star)^c)

        This model is particularly useful for:
        - Disordered systems
        - Glasses and polymers
        - Systems with complex relaxation dynamics
        - Materials with a distribution of correlation times

        The stretching exponent c (0 < c ≤ 1) indicates the degree of deviation from
        simple exponential behavior, with c = 1 recovering the standard exponential case.

        Args:
            t (ndarray): Time points of the relaxation curve
            T1_star (float): Characteristic relaxation time
            c (float): Stretching exponent, typically between 0 and 1

        Returns:
            ndarray: Stretched exponential relaxation curve values
        """
        return (1 - np.exp(-(t / T1_star)**c))


    def mono_expdec(self, t, T1, A, B, C):
        """
        Single-component exponential decay function for T1 relaxation analysis. This model
        describes the decay of magnetization following inversion or saturation, following
        the equation:
        M(t) = A*[C*exp(-t/T1) + C]

        This is the simplest decay model, appropriate for:
        - Homogeneous samples
        - Simple liquids
        - Systems with a single relaxation environment

        Args:
            t (ndarray): Time points of the decay curve
            T1 (float): Relaxation time constant
            A (float): Overall amplitude scaling factor
            C (float): Equilibrium offset

        Returns:
            ndarray: Exponential decay values at each time point
        """

        return A * ((B)*np.exp(-t/T1)) + C

    def di_expdec(self, t, T1, T2, A, C, D):
        """
        Two-component exponential decay function for complex T1 relaxation analysis.
        This model
        combines two independent decay processes, following the equation:
        M(t) = A*[C*exp(-t/T1) + D*exp(-t/T2)]

        Useful for analyzing:
        - Two-phase systems
        - Materials with distinct mobility regions
        - Systems with both surface and bulk relaxation
        - Heterogeneous materials with two distinct environments

        Args:
            t (ndarray): Time points of the decay curve
            T1, T2 (float): Relaxation time constants for each component
            A (float): Overall amplitude scaling factor
            C, D (float): Individual component scaling factors

        Returns:
            ndarray: Combined decay values from both components
        """

        return A * (((C)*np.exp(-t/T1)) + ((D)*np.exp(-t/T2)))

    def tri_expdec(self, t, T1, T2, T3, A, C, D, E):
        """
        Three-component exponential decay function for complex T1 relaxation analysis.
        This model
        describes systems with three distinct relaxation processes, following the equation:
        M(t) = A*[C*exp(-t/T1) + D*exp(-t/T2) + E*exp(-t/T3)]

        This sophisticated model is suitable for:
        - Highly heterogeneous materials
        - Multi-phase systems
        - Complex biological samples
        - Materials with multiple distinct chemical environments
        - Systems with multiple mobility regions

        Args:
            t (ndarray): Time points of the decay curve
            T1, T2, T3 (float): Relaxation time constants for each component
            A (float): Overall amplitude scaling factor
            C, D, E (float): Individual component scaling factors

        Returns:
            ndarray: Combined decay values from all three components
        """

        return A * (((C)*np.exp(-t/T1)) + ((D)*np.exp(-t/T2)) + ((E)*np.exp(-t/T3)))

    def stretch_expdec(self, t, T1, A, B):

        """
        Stretched exponential decay function for non-standard relaxation behavior. This model
        accounts for systems with a continuous distribution of relaxation times, following
        a modified Kohlrausch-Williams-Watts (KWW) function:
        M(t) = A*exp[-(t/T1)^B]

        This model is particularly valuable for:
        - Amorphous materials
        - Polymers and glasses
        - Systems with heterogeneous dynamics
        - Materials with complex structural organizations
        - Systems with correlated relaxation processes

        The stretching exponent B characterizes the distribution width of relaxation times,
        with B = 1 corresponding to simple exponential decay and B < 1 indicating
        increasing heterogeneity in the relaxation process.

        Args:
            t (ndarray): Time points of the decay curve
            T1 (float): Characteristic relaxation time constant
            A (float): Overall amplitude scaling factor
            B (float): Stretching exponent, typically between 0 and 1

        Returns:
            ndarray: Stretched exponential decay values at each time point,
                    representing the complex relaxation behavior of the system
        """

        return A * np.exp((-t/T1)**B)
