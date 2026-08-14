import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import nmrglue as ng
import os
import tempfile
from mrsimulator import signal_processor as sp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.relaxometrynmr.core import T1Functions
import matplotlib.pyplot as plt

class TestT1Functions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp(prefix='t1_test_')
        self.t1_funcs = T1Functions(self.temp_dir)
        
        # Create mock data for testing
        self.mock_fid = (np.random.random(1024) + 1j * np.random.random(1024)).astype(np.complex128)
        self.mock_spectrum = (np.random.random(1024) + 1j * np.random.random(1024)).astype(np.complex128)
        self.mock_vd_list = np.array([0.001, 0.01, 0.1, 1.0, 2.0, 5.0])

    def tearDown(self):
        """Clean up after each test."""
        plt.close('all')
        if os.path.exists(self.temp_dir):
            for f in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, f))
                except (OSError, PermissionError):
                    pass
            try:
                os.rmdir(self.temp_dir)
            except (OSError, PermissionError):
                pass

    @patch('nmrglue.bruker.read')
    @patch('nmrglue.bruker.guess_udic')
    def test_read_and_convert_bruker_data(self, mock_guess_udic, mock_read):
        """Test reading and converting Bruker data."""
        # Create a properly structured udic with all required fields
        mock_udic = {
            "ndim": 2,
            0: {
                "encoding": "states",
                "sw": 50000,
                "obs": 400,
                "car": 100.0,
                "size": 1024,
                "label": "F2",
                "complex": True,
                "time": True,
                "freq": True
            },
            1: {
                "encoding": "states",
                "sw": 50000,
                "obs": 400,
                "car": 100.0,
                "size": 256,
                "label": "F1",
                "complex": True,
                "time": True,
                "freq": True
            }
        }
        mock_dic = {"ndim": 2}
        mock_data = np.zeros((256, 1024))
        
        mock_read.return_value = (mock_dic, mock_data)
        mock_guess_udic.return_value = mock_udic
        
        # Create the delay list file with proper path handling
        vdlist_path = os.path.join(self.temp_dir, "vdlist")
        np.savetxt(vdlist_path, self.mock_vd_list)
        
        # Patch the file path property to use temp_dir
        with patch.object(self.t1_funcs, 'file_path', self.temp_dir + os.path.sep):
            spectra, vd_list, csdm_ds = self.t1_funcs.read_and_convert_bruker_data(save_nmrpipe=False)
            
            self.assertIsInstance(vd_list, np.ndarray)
            self.assertEqual(len(vd_list), len(self.mock_vd_list))
            self.assertTrue(np.allclose(vd_list, self.mock_vd_list))
        self.assertIsInstance(vd_list, np.ndarray)
        self.assertEqual(len(vd_list), len(self.mock_vd_list))
                
    def test_zero_fill(self):
        """Test zero-filling functionality."""
        test_data = np.ones(100)
        new_len = 256
        filled_data = self.t1_funcs.zero_fill(test_data, new_len)
        self.assertEqual(len(filled_data), new_len)
        self.assertEqual(np.sum(filled_data[100:]), 0)

    def test_zero_order_phasing(self):
        """Test zero-order phase correction."""
        test_data = (np.ones(100) + 1j * np.zeros(100)).astype(np.complex128)
        phased_data = self.t1_funcs.zero_order_phasing(test_data, 90)
        self.assertTrue(np.allclose(np.real(phased_data), 0, atol=1e-10))
        self.assertTrue(np.allclose(np.imag(phased_data), 1, atol=1e-10))

    def test_first_order_phasing(self):
        """Test first-order phase correction."""
        test_data = (np.ones(100) + 1j * np.zeros(100)).astype(np.complex128)
        phased_data = self.t1_funcs.first_order_phasing(test_data, 45)
        self.assertTrue(np.all(np.abs(phased_data) - 1 < 1e-10))

    def test_process_spectrum(self):
        """Test spectrum processing functionality."""
        # Create a CSDM-like mock object
        mock_spectrum = MagicMock()
        mock_dimensions = [MagicMock()]
        mock_dimensions[0].coordinates = MagicMock()
        mock_dimensions[0].coordinates.value = np.arange(1024)
        mock_spectrum.dimensions = mock_dimensions
        mock_spectrum.shape = [1024]
        # Create the array data
        array_data = np.zeros(1024, dtype=np.complex128)
        mock_spectrum.__array__ = lambda *args: array_data
        
        with patch.object(sp.SignalProcessor, 'apply_operations') as mock_apply:
            # Return a similar mock object
            return_mock = MagicMock()
            return_mock.dimensions = mock_dimensions
            return_mock.shape = [1024]
            return_mock.__array__ = lambda *args: array_data
            mock_apply.return_value = return_mock
            
            result = self.t1_funcs.process_spectrum(
                mock_spectrum, 
                fwhm=10, 
                zero_fill_factor=2,
                ph0=0,
                ph1=0
            )
            mock_apply.assert_called_once()

    def test_integrate_spectrum_region(self):
        """Test spectrum integration functionality."""
        x = np.linspace(-10, 10, 1000)
        y = np.exp(-x**2)
        
        mock_spectrum = MagicMock()
        mock_dim = MagicMock()
        mock_dim.coordinates.value = x
        mock_spectrum.dimensions = [mock_dim]
        
        mock_var = MagicMock()
        mock_comp = MagicMock()
        mock_comp.real = y
        mock_var.components = [mock_comp]
        mock_spectrum.dependent_variables = [mock_var]
        
        trapz, simps, x_reg, y_reg, uncert = self.t1_funcs.integrate_spectrum_region(
            mock_spectrum, -2, 2
        )
        self.assertLess(abs(trapz - simps), 1e-3)

    def test_plot_spectra_and_zoomed_regions(self):
        """Test plotting functionality."""
        exp_spectra = []
        x_regions = []
        y_regions = []
        
        for _ in range(2):
            mock_spectrum = MagicMock()
            # Create proper numpy array for real attribute
            mock_spectrum.real = np.zeros(100)
            mock_spectrum.dependent_variables = [MagicMock()]
            mock_comp = MagicMock()
            mock_comp.max.return_value = 1.0
            mock_spectrum.dependent_variables[0].components = [mock_comp]
            # Add array-like behavior
            mock_spectrum.__array__ = lambda *args: mock_spectrum.real
            exp_spectra.append(mock_spectrum)
            x_regions.append(np.linspace(-5, 5, 100))
            y_regions.append(np.zeros(100))
        
        intensities = self.t1_funcs.plot_spectra_and_zoomed_regions(
            exp_spectra, x_regions, y_regions, -5, 5
        )
        self.assertEqual(len(intensities), 2)
        
    def test_lorentzian_fit(self):
        """Test Lorentzian fitting functionality."""
        x = np.linspace(-30, 30, 300)
        A_true, x0_true, gamma_true, offset_true = 50.0, 2.0, 5.0, 1.0
        y = (A_true / np.pi) * (gamma_true / (gamma_true**2 + (x - x0_true)**2)) + offset_true

        prefix = self.temp_dir + os.path.sep
        result = self.t1_funcs.lorentzian_fit(
            'sample', x, y,
            plot_filepath=prefix, params_filepath=prefix, data_filepath=prefix
        )

        self.assertAlmostEqual(result['A'], A_true, places=2)
        self.assertAlmostEqual(result['x0'], x0_true, places=2)
        self.assertAlmostEqual(result['gamma'], gamma_true, places=2)
        self.assertAlmostEqual(result['offset'], offset_true, places=2)
        self.assertAlmostEqual(result['fwhm_hz'], gamma_true * 67.8 * 2, places=1)

        self.assertTrue(os.path.exists(prefix + 'lorentzian_fit.svg'))
        self.assertTrue(os.path.exists(prefix + 'fitted_fwhm_params.csv'))
        self.assertTrue(os.path.exists(prefix + 'sample.csv'))

    def test_is_pdata_path_raw_experiment(self):
        """A folder with a raw fid/ser file should be detected as raw, not pdata."""
        open(os.path.join(self.temp_dir, 'fid'), 'a').close()
        self.assertFalse(self.t1_funcs._is_pdata_path(self.temp_dir))

    def test_is_pdata_path_procno_dir(self):
        """A folder containing 1r (or 2rr) should be detected as a pdata procno dir."""
        open(os.path.join(self.temp_dir, '1r'), 'a').close()
        self.assertTrue(self.t1_funcs._is_pdata_path(self.temp_dir))

    def test_is_pdata_path_experiment_with_pdata_subfolder(self):
        """An experiment root with only a pdata subfolder (no fid/ser) is pdata-only."""
        os.makedirs(os.path.join(self.temp_dir, 'pdata', '1'))
        self.assertTrue(self.t1_funcs._is_pdata_path(self.temp_dir))

    def test_resolve_pdata_path_from_procno_dir(self):
        """Resolving a path that already points at a procno dir returns it unchanged."""
        open(os.path.join(self.temp_dir, '1r'), 'a').close()
        with patch.object(self.t1_funcs, 'file_path', self.temp_dir):
            resolved = self.t1_funcs._resolve_pdata_path(proc_no=1)
        self.assertEqual(resolved, self.temp_dir)

    def test_resolve_pdata_path_from_experiment_root(self):
        """Resolving an experiment root appends pdata/<proc_no>."""
        with patch.object(self.t1_funcs, 'file_path', self.temp_dir):
            resolved = self.t1_funcs._resolve_pdata_path(proc_no=1)
        self.assertEqual(resolved, os.path.join(self.temp_dir, 'pdata', '1'))

    @patch('nmrglue.bruker.read_pdata')
    @patch('nmrglue.bruker.guess_udic')
    def test_read_processed_bruker_data(self, mock_guess_udic, mock_read_pdata):
        """Test reading processed (pdata) Bruker data directly."""
        mock_udic = {
            "ndim": 2,
            0: {
                "encoding": "states", "sw": 50000, "obs": 400, "car": 100.0,
                "size": 6, "label": "F1", "complex": False, "time": True, "freq": False
            },
            1: {
                "encoding": "states", "sw": 50000, "obs": 400, "car": 100.0,
                "size": 1024, "label": "F2", "complex": True, "time": False, "freq": True
            }
        }
        mock_dic = {"ndim": 2}
        mock_data = np.zeros((6, 1024), dtype=np.complex128)

        mock_read_pdata.return_value = (mock_dic, mock_data)
        mock_guess_udic.return_value = mock_udic

        pdata_dir = os.path.join(self.temp_dir, 'pdata', '1')
        os.makedirs(pdata_dir)
        open(os.path.join(pdata_dir, '1r'), 'a').close()

        vdlist_path = os.path.join(self.temp_dir, "vdlist")
        np.savetxt(vdlist_path, self.mock_vd_list)

        with patch.object(self.t1_funcs, 'file_path', self.temp_dir):
            spectra, vd_list, ppm, dic = self.t1_funcs.read_processed_bruker_data(proc_no=1)

        self.assertEqual(len(spectra), 6)
        self.assertIsInstance(vd_list, np.ndarray)
        self.assertTrue(np.allclose(vd_list, self.mock_vd_list))
        self.assertEqual(len(ppm), 1024)
        self.assertEqual(dic, mock_dic)

    def test_load_relaxation_series_dispatches_to_processed(self):
        """load_relaxation_series should route pdata-only folders to read_processed_bruker_data."""
        os.makedirs(os.path.join(self.temp_dir, 'pdata', '1'))

        with patch.object(self.t1_funcs, 'read_processed_bruker_data') as mock_read_processed:
            mock_read_processed.return_value = ([self.mock_spectrum], self.mock_vd_list, np.arange(1024), {})
            source, spectra, vd_list, extra = self.t1_funcs.load_relaxation_series()

        self.assertEqual(source, "processed")
        mock_read_processed.assert_called_once()
        self.assertTrue(np.allclose(vd_list, self.mock_vd_list))

    def test_load_relaxation_series_dispatches_to_raw(self):
        """load_relaxation_series should route raw fid/ser folders to read_and_convert_bruker_data."""
        open(os.path.join(self.temp_dir, 'fid'), 'a').close()

        with patch.object(self.t1_funcs, 'read_and_convert_bruker_data') as mock_read_raw:
            mock_read_raw.return_value = ([self.mock_spectrum], self.mock_vd_list, MagicMock())
            source, spectra, vd_list, extra = self.t1_funcs.load_relaxation_series(save_nmrpipe=False)

        self.assertEqual(source, "raw")
        mock_read_raw.assert_called_once_with(save_nmrpipe=False)
        self.assertTrue(np.allclose(vd_list, self.mock_vd_list))

    def test_mono_satrec_func(self):
        """Test mono-exponential saturation recovery function."""
        t = np.linspace(0, 20, 1000)
        M0, T1, A, B = 1.0, 2.0, 1.0, 0.0
        result = self.t1_funcs.mono_satrec_func(t, M0, T1, A, B)
        
        # Test dimensions
        self.assertEqual(len(result), len(t))
        
        # Test initial and final values
        self.assertAlmostEqual(result[0], B, places=3)  # Initial value
        self.assertAlmostEqual(result[-1], A*M0 + B, places=3)  # Final value

    def test_di_satrec_func(self):
        """Test bi-exponential saturation recovery function."""
        t = np.linspace(0, 20, 1000)
        result = self.t1_funcs.di_satrec_func(t, M0=0.7, T1=2.0, A=1.0, M1=0.3, T2=0.5)
        self.assertEqual(len(result), len(t))
        self.assertTrue(np.all(result >= 0))

    def test_tri_satrec_func(self):
        """Test tri-exponential saturation recovery function."""
        t = np.linspace(0, 20, 1000)
        result = self.t1_funcs.tri_satrec_func(
            t, M0=0.5, T1=2.0, A=1.0, M1=0.3, T2=0.5, M2=0.2, T3=0.1
        )
        self.assertEqual(len(result), len(t))
        self.assertTrue(np.all(result >= 0))

    def test_stretch_t1_exponential(self):
        """Test stretched exponential function."""
        t = np.linspace(0, 10, 100)
        
        # Test normal behavior
        result = self.t1_funcs.stretch_t1_exponential(t, T1_star=2.0, c=0.5)
        self.assertEqual(len(result), len(t))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        
        # Test with c=1 (should reduce to normal exponential)
        result_normal = self.t1_funcs.stretch_t1_exponential(t, T1_star=2.0, c=1.0)
        result_stretched = self.t1_funcs.stretch_t1_exponential(t, T1_star=2.0, c=0.5)
        self.assertTrue(np.any(np.not_equal(result_normal, result_stretched)))

    def test_mono_expdec(self):
        """Test mono-exponential decay function."""
        t = np.linspace(0, 20, 100)  # Extended time range
        
        # Test normal behavior
        result = self.t1_funcs.mono_expdec(t, T1=2.0, A=1.0, B=1.0, C=0.0)
        self.assertEqual(len(result), len(t))
        
        # Test decay to baseline with relaxed tolerance
        self.assertAlmostEqual(result[-1], 0.0, places=1)
        
        # Test with offset
        result = self.t1_funcs.mono_expdec(t, T1=2.0, A=1.0, B=1.0, C=0.5)
        self.assertAlmostEqual(result[-1], 0.5, places=1)

    def test_di_expdec(self):
        """Test bi-exponential decay function."""
        t = np.linspace(0, 20, 100)
        result = self.t1_funcs.di_expdec(t, T1=2.0, T2=0.5, A=1.0, C=0.7, D=0.3)
        self.assertEqual(len(result), len(t))
        self.assertTrue(result[-1] < 0.1)

    def test_tri_expdec(self):
        """Test tri-exponential decay function."""
        t = np.linspace(0, 20, 100)
        result = self.t1_funcs.tri_expdec(
            t, T1=2.0, T2=0.5, T3=0.1, A=1.0, C=0.5, D=0.3, E=0.2
        )
        self.assertEqual(len(result), len(t))
        self.assertTrue(result[-1] < 0.1)

    def test_tri_expdec(self):
        """Test tri-exponential decay function."""
        t = np.linspace(0, 20, 100)
        result = self.t1_funcs.tri_expdec(
            t, T1=2.0, T2=0.5, T3=0.1, A=1.0, C=0.5, D=0.3, E=0.2
        )
        self.assertEqual(len(result), len(t))
        self.assertTrue(result[-1] < 0.1)
        
if __name__ == '__main__':
    
    cov = coverage.Coverage()
    cov.start()
    
    unittest.main(verbosity=2)
    
    cov.stop()
    
    cov.save()
    
    cov.html_report(directory='coverage_html')