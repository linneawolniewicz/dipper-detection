# imports
import unittest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from dipper.data_utils.utils import *

class TestGenerateAnomaly(unittest.TestCase):
    print("Testing generate_anomaly")

    def setUp(self):
        self.lightcurve = np.random.uniform(0, 1, 1000)  # Example signal
        self.rng = np.random.default_rng(seed=42)

    def test_generate_single_gaussian_anomaly(self):
        anomaly, anomaly_locs, anomaly_depth, anomaly_fwhm = generate_anomaly(
            num_anomalies=1,
            lightcurve=self.lightcurve,
            rng=self.rng,
            shapes=["gaussian"]
        )
        
        # Check that the anomaly has the same length as the signal
        self.assertEqual(len(anomaly), len(self.lightcurve))
        
        # Check that exactly one anomaly was generated
        self.assertEqual(len(anomaly_locs), 1)

        # Check that the anomaly depth is negative
        self.assertLess(anomaly_depth, 0)
        
        # Check that the anomaly FWHM is positive
        self.assertGreater(anomaly_fwhm, 0)

    def test_generate_multiple_anomalies(self):
        num_anomalies = 3
        anomaly, anomaly_locs, anomaly_depth, anomaly_fwhm = generate_anomaly(
            num_anomalies=num_anomalies,
            lightcurve=self.lightcurve,
            rng=self.rng
        )
        
        # Check that the correct number of anomalies were generated
        self.assertEqual(len(anomaly_locs), num_anomalies)
        
        # Check that the anomaly locations are within the signal range
        for loc in anomaly_locs:
            self.assertGreaterEqual(loc, 0)
            self.assertLess(loc, len(self.lightcurve))

    def test_generate_anomaly_with_fixed_locs(self):
        locs = [100, 300, 500]
        anomaly, anomaly_locs, anomaly_depth, anomaly_fwhm = generate_anomaly(
            num_anomalies=len(locs),
            lightcurve=self.lightcurve,
            rng=self.rng,
            locs=locs
        )
        
        # Check that the anomaly locations match the provided locations
        self.assertEqual(anomaly_locs, locs)

    def test_invalid_shape_raises_error(self):
        with self.assertRaises(ValueError):
            generate_anomaly(
                num_anomalies=1,
                lightcurve=self.lightcurve,
                rng=self.rng,
                shapes=["invalid_shape"]
            )

    def test_no_peaks_in_power_spectrum(self):
        # Signal with no clear peaks (e.g., uniform noise)
        lightcurve = np.random.uniform(0, 1, 1000)
        anomaly, anomaly_locs, anomaly_depth, anomaly_fwhm = generate_anomaly(
            num_anomalies=1,
            lightcurve=lightcurve,
            rng=self.rng
        )
        
        # Check that an anomaly is still generated
        self.assertEqual(len(anomaly), len(lightcurve))
        self.assertGreater(len(anomaly_locs), 0)
        self.assertGreater(anomaly_fwhm, 0)


class TestGenerateSyntheticLC(unittest.TestCase):
    print("Testing generate_synthetic_lc")

    def test_generate_synthetic_lc_basic(self):
        x, y, anomaly_locs, anomaly_amp, anomaly_fwhm = generate_synthetic_lc()
        
        # Check that the length of x and y match the number of steps
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 1000)

        # Check that at least one anomaly location is returned
        self.assertGreater(len(anomaly_locs), 0)

        # Check that the anomaly amplitude is negative
        self.assertLess(anomaly_amp, 0)

        # Check that the anomaly FWHM is positive
        self.assertGreater(anomaly_fwhm, 0)

    def test_generate_synthetic_lc_with_params(self):
        x, y, anomaly_locs, anomaly_amp, anomaly_fwhm = generate_synthetic_lc(
            num_anomalies=3,
            rednoise_amp=2.0,
            whitenoise_amp=0.5,
            num_steps=500
        )
        
        # Check that the length of x and y match the specified number of steps
        self.assertEqual(len(x), 500)
        self.assertEqual(len(y), 500)

        # Check that the correct number of anomalies were generated
        self.assertEqual(len(anomaly_locs), 3)

    def test_generate_synthetic_lc_invalid_num_steps(self):
        with self.assertRaises(AssertionError):
            generate_synthetic_lc(num_steps=0)

    def test_generate_synthetic_lc_fixed_anomaly_locations(self):
        anomaly_idx = [100, 300, 700]
        x, y, anomaly_locs, anomaly_amp, anomaly_fwhm = generate_synthetic_lc(
            num_anomalies=len(anomaly_idx),
            anomaly_idx=anomaly_idx
        )
        
        # Check that the anomaly locations match the provided locations
        self.assertEqual(anomaly_locs, anomaly_idx)


class TestInjectAnomaly(unittest.TestCase):
    print("Testing inject_anomaly")

    def setUp(self):
        self.lightcurve = np.random.uniform(0, 1, 1000)

    def test_inject_anomaly_basic(self):
        x, y, anomaly_locs, anomaly_amp, anomaly_fwhm = inject_anomaly(self.lightcurve)
        
        # Check that the length of x and y match the original signal
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), 1000)

        # Check that at least one anomaly location is returned
        self.assertGreater(len(anomaly_locs), 0)

        # Check that the anomaly amplitude is negative
        self.assertLess(anomaly_amp, 0)

        # Check that the anomaly FWHM is positive
        self.assertGreater(anomaly_fwhm, 0)

    def test_inject_anomaly_multiple(self):
        x, y, anomaly_locs, anomaly_amp, anomaly_fwhm = inject_anomaly(self.lightcurve, num_anomalies=3)
        
        # Check that the correct number of anomalies were generated
        self.assertEqual(len(anomaly_locs), 3)

    def test_inject_anomaly_invalid_signal(self):
        with self.assertRaises(AssertionError):
            inject_anomaly([])

    def test_inject_anomaly_fixed_locations(self):
        anomaly_idx = [200, 400, 600]
        x, y, anomaly_locs, anomaly_amp, anomaly_fwhm = inject_anomaly(
            self.lightcurve, 
            num_anomalies=len(anomaly_idx),
            anomaly_idx=anomaly_idx
        )
        
        # Check that the anomaly locations match the provided locations
        self.assertEqual(anomaly_locs, anomaly_idx)


class TestLoadK2Data(unittest.TestCase):
    print("Testing load_k2_data")
    @patch('astropy.io.fits.open')
    def test_load_k2_data(self, mock_fits_open):
        # Create a mock FITS file
        mock_fits = MagicMock()
        mock_fits.__enter__.return_value = mock_fits
        mock_fits_open.return_value = mock_fits
        
        # Mock the data
        mock_fits[1].data = {
            'TIME': np.arange(100),
            'PDCSAP_FLUX': np.random.normal(1, 0.1, 100),
            'PDCSAP_FLUX_ERR': np.random.normal(0.1, 0.01, 100)
        }

        x, y, y_err = load_k2_data("mock_file.fits")
        
        # Check that x, y, and y_err are all of the same length
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), len(y_err))

        # Check that the values are cleaned and normalized
        self.assertTrue(np.all(y > 0))
        self.assertTrue(np.all(y_err > 0))

    @patch('astropy.io.fits.open')
    def test_load_k2_data_invalid_file(self, mock_fits_open):
        mock_fits_open.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_k2_data("nonexistent_file.fits")

if __name__ == '__main__':
    unittest.main()
