import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks, periodogram, windows
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits

def get_dom_period(lightcurve):
    freqs, power = periodogram(lightcurve)
    peaks, _ = find_peaks(power)
    if len(peaks) == 0:
        print("No peaks found in power spectrum, using shoulder instead.")
        smooth_power = gaussian_filter1d(power, 2)
        slope = np.gradient(smooth_power, freqs)
        shoulder_idx = np.where(slope < 0)[0][0]
        dominant_period = 1 / freqs[shoulder_idx]
        
    else:
        dominant_peak = peaks[np.argmax(power[peaks])]
        dominant_period = 1 / freqs[dominant_peak]

    return dominant_period

def generate_anomaly(
        num_anomalies, 
        lightcurve,
        rng,
        shapes=["gaussian", "saw"],
        period_scale=None, # What ratio of dominant period should be anomaly have
        snr=None, # What signal to noise ratio should the anomaly have
        locs=None
    ):

    # Initialize
    num_steps = len(lightcurve)
    time_steps = np.arange(num_steps)
    anomaly = np.zeros(num_steps)
    anomaly_locs = []

    # Create anomaly with snr if not given
    if snr is None:
        snr = rng.uniform(1, 10) # depth of anomaly 

    # Create anomaly of amplitude corresponding to desired snr (using MAD for noise)
    noise = np.median(np.abs(lightcurve - np.median(lightcurve)))
    signal = snr * noise
    anomaly_amp = -1 * signal

    # Create anomaly period_scale if not given
    if period_scale is None: 
        period_scale = rng.uniform(0.1, 2) # period scaling of anomaly

    # Create anomaly_width from period of peak in power spectrum
    anomaly_period = period_scale * get_dom_period(lightcurve)
    anomaly_width = max(anomaly_period / (2 * np.sqrt(2 * np.log(2))), 1) # minimum value of 1. Note this is the sigma of the anomaly (assuming gaussian)
    anomaly_fwhm = 2.355 * anomaly_width # True for gaussian-shaped anomalies

    for i in range(num_anomalies):
        if locs is not None:
            assert len(locs) == num_anomalies, "Number of anomaly locations must match number of anomalies"
            assert isinstance(locs, list), "Locs must be a list"
            anomaly_loc = locs[i]
        else:
            anomaly_loc = num_steps * rng.random()
            
        anomaly_locs.append(anomaly_loc)

        assert isinstance(shapes, list), "Shapes must be a list"
        shape = rng.choice(shapes)

        if shape == "gaussian":
            # Gaussian-shape anomaly at x0 
            anomaly += anomaly_amp * np.exp(-0.5 * ((time_steps - anomaly_loc) / anomaly_width)**2) 
        elif shape == "saw":
            # Create anomaly that has a quick dip to anomaly_amp, then a slow rise back to 0 based on anomaly_width
            anomaly += anomaly_amp * (1 - np.exp(-np.abs(time_steps - anomaly_loc) / anomaly_width))
            # anomaly += anomaly_amp * (time_steps > anomaly_loc) * np.exp(-0.01 * (time_steps - anomaly_loc))
        else: 
            raise ValueError(f"Invalid shape {shape} for anomaly.")

    return anomaly, anomaly_locs, anomaly_amp, anomaly_fwhm

# Simulates a K2-like lightcurve with red-noisy periodic lightcurve with a step, a trend, and an inserted anomaly
def generate_synthetic_lc_parameterized(
        num_anomalies=1,
        rednoise_amp=1.0, 
        whitenoise_amp=1.0, 
        num_steps=1000, 
        seed=48,
        shapes=["gaussian", "saw"],
        period_scale=None,
        snr=None,
        anomaly_idx=None
    ):

    assert num_steps > 0, "Number of steps must be greater than 0"
    x = np.arange(num_steps)
    rng = np.random.default_rng(seed=seed)

    # Synthetic lightcurve 
    period = 175 + 50 * rng.random() # randomly chosen period of lightcurve
    phase = 2 * np.pi * rng.random() # randomly chosen phase
    amp = 0.9 * rng.random() # randomly chosen amplitude
    lightcurve = amp * np.cos(2 * np.pi * x / period + phase)

    # White noise 
    whitenoise = whitenoise_amp * rng.random(num_steps)

    # Red noise
    redscale = rng.integers(5, 15) # correlation time scale of red noise
    rednoise = np.convolve(rng.random(2 * num_steps), windows.gaussian(int(4 * redscale), redscale))
    x1 = int(len(rednoise)/2) - int(num_steps/2)
    x2 = x1 + num_steps
    rednoise = rednoise[x1:x2]
    rednoise = rednoise * rednoise_amp / np.std(rednoise)

    # Step parameters
    step_loc = num_steps * rng.random() # location of step
    step_amp = rng.uniform(-5 * np.std(rednoise), -1 * np.std(rednoise)) # amplitude of anomaly 
    step_width = rng.integers(int(0.001 * num_steps), int(0.01 * num_steps))
    step = step_amp * (x > step_loc) * (x < (step_loc + step_width))

    # Trend parameters
    slope = 0.001 - 0.002 * rng.random() # slope of trend
    trend = slope * (x - num_steps/2)

    # Inject anomalies
    anomaly, anomaly_locs, anomaly_amp, anomaly_fwhm = generate_anomaly(num_anomalies, rednoise, rng, shapes, period_scale, snr, anomaly_idx)

    # Combine
    y = lightcurve + anomaly + whitenoise + rednoise + step + trend
    y = y - np.mean(y)

    return x, y, anomaly_locs, anomaly_amp, anomaly_fwhm

# Inject an anomaly into an existing lightcurve
def inject_anomaly(
        y, 
        num_anomalies=1, 
        seed=48, 
        shapes=["gaussian", "saw"],
        period_scale=None,
        snr=None,
        anomaly_idx=None
    ):

    assert len(y) > 0, "Lightcurve must have at least one point"
    x = np.arange(len(y))
    rng = np.random.default_rng(seed=seed)
    
    # Inject anomalies
    anomaly, anomaly_locs, anomaly_amp, anomaly_fwhm = generate_anomaly(num_anomalies, y, rng, shapes, period_scale, snr, anomaly_idx)

    y = y + anomaly

    return x, y, anomaly_locs, anomaly_amp, anomaly_fwhm

# Load and clean K2 data
def load_k2_data(filename):
    fits_file = fits.open(filename)

    # Load data
    data = fits_file[1].data
    time = np.array(data['TIME'])
    pdc_flux = np.array(data['PDCSAP_FLUX'])*1.
    pdc_err = np.array(data['PDCSAP_FLUX_ERR'])*1.

    # Set x, y, and error
    x = time
    y = pdc_flux / np.nanmedian(pdc_flux)
    y_err = pdc_err / np.nanmedian(pdc_flux)

    # Clean
    clean = ((y < 1.5) & (y > 0.5) & (y_err > 0.))
    x = x[clean]
    y = y[clean]
    y_err = np.abs(y_err[clean]) 

    # Shift to start at time 0
    x = x - np.min(x)

    # Bin to 30-minute cadence
    num_bins = int(np.floor((np.max(x) - np.min(x)) * 48) + 1) # 48 bins per day
    x_bins = np.min(x) + np.arange(num_bins + 1) / 48.
    num_binned, bin_edges = np.histogram(x, bins= x_bins)
    num_binned = np.array(num_binned)
    y_binned, bin_edges = np.histogram(x, bins = x_bins, weights = y)
    var_binned, bin_edges = np.histogram(x, bins = x_bins, weights= 1 / y_err**2)

    # Where var_binned is 0, set to min value of var_binned
    var_binned[var_binned == 0] = np.min(var_binned[var_binned > 0])
    
    y_err_binned = 1 / np.sqrt(np.array(var_binned))
    y_binned = np.array(y_binned)
    y_binned = y_binned / (num_binned + 0.001)
    x_binned = x_bins[0:num_bins] + (x_bins[1] - x_bins[0]) / 2. 
    x = x_binned
    y = y_binned
    y_err = y_err_binned

    # Clean
    clean = ((y > 0.) & ~np.isnan(y_err) & (y_err > 0.))
    x = x[clean]
    y = y[clean]
    y_err = y_err[clean]

    assert len(x) == len(y) == len(y_err), 'Lengths of x, y, and y_err must be the same'

    return x, y, y_err