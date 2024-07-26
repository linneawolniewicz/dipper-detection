import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits

def generate_anomaly(
        time_steps,
        num_anomalies, 
        signal,
        rng,
        shapes=["gaussian", "saw"],
        stdev=None,
        amp=None,
        locs=None
    ):

    # Initialize
    num_steps = len(time_steps)
    anomaly = np.zeros(num_steps)
    anomaly_locs = []

    # Create anomalies
    if stdev is not None: 
        anomaly_stdev = stdev # width of anomaly
    else:
        anomaly_stdev = rng.integers(int(0.001 * num_steps), int(0.01 * num_steps)) # width of anomaly

    if amp is not None:
        anomaly_amp = amp # amplitude of anomaly
    else:
        anomaly_amp = rng.uniform(-15 * np.std(signal), -3 * np.std(signal)) # amplitude of anomaly 

    for i in range(num_anomalies):
        if locs is not None:
            assert len(locs) == num_anomalies, "Number of anomaly locations must match number of anomalies"
            anomaly_loc = locs[i]
        else:
            anomaly_loc = num_steps * rng.random()
            
        anomaly_locs.append(anomaly_loc)
        shape = rng.choice(shapes)

        if shape == "gaussian":
            anomaly += anomaly_amp * np.exp(-0.5 * ((time_steps - anomaly_loc) / anomaly_stdev)**2)  # Gaussian-shape anomaly at x0
        elif shape == "saw":
            # Create anomaly that has a quick dip to anomaly_amp, then a slow rise back to 0
            anomaly += anomaly_amp * (time_steps > anomaly_loc) * np.exp(-0.01 * (time_steps - anomaly_loc))
        else: 
            raise ValueError(f"Invalid shape {shape} for anomaly.")

    return anomaly, anomaly_locs

# Simulates a K2-like lightcurve with red-noisy periodic signal with a step, a trend, and an inserted anomaly
def generate_synthetic_lc(
        num_anomalies=1,
        rednoise_amp=1.0, 
        whitenoise_amp=1.0, 
        num_steps=1000, 
        seed=48,
        shapes=["gaussian", "saw"],
        anomaly_stdev=None,
        anomaly_amp=None,
        anomaly_idx=None
    ):

    assert num_steps > 0, "Number of steps must be greater than 0"
    
    rng = np.random.default_rng(seed=seed)
    x = np.arange(num_steps) # set of points

    # Synthetic signal 
    period = 175 + 50 * rng.random() # randomly chosen period of signal
    phase = 2 * np.pi * rng.random() # randomly chosen phase
    amp = 0.9 * rng.random() # randomly chosen amplitude
    signal = amp * np.cos(2 * np.pi * x / period + phase)

    # White noise 
    whitenoise = whitenoise_amp * rng.random(num_steps)

    # Red noise
    redscale = rng.integers(5, 15) # correlation time scale of red noise
    rednoise = np.convolve(rng.random(2 * num_steps), scipy.signal.windows.gaussian(int(4 * redscale), redscale))
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
    anomaly, anomaly_locs = generate_anomaly(x, num_anomalies, rednoise, rng, shapes, anomaly_stdev, anomaly_amp, anomaly_idx)

    # Combine
    y = signal + anomaly + whitenoise + rednoise + step + trend
    y = y - np.mean(y)

    return x, y, anomaly_locs

# Inject an anomaly into an existing lightcurve
def inject_anomaly(
        y, 
        num_anomalies=1, 
        seed=48, 
        shapes=["gaussian", "saw"],
        anomaly_stdev=None,
        anomaly_amp=None,
        anomaly_idx=None
    ):

    assert len(y) > 0, "Lightcurve must have at least one point"
    num_steps = len(y)
    x = np.arange(num_steps) # set of points
    rng = np.random.default_rng(seed=seed)
    
    # Inject anomalies
    shape = "gaussian"
    anomaly, anomaly_locs = generate_anomaly(x, num_anomalies, y, rng, shapes, anomaly_stdev, anomaly_amp, anomaly_idx)

    y = y + anomaly

    return x, y, anomaly_locs

# Load and clean K2 data
def load_k2_data(filename):
    fits_file = fits.open(filename)

    # Load data
    data = fits_file[1].data
    time = np.array(data['TIME'])
    pdc_flux = np.array(data['PDCSAP_FLUX'])
    pdc_err = np.array(data['PDCSAP_FLUX_ERR'])

    # Set x, y, and error
    x = time
    y = pdc_flux / np.nanmedian(pdc_flux)
    y_err = pdc_err / np.nanmedian(pdc_flux)

    # Clean
    clean = (y_err > 0.)
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
    y_err_binned = 1 / np.sqrt(np.array(var_binned))
    y_binned = np.array(y_binned)
    y_binned = y_binned / (num_binned + 0.001)
    x_binned = x_bins[0:num_bins] + (x_bins[1] - x_bins[0]) / 2. 
    x = x_binned
    y = y_binned
    y_err = y_err_binned

    # Clean
    clean = ((y > 0.) & ~np.isnan(y_err))
    x = x[clean]
    y = y[clean]
    y_err = y_err[clean]

    assert len(x) == len(y) == len(y_err), 'Lengths of x, y, and y_err must be the same'

    return x, y, y_err