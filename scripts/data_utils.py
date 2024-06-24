import numpy as np
import matplotlib.pyplot as plt
import scipy

# Simulates a K2-like lightcurve with red-noisy periodic signal with a step, a trend, and an inserted anomaly
def generate_synthetic_lc(
        num_anomalies=1,
        rednoise_amp=1.0, 
        whitenoise_amp=1.0, 
        num_steps=1000, 
        seed=48
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
    anomaly = np.zeros(num_steps)
    anomaly_locs = []
    
    if num_anomalies > 0:
        anomaly_stdev = rng.integers(int(0.001 * num_steps), int(0.01 * num_steps)) # width of anomaly
        anomaly_amp = rng.uniform(-15 * np.std(rednoise), -3 * np.std(rednoise)) # amplitude of anomaly 
        for i in range(num_anomalies):
            anomaly_loc = num_steps * rng.random()
            anomaly_locs.append(anomaly_loc)
            anomaly += anomaly_amp * np.exp(-0.5 * ((x - anomaly_loc) / anomaly_stdev)**2)  # Gaussian-shape anomaly at x0

    # Combine
    y = signal + anomaly + whitenoise + rednoise + step + trend
    y = y - np.mean(y)

    return x, y, anomaly_locs

# Inject an anomaly into an existing lightcurve
def inject_anomaly(y, num_anomalies=1, seed=48):
    assert len(y) > 0, "Lightcurve must have at least one point"
    num_steps = len(y)
    x = np.arange(num_steps) # set of points
    rng = np.random.default_rng(seed=seed)
    
    # Inject anomalies
    anomaly = np.zeros(num_steps)
    anomaly_locs = []
    
    if num_anomalies > 0:
        anomaly_stdev = rng.integers(int(0.001 * num_steps), int(0.01 * num_steps)) # width of anomaly
        anomaly_amp = rng.uniform(-15 * np.std(y), -3 * np.std(y)) # amplitude of anomaly

        for i in range(num_anomalies):
            anomaly_loc = num_steps * rng.random()
            anomaly_locs.append(anomaly_loc)
            anomaly += anomaly_amp * np.exp(-0.5 * ((x - anomaly_loc) / anomaly_stdev)**2)  # Gaussian-shape anomaly at x0

    y = y + anomaly

    return x, y, anomaly_locs