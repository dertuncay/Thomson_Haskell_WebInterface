# Thomson-Haskell Transfer Function Web Interface

This project implements a web-based interface for calculating the Thomson-Haskell transfer function for soil columns and building models, as well as estimating drift/displacement from seismic waveforms.

## Features

- **Thomson-Haskell Transfer Function Calculator**: Computes the transfer function $TF(f) = u_{top}/u_{bottom}$ for N-layered systems including damping.
- **Seismic Waveform Processing**: Upload `.mseed` files, apply corrections (detrend, taper, filter), and integration/differentiation.
- **Drift Estimation**: Calculates building drift by deconvolving the transfer function from the bottom motion to estimate top motion, and subtracting them.
- **Interactive Web UI**: Built with Flask, featuring interactive plots and downloadable results.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `flask`
- `obspy`

To install dependencies:
```bash
pip install -r requirements.txt
```

## Library Usage

You can use the core logic independently of the web interface.

### Example: Calculating Transfer Function

```python
import numpy as np
import matplotlib.pyplot as plt
from thomson_haskell import thomson_haskell_transfer_function

# 1. Define Layer Properties (Top to Bottom)
# Example: 1 Layer of Soil over Half-space
# Layer 1
h1 = 30.0    # Thickness [m]
vs1 = 200.0  # Shear Wave Velocity [m/s]
rho1 = 1800.0 # Density [kg/m^3]
qs1 = 20.0   # Quality Factor (Damping)

# Half-space (Bottom)
vs_hs = 800.0
rho_hs = 2400.0
qs_hs = 100.0

# Combine into lists
h = [h1]                 # Thicknesses (N layers)
vs = [vs1, vs_hs]        # Velocities (N+1 layers)
rho = [rho1, rho_hs]     # Densities (N+1 layers)
qs = [qs1, qs_hs]        # Quality Factors (N+1 layers)

# 2. Define Frequencies
freqs = np.linspace(0.1, 20, 200) # 0.1 to 20 Hz

# 3. Calculate Transfer Function
tf = thomson_haskell_transfer_function(freqs, h, vs, rho, qs)

# 4. Plot Result
plt.figure()
plt.loglog(freqs, np.abs(tf), 'k')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplification |TF(f)|')
plt.title('1-Layer Soil Transfer Function')
plt.grid(True, which="both")
plt.show()
```

## Running the Web App

```bash
python app.py
```
Open your browser at `http://localhost:5000`.
