import numpy as np


def thomson_haskell_transfer_function(freqs, h, vs, rho, qs):
	"""
	Generalized transfer function TF_{u-u}(f) for N-layered soil over a half-space,
	including damping via complex velocities.

	Parameters:
	-----------
	freqs : array_like
		Frequencies in Hz
	h : list or np.ndarray
		Thicknesses of the N layers [m]
	vs : list or np.ndarray
		Shear wave velocities of N+1 layers (last is half-space) [m/s]
	rho : list or np.ndarray
		Densities of N+1 layers (last is half-space) [kg/m³]
	qs : list or np.ndarray
		Quality factors of N+1 layers (last is half-space)

	Returns:
	--------
	tf : np.ndarray
		Complex transfer function TF_{u-u}(f) at each frequency
	"""

	omega = 2 * np.pi * freqs
	i = 1j
	Nf = len(freqs)
	N = len(h)  # Number of layers

	# Complex shear velocities and moduli
	vs_star = np.array(vs) / (1 + i / (2 * np.array(qs)))
	mu_star = np.array(rho) * vs_star ** 2
	Z = np.sqrt(mu_star * np.array(rho))

	tf = np.zeros(Nf, dtype=complex)

	for k, w in enumerate(omega):
		A = np.identity(2, dtype=complex)

		for j in range(N):
			rj = w * h[j] / vs_star[j]
			Zj = Z[j]

			cos_rj = np.cos(rj)
			sin_rj = np.sin(rj)

			Aj = np.array([
				[cos_rj, sin_rj / Zj],
				[-Zj * sin_rj, cos_rj]
			], dtype=complex)

			A = A @ Aj

		tf[k] = 1.0 / A[0, 0]

	return tf


def q_from_damping(damping):
    """
    Calculate Quality Factor (Q) from Damping ratio.
    Q = 1 / (2 * damping)
    """
    if damping == 0:
        return float('inf')
    return 1.0 / (2.0 * damping)


def damping_from_q(q):
    """
    Calculate Damping ratio from Quality Factor (Q).
    damping = 1 / (2 * Q)
    """
    if q == 0:
        return float('inf') # Or handle error
    return 1.0 / (2.0 * q)


'''
Example
'''
if __name__ == "__main__":
    building_height = 10 # meter
    building_shear  = 200 # m/s
    building_density= 2400 # kg/cm^3
    building_quality= 10
    dummy_shear  = 800 # m/s
    dummy_density= 100 # kg/cm^3
    dummy_quality= 500
    h   = [building_height]
    vs  = [building_shear, dummy_shear]
    rho = [building_density, dummy_density]
    qs  = [building_quality, dummy_quality]

    # Frequencies
    freqs = np.linspace(0.1, 50, 500)

    # Transfer Function
    tf = thomson_haskell_transfer_function(freqs, h,vs,rho,qs)

    # Plotting
    import matplotlib.pyplot as plt
    plt.loglog(freqs, np.abs(tf),c='k')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Transfer Function ($\dfrac{u_{top}}{u_{bottom}}$)')
    plt.grid('on','both')
    plt.show()
