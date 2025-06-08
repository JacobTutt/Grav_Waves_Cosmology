import time
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal
import bilby


from dynesty import NestedSampler
from dynesty import plotting as dyplot

def single_likelihood(detector, detector_strain, detector_psd, freq_array, waveform, gps_start_time, ra, dec, psi, t_geo, Dl):
    """
    Compute the unnormalised log-likelihood of a gravitational wave signal given a sky location, 
    polarization angle, coalescence time, and luminosity distance for a single detector.

    Parameters
    ----------
    detector : str
        The detector name, must be one of 'H1', 'L1', or 'V1'.
    detector_strain : array-like
        The observed strain data in the frequency domain for the detector.
    detector_psd : array-like
        The one-sided power spectral density (PSD) of the detector.
    delta_f : float
        The frequency resolution of the data (Hz).
    freq_array : array-like
        Array of frequencies corresponding to the Fourier components.
    waveform_h_plus : array-like
        Frequency-domain gravitational wave strain for the plus polarization (at 1 Gpc).
    waveform_h_cross : array-like
        Frequency-domain gravitational wave strain for the cross polarization (at 1 Gpc).
    ra : float
        Right ascension of the source in radians.
    dec : float
        Declination of the source in radians.
    psi : float
        Polarization angle in radians.
    t_geo : float
        Time of coalescence at the geocenter (Not GPS time).
    Dl : float
        Luminosity distance to the source in Gpc.

    Returns
    -------
    log_likelihood : float
        The log-likelihood of the observed data given the signal model at this sky location.
    """

    # Calculate the time delay from the geocenter
    time_delay = detector.time_delay_from_geocenter(ra=ra, dec=dec, time= gps_start_time)

    # Calculate the expected arrival time at the detector - relative to the model time
    shift_time = t_geo + time_delay
    
    # Calculate the antenna response for the given RA, Dec, and Psi
    Fplus = detector.antenna_response(ra, dec, gps_start_time, psi, 'plus')
    Fcross = detector.antenna_response(ra, dec, gps_start_time, psi, 'cross')

    # Calculate the expected strain at the detector - no fourier shift
    # Amplitude is affected by the distance to the source (1Gpc/Dl)
    expected_strain = (Fplus * waveform['plus'] + Fcross * waveform['cross'])/Dl 

    # With a time shift - using fourer transform property
    transformed_expected_strain = expected_strain * np.exp(-2j * np.pi * freq_array * shift_time)

    # The residual strain is the difference between the detector strain and the expected strain
    residual_strain = detector_strain - transformed_expected_strain

    # Determine the frequency resolution
    delta_f = freq_array[1] - freq_array[0]

    # Using a gaussian likelihood model

    log_likelihood = - 2 * delta_f * np.sum(np.real(residual_strain * np.conj(residual_strain) / detector_psd))

    return log_likelihood