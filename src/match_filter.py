import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import bilby
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal

def optimise_matched_filter(detector, detector_strain, detector_psd, h_plus, h_cross, sampling_frequency, ra_range, dec_range, psi_range, n_samples, gps_start_time, plot_times):
    """
    Perform a grid search over (RA, Dec, Psi) sky locations and polarization angles to 
    find the maximum matched filter SNR using a given waveform template.

    We ensure to sample from Declination values uniformly on the sphere, which is done by 
    sampling the sine of the declination angle. We also ignore the effects of time and do not
    optimise for this as the effects are negligible over the 4 second duration of the signal.

    This function computes the antenna pattern response for a given detector and
    set of sky positions, constructs the corresponding waveform projection using
    the plus and cross polarizations, and applies matched filtering in the 
    frequency domain. The maximum time-domain SNR is extracted from the inverse 
    Fourier transform of the matched filter result.

    Parameters
    ----------
    detector : str
        Name of the detector (e.g., 'H1', 'L1', 'V1').

    detector_strain : array_like (complex)
        Frequency-domain strain data from the detector.

    detector_psd : array_like (float)
        One-sided power spectral density (PSD) of the detector noise, same shape as strain.

    h_plus : array_like (complex)
        Frequency-domain waveform for the '+' polarization.

    h_cross : array_like (complex)
        Frequency-domain waveform for the 'x' polarization.

    sampling_frequency : float
        Sampling rate in Hz used to convert back to time domain via IFFT.

    ra_range : tuple of float
        Range of right ascension values (min, max) in radians, within [0, 2π].

    dec_range : tuple of float
        Range of declination values (min, max) in radians, within [-π/2, π/2].

    psi_range : tuple of float
        Range of polarization angles (min, max) in radians, within [0, π].

    n_samples : tuple of int
        Number of sample points to evaluate in (RA, Dec, Psi), e.g., (10, 10, 8).

    gps_start_time : float
        Geocentric GPS time for the waveform injection, used for antenna response.

    Returns
    -------
    max_snr : float
        The maximum matched filter SNR found across the grid.

    best_params : dict
        Dictionary containing the (RA, Dec, Psi) values corresponding to the max SNR.
    """

    # Ensure the detector either 'H1', 'L1', or 'V1'
    if detector not in ['H1', 'L1', 'V1']:
        raise ValueError("Detector must be one of 'H1', 'L1', or 'V1'.")

    # Make sure there is 3 values for the range of RA, Dec, and Psi
    if not isinstance(n_samples, tuple) or len(n_samples) != 3:
        raise ValueError("n_samples must be a tuple of three integers (n_ra, n_dec, n_psi).")

    # Ensure that min and max values are within the expected ranges for each of the orientation values
    if not (0.0 <= ra_range[0] <= 2 * np.pi and 0.0 <= ra_range[1] <= 2 * np.pi):
        raise ValueError("RA range must be within [0, 2π].")

    if not (-np.pi/2 <= dec_range[0] <= np.pi/2 and -np.pi/2 <= dec_range[1] <= np.pi/2):
        raise ValueError("Dec range must be within [-π/2, π/2].")

    if not (0.0 <= psi_range[0] <= np.pi and 0.0 <= psi_range[1] <= np.pi):
        raise ValueError("Psi range must be within [0, π].")

    # Create the interferometer object using bilby
    Inter_obj = bilby.gw.detector.InterferometerList([detector])[0]

    # Unpack the number of samples for RA, Dec, and Psi
    n_ra, n_dec, n_psi = n_samples

    # Sample for RA, Dec, and Psi uniformly across the specified ranges
    # RA is sampled uniformly in [0, 2π]
    ra_vals = np.linspace(ra_range[0], ra_range[1], n_ra)
    # Dec is sampled uniformly in [-π/2, π/2] using arcsin to ensure uniform distribution in sphere
    sin_dec_vals = np.linspace(np.sin(dec_range[0]), np.sin(dec_range[1]), n_dec)
    dec_vals = np.arcsin(sin_dec_vals)
    # Psi is sampled uniformly in [0, π]
    psi_vals = np.linspace(psi_range[0], psi_range[1], n_psi)

    # Keep a record of the maximum SNR and the best parameters
    max_snr = 0
    best_params = None
    best_time = 0
    all_times = []

    # Iterate over all combinations of RA, Dec, and Psi
    for ra in ra_vals:
        for dec in dec_vals:
            for psi in psi_vals:

                # Create a dictionary of injection parameters given this combination
                injection_parameters = {
                    'ra': ra,
                    'dec': dec,
                    'geocent_time': gps_start_time,
                    'psi': psi,
                }

                # Calculate the antenna response for the given RA, Dec, and Psi
                Fplus = Inter_obj.antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['geocent_time'], injection_parameters['psi'], 'plus')
                Fcross = Inter_obj.antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['geocent_time'], injection_parameters['psi'], 'cross')

                # Create the waveform template in frequency domain using the antenna responses and the waveform polarizations
                waveform_template = Fplus * h_plus + Fcross * h_cross

                # Calculate the matched filter convolution in the frequency domain
                matched_filter = detector_strain * np.conj(waveform_template) / detector_psd
                # Convert to time domain and calculate the SNR
                rho_sq_t = 4 * np.fft.irfft(matched_filter) * sampling_frequency
                # Determine the maximium SNR achieved using the matched filter
                snr = np.max(np.abs(rho_sq_t))

                snr_index = np.argmax(np.abs(rho_sq_t))
                # Time corresponding to the maximum SNR
                time_at_max_snr = plot_times[snr_index]

                # Store the time at which the maximum SNR occurs
                all_times.append(time_at_max_snr)

                # If this SNR is greater than the current maximum, update the maximum and best parameters
                if snr > max_snr:
                    max_snr = snr
                    best_params = injection_parameters

    


    # Print the results
    print(f"Maximum Rho^2 found for Dector {detector}: {max_snr} occured at time {best_time:.3f} seconds from GPS start time {gps_start_time}.")
    print(f"Best parameters for Dector {detector}: RA = {best_params['ra']:.3f}, Dec = {best_params['dec']:.3f}, Psi = {best_params['psi']:.3f}")
    # Print the range of times at which the maximum SNR occurs
    print(f"The standard deviation of the times at which the maximum SNR occurs is {np.std(all_times):.3f} seconds.")

    return max_snr, best_params


def matched_filter_optimised(detector, detector_strain, detector_psd, h_plus, h_cross, sampling_frequency, injection_parameters, plot_time, plotting = True):
    """
    Perform matched filtering for a single gravitational-wave detector and waveform projection,
    returning the peak SNR, time of peak, and associated timing uncertainty (FWHM).

    The function computes the matched filter output by correlating the detector strain data
    against a projected waveform template defined by given sky location and polarization 
    parameters. It also estimates the uncertainty on the signal arrival time using the 
    full-width at half-maximum (FWHM) of the SNR time series, with interpolation for precision.
    
    If `plotting=True`, a two-panel figure is shown with the full and zoomed-in SNR time series
    and visual markers for the SNR peak and FWHM region.

    Parameters
    ----------
    detector : str
        Name of the detector ('H1', 'L1', or 'V1').

    detector_strain : array_like (complex)
        Frequency-domain strain data from the detector.

    detector_psd : array_like (float)
        One-sided power spectral density of the detector, same shape as strain.

    h_plus : array_like (complex)
        Frequency-domain waveform corresponding to the '+' polarization.

    h_cross : array_like (complex)
        Frequency-domain waveform corresponding to the 'x' polarization.

    sampling_frequency : float
        Sampling frequency in Hz used to convert frequency-domain to time-domain.

    injection_parameters : dict
        Dictionary specifying 'ra', 'dec', 'psi', and 'geocent_time' values for the waveform projection.

    plot_time : array_like (float)
        Time array corresponding to the inverse FFT of the matched filter output.

    plotting : bool, optional
        If True, plots the full and zoomed-in SNR time series with FWHM indication.

    Returns
    -------
    max_snr : float
        The peak signal-to-noise ratio (SNR) found in the matched filter time series.

    time_max_snr : float
        The time at which the maximum SNR occurs (in the same units as plot_time).

    time_uncertainty : float
        Estimated uncertainty on the signal time, computed from full-width at half-maximum (FWHM).
    """

    # Ensure the detector either 'H1', 'L1', or 'V1'
    if detector not in ['H1', 'L1', 'V1']:
        raise ValueError("Detector must be one of 'H1', 'L1', or 'V1'.")
    
    # Create the interferometer object using bilby
    Inter_obj = bilby.gw.detector.InterferometerList([detector])[0]


    # Calculate the antenna response for the given RA, Dec, and Psi
    Fplus = Inter_obj.antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['geocent_time'], injection_parameters['psi'], 'plus')
    Fcross = Inter_obj.antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['geocent_time'], injection_parameters['psi'], 'cross')

    # Create the waveform template in frequency domain using the antenna responses and the waveform polarizations
    waveform_template = Fplus * h_plus + Fcross * h_cross

    # Calculate the matched filter convolution in the frequency domain
    matched_filter = detector_strain * np.conj(waveform_template) / detector_psd
    # Convert to time domain and calculate the SNR
    rho_sq_t = 4 * np.fft.irfft(matched_filter) * sampling_frequency

    # Convert the SNR to absolute values
    snr = np.sqrt(np.abs(rho_sq_t))
    max_abs_snr_index = np.argmax(snr)

    # --- Find Full Width at Half Maximum (FWHM) ---
    peak_snr = snr[max_abs_snr_index]
    half_max = peak_snr / 2.0

    # Finding the index of the point at which the SNR crosses the half maximum on the LHS
    i = max_abs_snr_index
    while i > 0 and snr[i] > half_max:
        i -= 1
    # Linear interpolation between the points either side of the half maximum crossing
    if i > 0:
        f_left = interp1d(snr[i:i+2], plot_time[i:i+2])
        t_left = f_left(half_max)
    else:
        t_left = plot_time[i]

    # Finding the index of the point at which the SNR crosses the half maximum on the RHS
    i = max_abs_snr_index
    while i < len(snr) - 1 and snr[i] > half_max:
        i += 1
    #  Linear interpolation between the points either side of the half maximum crossing
    if i < len(snr) - 1:
        f_right = interp1d(snr[i-1:i+1], plot_time[i-1:i+1])
        t_right = f_right(half_max)
    else:
        t_right = plot_time[i]

    # --- Final FWHM ---
    time_uncertainty = 1/2 * (t_right - t_left)



    if plotting:
        # Identify the maximum SNR and corresponding time for zooming
        zoom_center_time = plot_time[max_abs_snr_index]


        # Calculate the overall time range of the plot and set the zoom duration as a percentage of this range
        overall_plot_range = plot_time.max() - plot_time.min()
        zoom_duration = 0.05 * overall_plot_range
        zoom_start = zoom_center_time - zoom_duration / 2
        zoom_end = zoom_center_time + zoom_duration / 2
        zoom_start = max(zoom_start, plot_time.min())
        zoom_end = min(zoom_end, plot_time.max())

        # Begin Plotting
        fig = plt.figure(figsize=(18, 4)) 
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

        colour_dict = {
            'H1': 'red',
            'L1': 'blue',
            'V1': 'green'
        }

        # Left Column: Full Signal
        ax_full = fig.add_subplot(gs[0, 0])
        ax_full.plot(plot_time, snr, label=f"{detector}", color=colour_dict[detector])
        ax_full.set_ylabel(r"SNR ($\rho(t)$)")
        ax_full.legend()
        ax_full.set_xlabel(r"Coalescence Time, $t_c$ [s]")
        ax_full.grid(True)

        # Add a rectangle to the full view plots to show the zoomed area
        ax_full.axvspan(zoom_start, zoom_end, color='gray', alpha=0.2, hatch='/', label='Zoomed Area')
        ax_full.legend(loc='upper left')

        # Right Column: Zoomed-in View (
        ax_zoom = fig.add_subplot(gs[0, 1], sharey=ax_full)
        ax_zoom.plot(plot_time, snr, label=f"{detector}", color=colour_dict[detector])
        ax_zoom.set_xlim(zoom_start, zoom_end)
        ax_zoom.tick_params(labelleft=False)
        ax_zoom.grid(True)
        ax_zoom.set_xlabel(r"Coalescence Time, $t_c$ [s]")
        # Vertical and Horizontal lines for the maximum SNR and FWHM
        ax_zoom.axhline(peak_snr, color='black', linestyle='--', linewidth=2)
        ax_zoom.axhline(peak_snr/2, color='black', linestyle='--', linewidth=2)



        ax_zoom.axvspan(zoom_start, zoom_end, color='gray', alpha=0.2, hatch='/', label='Zoomed Area')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Find the maximum SNR and corresponding time
        max_snr = snr[max_abs_snr_index]
        time_max_snr = plot_time[max_abs_snr_index]

        print(f"Maximum SNR for {detector}: {max_snr:.2f} at time {time_max_snr:.6f} pm {time_uncertainty:.6f} s")

    return max_snr, time_max_snr, time_uncertainty