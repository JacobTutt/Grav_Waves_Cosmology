import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import bilby
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal

def optimise_matched_filter(detector, detector_strain, detector_psd, waveform, sampling_frequency, ra_range, dec_range, psi_range, n_samples, gps_start_time, plot_times):
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
    
    waveform : dict (complex)
        Dictionary containing the frequency-domain waveforms for the '+' and 'x' polarizations.

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

    # Make sure there is 3 values for each RA, Dec, and Psi - this will be the number of datapoints in the grid.
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
    # RA is sampled uniformly in [0, 2pi]
    ra_vals = np.linspace(ra_range[0], ra_range[1], n_ra)
    # sin(Dec) is sampled uniformly in [-1,1] using which ensures dec is sampled from evenly [-pi/2, pi/2]
    sin_dec_vals = np.linspace(np.sin(dec_range[0]), np.sin(dec_range[1]), n_dec)
    dec_vals = np.arcsin(sin_dec_vals)
    # Psi is sampled uniformly in [0, pi]
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
                waveform_template = Fplus * waveform['plus'] + Fcross * waveform['cross']

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


def matched_filter_optimised(detector, detector_strain, detector_psd, waveform, sampling_frequency, injection_parameters, plot_time, plotting = True):
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

    waveform : dict (complex)
        Dictionary containing the frequency-domain waveforms for the '+' and 'x' polarizations.

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
    waveform_template = Fplus * waveform['plus'] + Fcross * waveform['cross']

    # Calculate the matched filter convolution in the frequency domain
    matched_filter = detector_strain * np.conj(waveform_template) / detector_psd
    # Convert to time domain and calculate the SNR
    rho_sq_t = 4 * np.fft.irfft(matched_filter) * sampling_frequency

    # Convert the SNR to absolute values
    snr = np.sqrt(np.abs(rho_sq_t))
    max_abs_snr_index = np.argmax(snr)

    # Find Full Width at Half Maximum (FWHM)
    # Find the half maximum value
    peak_snr = snr[max_abs_snr_index]
    half_max = peak_snr / 2.0

    # LEFT HAND SIDE
    # Finding the index of the point at which the SNR crosses the half maximum 
    i = max_abs_snr_index
    while i > 0 and snr[i] > half_max:
        i -= 1
    # Linear interpolation between the points either side of the half maximum crossing
    if i > 0:
        f_left = interp1d(snr[i:i+2], plot_time[i:i+2])
        t_left = f_left(half_max)
    else:
        t_left = plot_time[i]

    # RIGHT HAND SIDE
    # Finding the index of the point at which the SNR crosses the half maximum 
    i = max_abs_snr_index
    while i < len(snr) - 1 and snr[i] > half_max:
        i += 1
    #  Linear interpolation between the points either side of the half maximum crossing
    if i < len(snr) - 1:
        f_right = interp1d(snr[i-1:i+1], plot_time[i-1:i+1])
        t_right = f_right(half_max)
    else:
        t_right = plot_time[i]

    # Final FWHM - Half the fwhm width is the uncertainty on the time of the signal
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

        # Right Column: Zoomed-in View 
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


def loglikelihood_delay(ra, dec, detectors = ['H1', 'L1'], arrival_times =[0, 0], arrival_time_uncert = [0.1, 0.1], gps_start_time = 0):
    """
    Compute the log-likelihood of a sky location (RA, Dec) given observed arrival 
    time differences between two detectors using a Gaussian timing model.

    Parameters
    ----------
    ra : float
        Right ascension of the source in radians.
    dec : float
        Declination of the source in radians.
    detectors : list of str
        List of two detector names (e.g., ['H1', 'L1']). Must be two of 'H1', 'L1', or 'V1'.
    arrival_times : list of float
        Measured arrival times (in seconds) at each detector (typically wrt GPS reference).
    arrival_time_uncert : list of float
        1-sigma uncertainties on the arrival times for each detector.
    gps_start_time : float
        Reference GPS time at which to compute time delays from the geocenter.

    Returns
    -------
    log_likelihood : float
        The log-likelihood of the input sky position under the observed time delay 
        using a Gaussian likelihood model.
    
    sigma_away : float
        The number of standard deviations the observed time delay is away from the
        expected time delay based on the sky position and detector timing uncertainties.
        
    Notes
    -----
    The log-likelihood is computed as:
        logL = -0.5 * ((delta_t_obs - delta_t_model)^2 / sigma^2 + log(2π sigma^2))
    where:
        delta_t_obs = observed arrival time difference between detectors
        delta_t_model  = predicted time difference from the sky position
        sigma         = combined timing uncertainty (quadrature sum)
    """

    # Ensure that only two detectors are named and they are either H1, L1 or V1
    if len(detectors) != 2 or not all(detector in ['H1', 'L1', 'V1'] for detector in detectors):
        raise ValueError("Only two detectors are allowed, and they must be either 'H1', 'L1', or 'V1'.")
    
    # Ensure that arrival_times and arrival_time_uncert are lists of the same length as detectors
    if len(arrival_times) != len(detectors) or len(arrival_time_uncert) != len(detectors):
        raise ValueError("arrival_times and arrival_time_uncert must have the same length as detectors.")
    
    # Calculate the difference in arrival times with respective error
    delta_t_obs = arrival_times[0] - arrival_times[1]
    # Add errors in quadrature
    sigma = np.sqrt(arrival_time_uncert[0]**2 + arrival_time_uncert[1]**2) 

    # Calculate the expected time difference based on the sky location
    # Initialise the bilby interferometer objects
    interferometer_0 = bilby.gw.detector.InterferometerList([detectors[0]])[0]
    interferometer_1 = bilby.gw.detector.InterferometerList([detectors[1]])[0]

    # Expected time difference based on the sky location
    # We use the gps start time to calculate the time delay from the geocenter as over 4 seconds the time delay is constant
    time_delay_0 = interferometer_0.time_delay_from_geocenter(ra=ra, dec=dec, time = gps_start_time)
    time_delay_1 = interferometer_1.time_delay_from_geocenter(ra=ra, dec=dec, time = gps_start_time)
    delta_t_model = time_delay_0 - time_delay_1

    # Calculate the number of sigma errors 
    sigma_away = abs((delta_t_obs - delta_t_model) / sigma)

    # A gaussian likelihood is assumed with expected - mean and standard deviation as uncertainty
    log_likelihood = -0.5 * (((delta_t_obs - delta_t_model) / sigma)) ** 2

    return log_likelihood, sigma_away




def generate_likelihood_skymap(detectors, arrival_times, arrival_time_uncert, gps_start_time, n_ra=100, n_dec=50):
    """
    Generate a sky map of the log-likelihood and sigma deviations from observed 
    arrival time differences between gravitational wave detectors.

    This function evaluates the Gaussian timing likelihood for each (RA, Dec) point 
    on a sky grid based on observed time differences and uncertainties from two detectors.
    It also returns the number of standard deviations ("sigma away") each sky location 
    is from the observed time delay.

    Parameters
    ----------
    detectors : list of str
        Names of two detectors used (e.g., ['H1', 'L1']).
    arrival_times : list of float
        Observed arrival times (in seconds) at each detector.
    arrival_time_uncert : list of float
        1-sigma uncertainties on arrival times at each detector.
    gps_start_time : float
        GPS time used to compute time delays from the geocenter.
    n_ra : int, optional
        Number of grid points in right ascension (default is 100).
    n_dec : int, optional
        Number of grid points in declination (default is 50).

    Returns
    -------
    logL_grid : ndarray
        2D array of log-likelihood values evaluated over the sky grid.
    sigma_away : ndarray
        2D array representing how many sigma each sky location is away 
        from the observed time delay difference.
    """

    # Create a grid of RA and Dec values that will be interpolated between in the countour plot
    # RA ranges from 0 to 2i
    ra = np.linspace(0, 2 * np.pi, n_ra)
    # Dec ranges from -pi/2 to pi/2
    dec = np.linspace(-np.pi / 2, np.pi / 2, n_dec)
    # Create a meshgrid for RA and Dec
    RA, DEC = np.meshgrid(ra, dec)

    # Initialize arrays to hold log-likelihood and sigma values
    logL_grid = np.zeros_like(RA)
    sigma_away = np.zeros_like(RA)

    # Compute log-likelihood over grid
    for i in range(RA.shape[0]):
        for j in range(RA.shape[1]):
            # Calculate log-likelihood and sigma away for each (RA, Dec) point
            logL_grid[i, j], sigma_away[i, j]= loglikelihood_delay( RA[i, j], DEC[i, j], detectors=detectors, arrival_times=arrival_times,
                arrival_time_uncert=arrival_time_uncert, gps_start_time=gps_start_time )

    return logL_grid, sigma_away
    


def plot_confidence_map_sigma(sigma_away, n_ra=100, n_dec=50):
    """
    Plot confidence regions (1, 2, 3) sigma from a sigma-away map using Cartopy.

    This are the uncertainties in the arrival times of the gravitational wave signal
    at the detectors, which can be used to assess the confidence in the sky localization.

    Parameters
    ----------
    sigma_away : 2D array
        Array of sigma differences between observed and modelled arrival times.
    n_ra : int
        Number of RA grid points.
    n_dec : int
        Number of Dec grid points.
    """

    # Coordinate grid
    ra = np.linspace(0, 2 * np.pi, n_ra)
    dec = np.linspace(-np.pi / 2, np.pi / 2, n_dec)
    RA, DEC = np.meshgrid(ra, dec)

    # Shift RA from [0, 2pi] to [0, pi, -pi] for continuity
    RA_shifted = ((RA + np.pi) % (2 * np.pi)) - np.pi

    # Ensure Continuity in values ie roll them for RA, DEC and sigma
    RA_shifted = np.roll(RA_shifted, shift=RA.shape[1] // 2, axis=1)
    DEC = np.roll(DEC, shift=RA.shape[1] // 2, axis=1)
    sigma_away = np.roll(sigma_away, shift=RA.shape[1] // 2, axis=1)

    # Confidence sigma levels
    levels = [0, 1, 2, 3]
    # Dark blue, light blue, lightest blue, white
    colors = ['#08306b', '#2171b5', '#deebf7', 'white'] 

    # Create plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='mollweide')

    # Filled contour
    cf = ax.contourf(RA_shifted, DEC, sigma_away, levels=levels, colors=colors, extend='max', zorder=0)

    # Contour lines
    cs = ax.contour(RA_shifted, DEC, sigma_away, levels=levels[1:], colors='red', linewidths=1, zorder=1)
    fmt = {1: "1σ", 2: "2σ", 3: "3σ"}
    ax.clabel(cs, fmt=fmt, fontsize=13, colors='red')

    # Gridlines (above contours)
    ax.grid(True, linestyle='--', color='gray', linewidth=1, alpha=0.8, zorder=2)

    # Custom RA tick labels (in hours, mirrored for astronomical convention)
    ra_labels_deg = np.arange(-150, 179, 30)  # e.g., [-150, -120, ..., 180]
    ra_labels_text = [f"{x}°" for x in ra_labels_deg]
    ax.set_xticks(np.radians(ra_labels_deg))
    ax.set_xticklabels(ra_labels_text, fontsize=12, color = 'black', zorder=3)

    # Dec tick labels
    dec_labels_deg = [-60, -30, 0, 30, 60]
    ax.set_yticks(np.radians(dec_labels_deg))
    ax.set_yticklabels([f"{d}°" for d in dec_labels_deg], fontsize=12)

    plt.tight_layout()
    plt.show()


   
    


def plot_confidence_map_wilkes(log_likelihood, n_ra=100, n_dec=50):
    """
    Plot confidence regions (1, 2, 3) sigma using Wilks' theorem and log-likelihood values.

    Wilkes theorem states that for a likelihood ratio test, the distribution of the
    test statistic under the null hypothesis is asymptotically chi-squared distributed.


    Parameters
    ----------
    log_likelihood : 2D array
        Log-likelihood evaluated on a (RA, Dec) grid.
    n_ra : int
        Number of RA grid points.
    n_dec : int
        Number of Dec grid points.
    """
    
    # Coordinate grid
    ra = np.linspace(0, 2 * np.pi, n_ra)
    dec = np.linspace(-np.pi / 2, np.pi / 2, n_dec)
    RA, DEC = np.meshgrid(ra, dec)

    # Convert log-likelihood to delta chi²
    delta_chi2 = -2 * (log_likelihood - np.max(log_likelihood))
    RA_shifted = ((RA + np.pi) % (2 * np.pi)) - np.pi # Shift RA to [0, 2π] to [0, π, -π]
    # Ensure Continuity in values ie roll them for RA, DEC and log-likelihood
    RA_shifted = np.roll(RA_shifted, shift=RA.shape[1] // 2, axis=1)
    DEC = np.roll(DEC, shift=RA.shape[1] // 2, axis=1)
    delta_chi2 = np.roll(delta_chi2, shift=RA.shape[1] // 2, axis=1)

    # Confidence levels (Wilks’ theorem, 2 DOF)
    levels = [0.0, 2.30, 6.17, 11.8]
    colors = ['#08306b', '#2171b5', '#deebf7', 'white' ]

    # Create plot
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    # Plot filled contours
    cf = ax.contourf(RA_shifted, DEC, delta_chi2, levels=levels, colors=colors, extend='max', zorder=0)

    # Contour lines
    cs = ax.contour(RA_shifted, DEC, delta_chi2, levels=levels[1:], colors='red', linewidths=1, zorder=1)
    fmt = {2.30: "1σ", 6.17: "2σ", 11.8: "3σ"}
    ax.clabel(cs, fmt=fmt, fontsize=13, colors = 'red')

    # Gridlines (must be drawn AFTER contours to appear on top)
    ax.grid(True, linestyle='--', color='grey', linewidth=1, alpha=1, zorder=2)

    # Custom RA tick labels
    ra_labels_deg = np.arange(-150, 179, 30)
    ra_labels_text = [f"{x}°" for x in ra_labels_deg]
    ax.set_xticks(np.radians(ra_labels_deg))
    ax.set_xticklabels(ra_labels_text, fontsize=12, color = 'black')

    # Custom Dec tick labels
    dec_labels_deg = [-60, -30, 0, 30, 60]
    ax.set_yticks(np.radians(dec_labels_deg))
    ax.set_yticklabels([f"{d}°" for d in dec_labels_deg], fontsize=12)

    # X and Y labels
    ax.set_xlabel("Right Ascension (RA)", fontsize=14)
    ax.set_ylabel("Declination (Dec)", fontsize=14)
    

    # Title and layout
    plt.tight_layout()

