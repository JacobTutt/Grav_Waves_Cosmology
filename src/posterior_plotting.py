import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker



from scipy.stats import norm
import healpy as hp




import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Set Plotting Style
mpl.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "text.usetex": False,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
})


def table_dynesty_posteriors(dynesty_results, shift_psi=False):
    """
    Generate a summary table of posterior statistics for Dynesty samples.

    Parameters
    ----------
    dynesty_results : dynesty.results.Results
        The results object from Dynesty containing samples and weights.

    param_labels : list of str, optional
        Optional list of parameter labels to include in the table.

    shift_psi : bool, optional
        If True, wraps psi from [0, pi] to [pi/2, 3pi/2] to obtain a more
        accurate estimate of polarization angle. Accounts for symmetry in 
        the waveform.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows for each parameter and columns for:
        - Mean
        - Median
        - 68%, 90%, 95%, 99% CI
    """

    # Extract samples and weights from the Dynesty results
    ndim = dynesty_results.samples.shape[1]
    samples = dynesty_results.samples.copy()
    weights = dynesty_results.importance_weights()
    weights /= np.sum(weights)

    # Define parameter labels for this specific analysis
    if ndim == 5:
        param_labels = [r"$\mathrm{RA}$", r"$\mathrm{Dec}$", r"$\psi$", r"$t_c^{\mathrm{(geo)}}$", r"$D_L$"]
    elif ndim == 3:
        param_labels = [r"$\psi$", r"$t_c^{\mathrm{(geo)}}$", r"$D_L$"]

    # Enable a shift in psi if it has hit the edge of the range which can effect the statistics
    if shift_psi:
        psi = samples[:, 2]
        samples[:, 2] = np.where(psi < np.pi / 2, psi + np.pi, psi)

    # Resample with equal weights for robust statistics
    equal_samples = resample_equal(samples, weights)

    stats_list = []

    # Define the credible intervals and their corresponding percentiles
    ci_levels = {
        "68%": [16, 84],
        "90%": [5, 95],
        "95%": [2.5, 97.5],
        "99%": [0.5, 99.5]
    }

    for i, label in enumerate(param_labels):
        param = equal_samples[:, i]
        row_stats = {"parameter": label}

        # Calculate mean and median
        mean_val = np.mean(param)
        median_val = np.median(param)

        # Format mean and median to 4 decimal places
        row_stats["mean"] = f"{mean_val:.4f}" 
        row_stats["median"] = f"{median_val:.4f}"

        # Calculate and format credible intervals
        for ci_label, percentiles in ci_levels.items():
            p_lower, p_upper = np.percentile(param, percentiles)

            # Convert the percentile thresholds to deviations from the median by calculating the differences
            upper_deviation = p_upper - median_val
            lower_deviation = median_val - p_lower

            # Format it for printing ie in text format
            row_stats[ci_label] = f"+{upper_deviation:.4f} − {lower_deviation:.4f}"

        stats_list.append(row_stats)

    # Create a DataFrame from the statistics list
    summary_df = pd.DataFrame(stats_list)
    summary_df.set_index("parameter", inplace=True)

    # This is best displayed as a matplotlib table
    fig, ax = plt.subplots(figsize=(10, summary_df.shape[0] * 0.5)) 
    ax.axis('off')
    table = ax.table(cellText=summary_df.values, rowLabels=summary_df.index, colLabels=summary_df.columns,
                    loc='center', cellLoc='center', rowLoc='right') 

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.show()

    return None


def plot_posterior_skymap_samples(dynesty_results, ra_range = [101, 160], dec_range = [-79, -41], title='Sky Map: H1 and L1', colors=['red', 'blue'], background = None):
    """
    Plot a gravitational-wave sky localization using posterior samples from Dynesty.

    The function creates a Mollweide projection showing posterior samples for sky position (RA, Dec),
    with an optional background contour from a Wilks' theorem-derived log-likelihood map.
    A zoom-in inset displays a selected RA Dec region with additional KDE-based confidence contours and 1 sigma bounds.

    Parameters
    ----------
    dynesty_results : dynesty.results.Results
        A Dynesty results object containing posterior samples and weights.
        Assumes RA is the first column and Dec is the second, both in radians.

    ra_range : list of float, optional
        Range of Right Ascension (RA) in degrees for the inset zoom region. Default is [101, 160].

    dec_range : list of float, optional
        Range of Declination (Dec) in degrees for the inset zoom region. Default is [-79, -41].

    title : str, optional
        Title for the main Mollweide sky map plot. Default is 'Sky Map: H1 and L1'.

    colors : list of str, optional
        Two colors: the first is used for plotting posterior samples and contours,
        the second is used for median lines, markers, and 1 sigma uncertainty shading. Default is ['red', 'blue'].

    background : str or None, optional
        File path to a `.npy` file containing a 2D log-likelihood array (e.g. from Wilks' theorem),
        used to plot background confidence contours. Default is None (no background).

    Notes
    -----
    - RA samples are shifted to [-π, π] for display in the Mollweide projection.
    - The zoom-in subplot highlights a rectangular RA Dec region with:
        - A scatter plot of posterior samples,
        - KDE-based confidence contours,
        - Median crosshairs and marker,
        - 1 sigma uncertainty regions filled with hatch lines.
    - If `background` is provided, the plot overlays Wilks contours (1sigma, 2sigma, 3sigma) using light grey shading.

    """

    # Extract samples and weights from Dynesty results    
    samples = dynesty_results.samples
    weights = dynesty_results.importance_weights()
    weights /= np.sum(weights)
    # Resample for equal-weight posterior 
    equal_samples = resample_equal(samples, weights)

    # Extract RA and Dec from the samples
    ra = equal_samples[:, 0]
    dec = equal_samples[:, 1]
    # Wrap RA and Dec to the range [-pi, pi] for Mollweide projection
    ra_shifted = ((ra + np.pi) % (2 * np.pi)) - np.pi

    # Create figure and main Mollweide plot
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    if background is not None:
        # Load the numpy file from the background path
        log_likelihood = np.load(background)
        # Coordinate grid
        ra_back = np.linspace(0, 2 * np.pi, 100)
        dec_back = np.linspace(-np.pi / 2, np.pi / 2, 50)
        RA_back, DEC_back = np.meshgrid(ra_back, dec_back)
        # Convert log-likelihood to delta chi
        delta_chi2 = -2 * (log_likelihood - np.max(log_likelihood))
        RA_shifted_back = ((RA_back + np.pi) % (2 * np.pi)) - np.pi 
        # Ensure Continuity in values ie roll them for RA, DEC and log-likelihood
        RA_shifted_back = np.roll(RA_shifted_back, shift=RA_back.shape[1] // 2, axis=1)
        DEC_back = np.roll(DEC_back, shift=RA_back.shape[1] // 2, axis=1)
        delta_chi2 = np.roll(delta_chi2, shift=RA_back.shape[1] // 2, axis=1)
        # Confidence levels (Wilks’ theorem, 2 DOF)
        levels = [0.0, 2.30, 6.17, 11.8]
        colors_back = ['#cccccc', '#e0e0e0', '#f2f2f2', 'white'] 
        # Plot filled contours
        cf = ax.contourf(RA_shifted_back, DEC_back, delta_chi2, levels=levels, colors=colors_back, extend='max', zorder=0)

        # Contour lines
        cs = ax.contour(RA_shifted_back, DEC_back, delta_chi2, levels=levels[1:], colors='black', linewidths=1, zorder=1)
        fmt = {2.30: "1σ", 6.17: "2σ", 11.8: "3σ"}
        ax.clabel(cs, fmt=fmt, fontsize=13, colors = 'black')

    # Format RA and Dec ticks
    ax.grid(True, linestyle='--', color='grey', linewidth=1, alpha=1)
    ax.set_xticks(np.radians(np.arange(-150, 180, 30)))
    ax.set_xticklabels([f"{x}°" for x in np.arange(-150, 180, 30)], fontsize=12)
    ax.set_yticks(np.radians([-60, -35, 0, 30, 60]))
    ax.set_yticklabels([f"{y}°" for y in [-60, -30,'' ,'' ,'' ]], fontsize=12)
    ax.set_title(title, fontsize=14)

    # Plot posterior samples on sky map
    ax.scatter(ra_shifted, dec, s=2, alpha=0.03, color=colors[0])

    # Zommed in inset
    axins = inset_axes(ax, width="60%", height="60%", loc='upper left', borderpad=0)

    # Convert RA to degrees for zoom
    ra_deg = np.degrees(ra)
    dec_deg = np.degrees(dec)

    # Set the limits for the inset zoom
    ra_min, ra_max = ra_range[0], ra_range[1]
    dec_min, dec_max = dec_range[0], dec_range[1]
    # Set the inset axes properties
    axins.grid(True, linestyle='--', color='grey', linewidth=1, alpha=0.5)
    axins.scatter(ra_deg, dec_deg, s=2, alpha=0.03, color=colors[0])
    axins.set_xlim(ra_min, ra_max) 
    axins.set_ylim(dec_min, dec_max)
    axins.set_xlabel("Right Ascension", fontsize=12, fontweight='bold', labelpad=1)   
    axins.set_ylabel("Declination", fontsize=12, fontweight='bold', labelpad=1)
    axins.tick_params(labelsize=12, pad=2)

    # KDE-based contours over the zoom region
    xy = np.vstack([ra_deg, dec_deg])
    kde = gaussian_kde(xy)
    x = np.linspace(ra_min, ra_max, 100)
    y = np.linspace(dec_min, dec_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    axins.contour(X, Y, Z, levels=3, colors='black', linewidths=0.8)

    # Add confidence regions
    declination_median = np.median(dec_deg)
    lower_dec_bound = np.percentile(dec_deg, 16)
    upper_dec_bound = np.percentile(dec_deg, 84)
    ra_median = np.median(ra_deg)
    lower_ra_bound = np.percentile(ra_deg, 16)
    upper_ra_bound = np.percentile(ra_deg, 84)

    # Add cross section lines at median values
    axins.axhline(declination_median, color=colors[1], linestyle='-', linewidth=1.5)
    axins.axvline(ra_median, color=colors[1], linestyle='-', linewidth=1.5)
    axins.scatter(ra_median, declination_median, color=colors[1], s=40, marker='s', zorder=5)
    # Add shaded region for 1-sigma uncertainty
    axins.fill_betweenx([lower_dec_bound, upper_dec_bound], ra_min, ra_max, color=colors[1], alpha=0.1, hatch='///')
    axins.fill_betweenx([dec_min, dec_max], lower_ra_bound, upper_ra_bound, color=colors[1], alpha=0.1, hatch='///')

    plt.tight_layout()
    plt.show()

def plot_corner_dynesty(dynesty_results, title = 'Posterior Samples: H1 and L1', colors=['red', 'blue'], shift_psi = False):
    """
    Plot a corner plot of the Dynesty results.

    Parameters
    ----------
    dynesty_results : dynesty.results.Results
        The results object from Dynesty containing samples and weights.
    """
    # Get samples and weights from Dynesty results
    ndim = dynesty_results.samples.shape[1]
    samples = dynesty_results.samples
    weights = dynesty_results.importance_weights()
    weights /= np.sum(weights)

    if shift_psi:
        # Shift psi to be in the range [pi/2, 3pi/2] from [0, pi]
        if ndim == 5:
            index = 2
        if ndim == 3:
            index = 1
        
        psi = samples[:, index]
        psi_wrapped = np.where(psi < np.pi / 2, psi + np.pi, psi)
        samples[:, index] = psi_wrapped

        
    # Resample for equal-weight posterior
    equal_weight_samples = resample_equal(samples, weights)

    # Compute means for each parameter
    medians = np.median(equal_weight_samples, axis=0)
    lower_bounds = np.percentile(equal_weight_samples, 16, axis=0)
    upper_bounds = np.percentile(equal_weight_samples, 84, axis=0)
    

    # Define nicely formatted labels
    if ndim == 5:
        labels = [r"$\mathrm{RA}$", r"$\mathrm{Dec}$", r"$\psi$", r"$t_c^{\mathrm{(geo)}}$", r"$D_L$"]
    if ndim == 3:
        labels = [r"$\psi$", r"$t_c^{\mathrm{(geo)}}$", r"$D_L$"]

    # Set up the Axes for the corner plot
    fig, axes = plt.subplots(ndim, ndim, figsize=(ndim * 2, ndim * 2), squeeze=False)

    # Use Corner package to create the corner plot
    corner.corner(equal_weight_samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt=".2f", color=colors[0], smooth=0.2, plot_datapoints=True,
        use_math_text=True, fig=fig, max_n_ticks=3)

    # Add vertical/horizontal lines at the mean in blue
    for i in range(ndim):
        for j in range(i + 1):
            ax = axes[i, j]
            if i == j:
                # 1D histogram (diagonal)
                ax.axvline(medians[i], color=colors[1], linestyle='-', lw=2)
                ax.axvline(lower_bounds[i], color=colors[1], linestyle='--', lw=2)
                ax.axvline(upper_bounds[i], color=colors[1], linestyle='--', lw=2)
                # Add shaded region for 1-sigma uncertainty
                ax.fill_betweenx(ax.get_ylim(), lower_bounds[i], upper_bounds[i], color=colors[1], hatch='///', alpha=0.15)

            else:
                # 2D contour plot
                ax.axvline(medians[j], color=colors[1], linestyle='-', lw=1.5)
                ax.axhline(medians[i], color=colors[1], linestyle='-', lw=1.5)
                ax.scatter(medians[j], medians[i], color=colors[1], s=40, marker = 's', zorder=5)

    # Add title box in top-left corner (outside main grid)
    fig.text(0.93, 0.96, title, fontsize=20, fontweight='bold',
             horizontalalignment='right', verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='black'))


    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()


def compare_ra_dec(dynesty_results_1, dynesty_results_2, colors=['red', 'blue'], labels =['H1 and L1', 'H1, L1 and V1'], dec_range=[-74, -43], ra_range=[41, 158]):
    """
    Generate a professional RA vs Dec posterior comparison plot using Matplotlib.

    This function visualizes and compares sky localization posterior distributions from two
    gravitational wave parameter estimation runs (e.g., using Dynesty). It combines 2D KDE contours 
    with overlaid samples and 1D marginal distributions in a compact, publication-ready layout.

    Parameters
    ----------
    dynesty_results_1 : dynesty.results.Results
        The first Dynesty results object containing posterior samples and weights.

    dynesty_results_2 : dynesty.results.Results
        The second Dynesty results object containing posterior samples and weights.

    colors : list of str, optional
        Colors used for each dataset's contours, scatter, and hatch fill. Default is ['blue', 'red'].

    labels : list of str, optional
        Labels for each dataset (used in legend and annotations). Default is 
        ['H1 and L1', 'H1, L1 and V1'].

    dec_range : list of float, optional
        Y-axis limits (Declination in degrees) for the central joint plot. Default is [-74, -43].

    ra_range : list of float, optional
        X-axis limits (Right Ascension in degrees) for the central joint plot. Default is [41, 158].

    Notes
    -----
    - This plot includes:
        * KDE contours in RA-Dec
        * Sample scatter points
        * 1D marginal histograms and KDEs
        * Crosshairs at the median
        * Hatched bands showing the 68% credible region (±1 sigma)
    - Marginal plots are stripped of ticks and labels for visual clarity.
    - Layout is tightly packed using GridSpec with shared axes.
    """

    # Extract RA and Dec samples from the Dynesty results
    def get_deg_samples(results):
        w = results.importance_weights()
        w /= np.sum(w)
        s_eq = resample_equal(results.samples, w)
        return np.degrees(s_eq[:, 0]), np.degrees(s_eq[:, 1])
    
    ra1, dec1 = get_deg_samples(dynesty_results_1)
    ra2, dec2 = get_deg_samples(dynesty_results_2)

    # Setup the figure and gridspec layout
    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.00, hspace=0.00)

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    # For each dataset, plot the RA-Dec joint distribution and marginal distributions
    for ra, dec, label, color in zip([ra1, ra2], [dec1, dec2],labels, colors):

        # Add crosshairs and hatched regions for joint plot
        ra_median = np.median(ra)
        dec_median = np.median(dec)
        lower_ra_bound = np.percentile(ra, 16)
        upper_ra_bound = np.percentile(ra, 84)
        lower_dec_bound = np.percentile(dec, 16)
        upper_dec_bound = np.percentile(dec, 84)

        ax_joint.axhline(dec_median, color=color, linestyle='--', linewidth=1)
        ax_joint.axvline(ra_median, color=color, linestyle='--', linewidth=1)

        ax_joint.fill_betweenx([lower_dec_bound, upper_dec_bound], ra_range[0], ra_range[1], color=color, alpha=0.06, hatch='///', edgecolor=color)
        ax_joint.fill_between([lower_ra_bound, upper_ra_bound], dec_range[0], dec_range[1], color=color, alpha=0.06, hatch='///', edgecolor=color)

        ax_marg_x.axvspan(lower_ra_bound, upper_ra_bound, color=color, alpha=0.06, hatch='///', edgecolor=color)
        ax_marg_y.axhspan(lower_dec_bound, upper_dec_bound, color=color, alpha=0.06, hatch='///', edgecolor=color)

        # Plit the Scatter points for the joint plot
        ax_joint.scatter(ra, dec, s=2, alpha=0.1, color=color, label=label)

        # Marginal Side Plots
        kde_ra = gaussian_kde(ra)
        x_ra = np.linspace(ra.min(), ra.max(), 100)
        ax_marg_x.plot(x_ra, kde_ra(x_ra), color=color, lw=1)
    
        kde_dec = gaussian_kde(dec)
        y_dec = np.linspace(dec.min(), dec.max(), 100)
        ax_marg_y.plot(kde_dec(y_dec), y_dec, color=color, lw=1)

        ax_marg_x.hist(ra, bins=40, color=color, alpha=0.3, edgecolor='none', density=True)
        ax_marg_y.hist(dec, bins=40, orientation='horizontal', color=color, alpha=0.3, edgecolor='none', density=True)

    # Set axis labels for plot
    ax_joint.set_xlabel("Right Ascension")
    ax_joint.set_ylabel("Declination")
    ax_joint.grid(True, linestyle='--', alpha=0.6)
    ax_joint.tick_params(direction='in')

    # Add degree symbols to marginal plots
    ax_marg_x.set_xticks(ax_marg_x.get_xticks())
    ax_marg_x.set_xticklabels([f"{tick:.1f}°" for tick in ax_marg_x.get_xticks()])
    ax_marg_y.set_yticks(ax_marg_y.get_yticks())
    ax_marg_y.set_yticklabels([f"{tick:.1f}°" for tick in ax_marg_y.get_yticks()])
    
    # Add grid to marginal plots and turn off ticks
    ax_marg_x.grid(True, linestyle='--', alpha=0.6)
    ax_marg_y.grid(True, linestyle='--', alpha=0.6)
    ax_marg_x.tick_params(bottom=False, labelbottom=False)
    ax_marg_y.tick_params(left=False, labelleft=False)
    
    ax_joint.set_xlim(ra_range)
    ax_joint.set_ylim(dec_range)
    
    # Remove axis lines and ticks from marginals
    for ax in [ax_marg_x, ax_marg_y]:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                       right=False, labelbottom=False, labelleft=False)

    ax_joint.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_luminosity_distance_marginal(dynesty_results_1, dynesty_results_2, colors=['red', 'blue'], labels=['H1 and L1', 'H1, L1 and V1']):
    """
    Plot normalized 1D probability distributions of luminosity distance (D_L)
    for two Dynesty posterior samples.

    Parameters
    ----------
    dynesty_results_1 : dynesty.results.Results
        First Dynesty result object.

    dynesty_results_2 : dynesty.results.Results
        Second Dynesty result object.

    colors : list of str
        Colors for each dataset's line and fill.

    labels : list of str
        Labels for each dataset used in the legend.
    """ 

    # Extract D_L samples from Dynesty results
    def get_d_l_samples(results):
        w = results.importance_weights()
        w /= np.sum(w)
        samples_eq = resample_equal(results.samples, w)
        return samples_eq[:, 4]  # Assuming D_L is at index 4

    d1 = get_d_l_samples(dynesty_results_1)
    d2 = get_d_l_samples(dynesty_results_2)

    fig, ax = plt.subplots(figsize=(6, 4))

    for d_l, color, label in zip([d1, d2], colors, labels):
        # KDE for D_L
        kde = gaussian_kde(d_l)
        x = np.linspace(d_l.min(), d_l.max(), 500)
        y = kde(x)
        y /= np.trapz(y, x)  # Normalize to area = 1

        ax.plot(x, y, color=color, label=label)

        # Histogram 
        ax.hist(d_l, bins=40, density=True, alpha=0.3, edgecolor='none', color=color)

        # Median and and sigma bounds
        median = np.median(d_l)
        lower = np.percentile(d_l, 16)
        upper = np.percentile(d_l, 84)
        ax.axvline(median, color=color, linestyle='--', linewidth=1)
        ax.axvspan(lower, upper, color=color, alpha=0.08, hatch='///', edgecolor=color)

    ax.set_xlabel(r"Luminosity Distance $D_L$ [Gpc]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', frameon=True)
    ax.tick_params(direction='in')

    plt.tight_layout()
    plt.show()


def convergence_dynesty(results_1, results_2, legend = ["H1 and L1", "H1, L1 and V1"]):
    """
    Plot a trace comparison of two Dynesty nested sampling results to assess convergence.

    Parameters
    ----------
    results_1 : dynesty.results.Results
        The first nested sampling result object (typically with fewer detectors).
    results_2 : dynesty.results.Results
        The second nested sampling result object (typically with more detectors).
    legend : list of str, optional
        Labels to use in the legend for each dataset, by default ["H1 and L1", "H1, L1 and V1"].

    Notes
    -----
    This function overlays trace plots of the posterior samples and log-evidence evolution
    from two Dynesty runs. It highlights medians and 68% credible intervals for each parameter,
    and sets log scaling on the distance axis for clearer visualization.
    """
    # Using dynesty's traceplot to visualize the convergence of the two results - the broader one should be placed first
    fig, axes = dyplot.traceplot(
        results_1,
        quantiles=None,
        show_titles=False,
        post_color = 'red',
        trace_cmap='hot',
        labels=[r"$\mathrm{RA}$", r"$\mathrm{Dec}$", r"$\psi$", r"$t_c^{\mathrm{(geo)}}$", r"$D_L$"],
    )

    # Add the second set of results to the same figure
    fig, axes = dyplot.traceplot(
        results_2,
        quantiles=None,
        show_titles=False,
        trace_cmap='winter',
        labels=[r"$\mathrm{RA}$", r"$\mathrm{Dec}$", r"$\psi$", r"$t_c^{\mathrm{(geo)}}$", r"$D_L$"],
        fig =(fig, axes),
    )

    # Set the Luminosity distance axes to log scale
    axes[4][0].set_yscale('log')
    axes[4][1].set_xscale('log')

    # Set the x-axis limits for the axes
    samples_1 = results_1.samples
    samples_2 = results_2.samples
    samples_1_weights = results_1.importance_weights()
    samples_1_weights /= np.sum(samples_1_weights)
    samples_2_weights = results_2.importance_weights()
    samples_2_weights /= np.sum(samples_2_weights)
    equal_weight_samples_1 = resample_equal(samples_1, samples_1_weights)
    equal_weight_samples_2 = resample_equal(samples_2, samples_2_weights)

    # Find rhe median of the samples for each parameter
    median_samples_1 = np.median(equal_weight_samples_1, axis=0)
    median_samples_2 = np.median(equal_weight_samples_2, axis=0)
    # Find the 16th and 84th percentiles for each parameter
    lower_samples_1 = np.percentile(equal_weight_samples_1, 16, axis=0)
    upper_samples_1 = np.percentile(equal_weight_samples_1, 84, axis=0)
    lower_samples_2 = np.percentile(equal_weight_samples_2, 16, axis=0)
    upper_samples_2 = np.percentile(equal_weight_samples_2, 84, axis=0)


    for i in range(5):
        # 10th percentile of both datasets
        min_val = min(np.percentile(equal_weight_samples_1[:, i], 10), np.percentile(equal_weight_samples_2[:, i], 10))
        # 90th percentile of both datasets
        max_val = max(np.percentile(equal_weight_samples_1[:, i], 90), np.percentile(equal_weight_samples_2[:, i], 90))
        # Set the x-axis limits for both axes and autoscale y-axis
        axes[i][1].set_xlim(min_val, max_val)
        axes[i][1].autoscale(enable=True, axis='y')
        # Plot the median (solid) and 16th and 84th (dashed) percentiles as vertical lines
        axes[i][1].axvline(median_samples_1[i], color='red', linestyle='-', linewidth=2)
        axes[i][1].axvline(median_samples_2[i], color='blue', linestyle='-', linewidth=2)
        axes[i][1].axvline(lower_samples_1[i], color='red', linestyle='--', linewidth=2)
        axes[i][1].axvline(upper_samples_1[i], color='red', linestyle='--', linewidth=2)
        axes[i][1].axvline(lower_samples_2[i], color='blue', linestyle='--', linewidth=2)
        axes[i][1].axvline(upper_samples_2[i], color='blue', linestyle='--', linewidth=2)


    # Add legends to each axis
    for i in range(5):
        axes[i][0].legend([legend[0], legend[1]], loc='upper right', fontsize=8)
        axes[i][1].legend([legend[0],  legend[1]], loc='upper right', fontsize=8)

    # Help formatting of the luminosity distance axes
    axes[4][1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axes[4][1].get_xaxis().set_minor_formatter(ticker.ScalarFormatter())
    axes[4][1].ticklabel_format(style='plain', axis='x')

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.tight_layout()
    plt.show()

def posterior_to_sky_area(results, sigma_levels=[1,2,3], title = 'Posterior from H1 and L1', grid_size=100):
    """
    Compute the sky area covered by confidence contours from posterior samples using 2D KDE and HEALPix pixelisation.

    Parameters
    ----------
    results : dynesty.results.Results
        Dynesty result object containing posterior samples and associated importance weights.

    sigma_levels : list of int, optional
        List of sigma levels (e.g., [1, 2, 3]) for which to compute confidence contours and sky areas.
        These are converted to corresponding confidence levels assuming a Gaussian distribution.

    title : str, optional
        Title used in printed output to identify the context (e.g., detector combination used).

    grid_size : int, optional
        Resolution of the 2D grid for evaluating the KDE over RA and Dec space.

    Returns
    -------
    None
        Prints a summary table showing the sky area (in square degrees) and the fractional sky coverage
        corresponding to each sigma level contour.
    """
    # Create evenly weighted samples
    # Extract samples and weights from Dynesty results    
    samples = results.samples
    weights = results.importance_weights()
    weights /= np.sum(weights)
    # Resample for equal-weight posterior 
    equal_samples = resample_equal(samples, weights)

    ra_samples = equal_samples[:, 0]
    dec_samples = equal_samples[:, 1]
    dl_samples = equal_samples[:, 4]
    data = np.vstack([ra_samples, dec_samples])
    
    ## ======= 2D KDE Confidence Contour =======
    
    # Calculate the Gaussian KDE distribution
    kde = gaussian_kde(data)

    # Create a grid for the KDE to be evaluated on
    x = np.linspace(ra_samples.min(), ra_samples.max(), grid_size)
    y = np.linspace(dec_samples.min(), dec_samples.max(), grid_size)
    X, Y = np.meshgrid(x, y)

    # Evaluate the KDE on the grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde_grid_values = kde(positions).reshape(X.shape).flatten()

    # Calculate the contour level for the specified confidence level
    kde_sorted = np.sort(kde_grid_values)[::-1]
    # Normalize the KDE values
    kde_cumsum = np.cumsum(kde_sorted)
    kde_cumsum /= kde_cumsum[-1]

    # Convert sigma levels to confidence levels ie 1 sigma = 68.27% confidence level
    confidence_levels = [norm.cdf(sigma) - norm.cdf(-sigma) for sigma in sigma_levels]

    # Get density thresholds
    threshold_idxs = [np.searchsorted(kde_cumsum, cl) for cl in confidence_levels]
    density_thresholds = [kde_sorted[idx] for idx in threshold_idxs]

    # === Healpy Area Comparison ===
    nside = 64
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), lonlat=False)

    kde_values = np.zeros(npix)
    for i in range(npix):
        ra, dec = phi[i], 0.5 * np.pi - theta[i]
        if (ra_samples.min() <= ra <= ra_samples.max()) and (dec_samples.min() <= dec <= dec_samples.max()):
            kde_values[i] = kde(np.array([[ra], [dec]]))[0]

        else:
            kde_values[i] = 0

    total_sky_area = 4 * np.pi * (180 / np.pi) ** 2

    # Count pixels above threshold
    rows = []
    for sigma, threshold in zip(sigma_levels, density_thresholds):
        mask = kde_values >= threshold
        area = np.sum(mask) * pixel_area
        fraction = area / total_sky_area
        rows.append({"σ": f"{sigma}σ", "Area [deg²]": area, "Fraction of Sky": fraction})

    df = pd.DataFrame(rows)

    print(f"\nSky areas covered for different sigma confidence levels in '{title}':")
    print(df.to_string(index=False, formatters={
        "Area [deg²]": "{:.2f}".format,
        "Fraction of Sky": "{:.4f}".format
    }))

    return None


def plot_Hubble_marginal(samples_1, samples_2, redshift, colors=['red', 'blue'], labels=['H1 and L1', 'H1, L1 and V1']):
    """
    Plot the inferred Hubble constant from two posterior sample sets and a common redshift.

    Parameters
    ----------
    samples_1 : ndarray
        Posterior samples array for the first detector combination. Should have luminosity distance in column 4 (in Gpc).
    samples_2 : ndarray
        Posterior samples array for the second detector combination. Same format as samples_1.
    redshift : float
        Redshift of the host galaxy.
    colors : list of str, optional
        Colors to use for the two plots. Default is ['red', 'blue'].
    labels : list of str, optional
        Labels for the legend. Default is ['H1 and L1', 'H1, L1 and V1'].
    """

    speed_of_light = 299792.458  # km/s

    fig, ax = plt.subplots(figsize=(7, 4))
    i = 0
    for samples, color, label in zip([samples_1, samples_2], colors, labels):
        i += 1
        dl_samples = samples[:, 4] * 1000  # Convert Gpc to Mpc
        z = np.full(dl_samples.shape, redshift)

        # Compute Hubble constant samples
        hubble_values = speed_of_light * z / dl_samples

        # KDE and histogram
        kde = gaussian_kde(hubble_values)
        x = np.linspace(hubble_values.min(), hubble_values.max(), 500)
        y = kde(x)
        y /= np.trapezoid(y, x)  # Normalize

        ax.plot(x, y, color=color, label=label)
        ax.hist(hubble_values, bins=40, density=True, alpha=0.2, edgecolor='none', color=color)

        # Median and 68% interval
        median = np.median(hubble_values)
        lower = np.percentile(hubble_values, 16)
        upper = np.percentile(hubble_values, 84)
        ax.axvline(median, color=color, linestyle='-', linewidth=1)
        ax.axvline(lower, color=color, linestyle='--', linewidth=1)
        ax.axvline(upper, color=color, linestyle='--', linewidth=1)
        if i == 1:
            ax.axvspan(lower, upper, color=color, alpha=0.1, hatch='///', edgecolor=color)
        else:
            ax.axvspan(lower, upper, color=color, alpha=0.1, hatch='\\\\', edgecolor=color)

        # Add annotated textbox
        textstr = f"{label}\nMedian: {median:.2f} km/s/Mpc\n68% CI: +{median-lower:.2f}–{upper-median:.2f}"
        ax.text(0.98, 0.95 - 0.15 * labels.index(label), textstr,
                transform=ax.transAxes, fontsize=9, color=color,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_xlabel(r"Hubble Constant $H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(direction='in')
    ax.set_xlim(np.percentile(hubble_values, 0), np.percentile(hubble_values, 100))
    plt.tight_layout()
    plt.show()
