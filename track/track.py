#!/usr/bin/env python

#code to check for traients events in image cubes
#TRACK

#Update: include sourc RA DEC
#Filter based on RA DEC
#Plot sky map

#Kshitij July 2025
#---------------------
import warnings
import os
import argparse
import sys
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.ndimage import median_filter
from astropy.coordinates import Angle
from matplotlib.ticker import FuncFormatter
import ast
import datetime
from tqdm import tqdm 
import importlib.util
from plotly import graph_objects as go

warnings.simplefilter('ignore', category=AstropyWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

def ra_formatter(x, pos):
    ra_angle = Angle(x, unit=u.deg)
    return ra_angle.to_string(unit=u.hourangle, sep=':', precision=1, pad=True)

def dec_formatter(x, pos):
    dec_angle = Angle(x, unit=u.deg)
    return dec_angle.to_string(unit=u.deg, sep=':', precision=1, alwayssign=True, pad=True)

def extract_source_id(filename):
    match = re.search(r"source_(\d+)_cube\.fits", filename)
    return int(match.group(1)) if match else None

def read_central_lightcurve(fits_path):
    # try:
    with fits.open(fits_path) as hdul:
        #Central pixel
        header = hdul[0].header
        crpix1 = header['CRPIX1']
        crval1 = header['CRVAL1']
        #cdelt1 = header['CDELT1']

        crpix2 = header['CRPIX2']
        crval2 = header['CRVAL2']
        #cdelt2 = header['CDELT2']

        x=crpix1
        y=crpix2
        #x,y=20,20

            
        # pixel_area = abs(header['CDELT1']) * abs(header['CDELT2'])
        # # print(f'pixel area in arcseconds = {pixel_area * 3600} arcsec^2')
        # beam_area = 1.133 * header['BMAJ'] * header['BMIN']

        # factor = pixel_area / beam_area

        ra = crval1
        dec = crval2

        #Lightcurve
        data = hdul[0].data
        size = 10
        offset = 40
        lightcurve = data[:, y-size:y+size, x-size:y+size]
        lightcurve = np.nansum(lightcurve,axis=(1,2))
        # lightcurve = lightcurve*factor

        error = data[:, y-size+offset:y+size+offset,x-size+offset:x+size+offset]
        error = np.nansum(error,axis=(1,2))
        # error = np.abs(error)/
        # error = error*factor

        lightcurve = np.where(np.isnan(lightcurve), np.nanmedian(lightcurve), lightcurve)
        error = np.where(np.isnan(error), np.nanmedian(error), error)
        return lightcurve, error, data, ra, dec
        
    # except Exception as e:
    #     print(f"Error reading {fits_path}: {e}")
    #     return None

### Define function to search traneint events
def detect_transients(lightcurve, error, n_sigma):
    if lightcurve is None or len(lightcurve) == 0:
        return []

    # Compute robust baseline
    median = np.median(error)
    mad = np.median(np.abs(lightcurve - median))
    sigma = 1.4826 * mad
    threshold = median + n_sigma * sigma
    #print(f'Median:{median}')
    #print(f'Threshold: {threshold}')

    # Get all indices above threshold
    transient_indices = np.where(lightcurve >= threshold)[0]
    if len(transient_indices) == 0:
        return []


    # Group indices allowing small gaps
    max_gap=4
    events = []
    start = transient_indices[0]
    for i in range(1, len(transient_indices)):
        if transient_indices[i] > transient_indices[i - 1] + max_gap:
            end = transient_indices[i - 1]
            events.append((start, end))
            start = transient_indices[i]
    events.append((start, transient_indices[-1]))

    # Gather event info
    result = []
    for start, end in events:
        duration = end - start + 1
        if duration == 1 or start < 5 or start > len(lightcurve) - 5 :
           continue  # Skip 1-index-long events, skip events in the beginning and end of the cube

        
        peak_flux = np.max(lightcurve[start:end+1])
        snr = (peak_flux - median) / sigma
        snr = round(snr, 3) 
        
        result.append({
            "start_index": start,
            "end_index": end,
            "peak_flux": peak_flux,
            "snr" : snr,
            "duration": duration
        })
    return result


### Define function for plotting
def plot_lightcurve_with_events(lightcurve, error, source_id, events, output_dir):
    plt.figure(figsize=(10, 5))
    #plt.figure(figsize=(10, 5), constrained_layout=True)
    plt.plot(lightcurve, label='Flux', color='tab:blue', zorder = 2)
    # plt.plot(error, label = 'Error', color='tab:green')
    plot_error = np.abs(error)
    plt.errorbar(np.arange(len(lightcurve)), lightcurve, yerr=plot_error, fmt='none', ecolor='green', alpha=0.5, zorder = 1)


    if len(events) > 0:

        # Set Y-axis limits around the peak
        peak_fluxes = [e["peak_flux"] for e in events]
        max_peak = max(peak_fluxes)
        y_min = np.nanmin(lightcurve)
        y_max = max_peak * 1.2
        plt.ylim(y_min, y_max)
        
        for idx, event in enumerate(events):
            start, end = event["start_index"], event["end_index"]
            peak = event["peak_flux"]

            #Highlight
            plt.axvspan(start, end, color='red', alpha=0.3)
            plt.plot((start + end) / 2, peak, 'ro')

            #Label
        # plt.text((start + end) / 2, peak + 0.02 * np.nanmax(lightcurve), f'#{idx + 1}', color='black', fontsize=9, ha='center', va='bottom')
            plt.text((start + end) / 2, peak + 0.02 * max_peak,
                    f'#{idx + 1}', color='black', fontsize=9, ha='center', va='bottom')


    plt.xlabel("Time Slice Index")
    plt.ylabel("Flux (Jy/beam)")
    plt.ylim([-0.5,4])
    plt.title(f"Lightcurve for Source {source_id} (Central Pixel)")
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    out_path = os.path.join(output_dir, f"source_{source_id}_lightcurve.png")
    plt.savefig(out_path)
    plt.close()

### Define function to get event GIF
def generate_event_animation(data, source_id, event_index, event, output_dir, buffer=10):
    start = max(0, event["start_index"] - buffer)
    end = min(data.shape[0] - 1, event["end_index"] + buffer)

    vmin = np.nanpercentile(data[start:end+1], 5)
    vmax = np.nanpercentile(data[start:end+1], 99)

   
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = 'plasma'  # Try 'viridis', 'inferno', or 'magma' for other options

    img = ax.imshow(data[start], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    title = ax.set_title(f"Source {source_id} – Slice {start}")
    ax.axis('off')

    #cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label('Jy/beam')

    # Frame update function
    def update(t):
        img.set_data(data[t])
        title.set_text(f"Source {source_id} – Slice {t}")
        return img, title

    ani = animation.FuncAnimation(
        fig, update, frames=range(start, end + 1),
        interval=500,  # ms
        blit=False
    )

    # Save as .gif
    out_path = os.path.join(output_dir, f"source_{source_id}_event_{event_index+1}.gif")
    ani.save(out_path, writer='pillow')
    plt.close(fig)

#### Define Running  Median filter
def detrend(lightcurve, window_fraction):
    if window_fraction is None or window_fraction == 0:
       detrended = lightcurve  # no subtraction
    else:
       window_size = int(len(lightcurve) * window_fraction / 100)
       if window_size % 2 == 0:
          window_size += 1
       window_size = max(window_size, 3)

       baseline = median_filter(lightcurve, size=window_size)
       detrended = lightcurve - baseline
    return detrended


### Function to filter candideta based on RA and DEC and plot sky map
def filter_and_plot_candidates(df, output_csv, output_plot, max_sep_arcsec):
    print("Filtering the candidates based on sky coordinates...")

    # Convert list-like strings to actual lists
    for col in ['source_RA', 'source_DEC', 'peak_snr']:
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # # xplode all list columns so each event is its own row
    df_exploded = df.explode(['source_RA', 'source_DEC', 'peak_snr'])
    # df_exploded = df_exploded.dropna(subset=['peak_snr'])
    df_exploded['peak_snr'] = pd.to_numeric(df_exploded['peak_snr'], errors='coerce')
    df_exploded = df_exploded.dropna(subset=['peak_snr'])

    # Compute sky coordinates for all exploded rows
    coords = SkyCoord(ra=df_exploded['source_RA'].values * u.deg, dec=df_exploded['source_DEC'].values * u.deg)

    # Find close groups using search_around_sky
    idx1, idx2, _, _ = coords.search_around_sky(coords, max_sep_arcsec * u.arcsec)

    #  Disjoint Set Union (Union-Find) for group assignment
    parent = np.arange(len(df_exploded))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i, j in zip(idx1, idx2):
        union(i, j)

    group_ids = [find(i) for i in range(len(df_exploded))]
    df_exploded['group_id'] = group_ids

    #  Filter: keep only the best SNR event in each group
    df_filtered = df_exploded.loc[df_exploded.groupby('group_id')['peak_snr'].idxmax()]
    df_filtered = df_filtered.sort_values(by='source_id')
    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered candidates saved to: {output_csv}")

    #####################  Plotting
    # create mapping from original group IDs to sequential IDs
    unique_group_ids = sorted(df_exploded['group_id'].unique())
    group_id_map = {old: new for new, old in enumerate(unique_group_ids)}
    df_exploded['plot_group_id'] = df_exploded['group_id'].map(group_id_map)

    plt.figure(figsize=(10, 8))
    cmap = plt.colormaps['tab20'].resampled(len(unique_group_ids))

    for idx, gid in enumerate(unique_group_ids):
        group_df = df_exploded[df_exploded['group_id'] == gid]
        color = cmap(idx)

        # Plot all in group
        plt.scatter(group_df['source_RA'], group_df['source_DEC'], s=20, color=color, label=f'Group {idx}')

        # Highlight selected candidate
        max_snr_source = df_filtered[df_filtered['group_id'] == gid]
        plt.scatter(max_snr_source['source_RA'], max_snr_source['source_DEC'],
                    s=80, color=color, marker='*', edgecolor='black', linewidth=1.2)

        # Annotate with source IDs
        for _, row in group_df.iterrows():
            plt.text(row['source_RA'] + 0.002, row['source_DEC'], str(int(row['source_id'])), fontsize=6)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(ra_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(dec_formatter))
    plt.xlabel("RA (H:M:S)")
    plt.ylabel("DEC (D:M:S)")
    plt.title("Grouped Transient Candidates by Sky Position")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=6)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Sky map saved to: {output_plot}")

    return df_filtered

        
#### Process the directory of image cubes
def process_fits_directory(fits_dir, n_sigma, w, sep):

    output_csv = os.path.join(fits_dir, 'transient_candidates.csv')
    output_filtered_csv = os.path.join(fits_dir, 'transient_candidates_filtered.csv')
    output_plot = os.path.join(fits_dir, 'transient_groups_skyplot.png')

    print(f"Processing cubes in : {fits_dir}")

    rows = []
    for fname in sorted(os.listdir(fits_dir)):
        if fname.endswith(".fits") and "source_" in fname:
            source_id = extract_source_id(fname)
            if source_id is None:
                continue
            print(f"Source {source_id}")

            fpath = os.path.join(fits_dir, fname)
            lightcurve , error, data, ra, dec = read_central_lightcurve(fpath)

            #Running Median subtraction
            # detrended_lightcurve = detrend(lightcurve, w) #Median sub, window size %, 0 for no filtering
            
            #Search transients
            events = detect_transients(lightcurve, error, n_sigma)

            if events:
               print('Transient events detected. Generating lightcurve and saving details')
               event_starts = [e["start_index"] for e in events]
               event_ends = [e["end_index"] for e in events]
               peak_fluxes = [e["peak_flux"] for e in events]
               snrs = [e["snr"] for e in events]
               durations = [e["duration"] for e in events]

               rows.append({
                   "source_id": source_id,
                   "source_RA" : ra,
                   "source_DEC" : dec,
                   "num_events": len(events),
                   "event_starts": event_starts,
                   "event_ends": event_ends,
                   "durations": durations,
                   #"peak_fluxes": peak_fluxes,
                   "peak_snr" : snrs
                    })

               #Create animations
               for i, event in enumerate(events):
                   generate_event_animation(data, source_id, i, event, fits_dir)
            
            #plot lightcurve
            plot_lightcurve_with_events(lightcurve, error, source_id, events, fits_dir)

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(by='source_id', inplace=True)
        df.to_csv(output_filtered_csv, index=False)
        # print(f"Results written to {output_csv}")

        # #Filtering and plotting
        # filter_and_plot_candidates(df, output_filtered_csv,output_plot, sep)
        
    else:
        print("No transients detected in any FITS cube.")

def process_arguments(job_dir, outdir):

    if os.path.exists(outdir):
        print("Output directory already exists, do you want to delete it and rerun cropping? (y/n)")
        rerun = input().strip().lower()
    
    else:

        rerun = 'y'

    if rerun == 'y':
            os.system(f"rm -rf {outdir}")
            os.makedirs(outdir)

            image_gen_script = os.path.join(job_dir, 'images_to_process.py')
            spec = importlib.util.spec_from_file_location("images_to_process", image_gen_script)
            images_to_process = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(images_to_process)

            cube_slices = images_to_process.images

            return cube_slices, outdir, rerun 

    elif rerun == 'n':
        return None, outdir, rerun
    else:
        print("Invalid option. Exitting.")
        sys.exit(0)
        

def load_sources(trap_output,nimg):
    df = pd.read_csv(trap_output, delimiter= ', ')
    mask = df['datapoints'] > 1 
    mask &= df['datapoints'] < 0.9*nimg
    new_df = (df[['wm_ra (deg)', 'wm_decl (deg)', 'flux max']][mask].rename(
    columns={'wm_ra (deg)':'source_RA', 'wm_decl (deg)':'source_DEC', 'flux max':'peak_snr'}).reset_index().rename(columns={'index':'source_id'}))
    return new_df, df.index[mask].values

def process_cube_slice(cube_slice, sources, crop_radius):
    
    with fits.open(cube_slice, memmap=False) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        data = np.squeeze(data)
        
    wcs = WCS(header)
    time = Time(header['DATE-OBS'], format='isot', scale='utc')
    naxis1, naxis2 = header['NAXIS1'], header['NAXIS2']
    
    # Batch convert all source coordinates to pixel positions
    coords = SkyCoord(ra=sources['source_RA'].values*u.deg, dec=sources['source_DEC'].values*u.deg)
    x, y = coords.to_pixel(wcs)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    
    # Vectorized boundary checking
    valid = (x >= crop_radius) & (x < naxis1 - crop_radius) & \
            (y >= crop_radius) & (y < naxis2 - crop_radius)
    
    # Crop all valid sources in one pass
    crops = []
    for i in np.where(valid)[0]:
        xi, yi = x[i], y[i]
        crop = data[yi-crop_radius:yi+crop_radius, xi-crop_radius:xi+crop_radius]
        ra = sources['source_RA'].values[i]
        dec = sources['source_DEC'].values[i]
        crops.append((i, crop, data[yi, xi], ra, dec))
        
    return crops, time, header

#========================= Main ================================#
def main():
    parser = argparse.ArgumentParser(
        description=(""" TRACK ---
        Detect transients in image cubes.
        This will find transients using MAD based thresholding for all the image cubes in given directroy using central pixel lightcurve. 
        For the cube in which traneints are dteceted, a lightcurve PNG will be saved
         and for every transient a GIF will be saved.
        A CSV file summarising all detection will also be saved. Another CSV file with sources filtered by sky position will also be written
         and a plot will be generated. 
        A log file for the code run will be saved with input values.
        (Kshitij July 2025 for the LPT project @NCRA)
        """),
        epilog=("""Example:\n  track  path/to/trap.csv path/to/trap_job_dir complete/path/to/cubes --sigma 10 --w 0 --sep 20
        """)
    )
    

    # Required argument
    parser.add_argument(
        "trap_output",
        type=str,
        help="Path to trap output csv (.csv)"
    )

    parser.add_argument(
        "trap_job_dir",
        type=str,
        help="Path to trap job directory"
    )

    parser.add_argument(
        "output_fits_directory",
        type=str,
        help="Path to the directory containing image cubes (.fits files)"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--crop_radius",
        type=int,
        default=20,
        help="Crop radius in pixels to crop cubes (default: 20)"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help="Threshold in units of sigma for transient detection (default: 10.0)"
    )

    parser.add_argument(
        "--w",
        type=float,
        default=0,
        help="Fraction of lightcurve length in percent used as window size for running median filter.This can be used to detrend the timeseries. Use 0 to disable (default 0)."
    )

    parser.add_argument(
         "--sep",
         type=float,
         default=30,
         help="Max angular separtion in arcsec betwenn sources for grouping (default 30)."
     )

    
    args = parser.parse_args()

    #Inputs
    trap_output = args.trap_output
    trap_job_dir = args.trap_job_dir
    fits_directory = args.output_fits_directory
    crop_radius = args.crop_radius
    sigma = args.sigma
    w = args.w
    max_sep_arcsec = args.sep

    cube_slices, outdir , rerun = process_arguments(trap_job_dir, fits_directory)

    # Get timestamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(fits_directory, 'transient_candidates.csv')
    output_filtered_csv = os.path.join(fits_directory, 'transient_candidates_filtered.csv')
    output_plot = os.path.join(fits_directory, 'transient_groups_skyplot.png')
    #log contents
    log_contents = (
       f"TRACK Transient Detection Run Log\n"
      f"Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
      f"Trap output: {trap_output}\n"
      f"Trap job directory: {trap_job_dir}\n"
      f"Output Directory: {fits_directory}\n"
      f"Crop radius for cubes (pixels): {crop_radius}\n"
      f"Sigma Threshold: {sigma}\n"
      f"Running Median Window Fraction: {w}\n"
      f"Max Separation (arcsec): {max_sep_arcsec}\n"
    )

    # Log file 
    log_file_path = os.path.join(fits_directory, f"track_runlog_{timestamp_str}.txt")
    with open(log_file_path, "w") as f:
      f.write(log_contents)

    print(f"Log saved to {log_file_path}")

    if rerun == 'y':
    
        nimg = len(cube_slices)
        sources, index = load_sources(trap_output, nimg)

        sources = filter_and_plot_candidates(sources, output_csv, output_plot, max_sep_arcsec)

        # print(sources['source_id'].values)

        # sys.exit()
        
        # Initialize storage for all sources
        source_data = {i: {'cubes': [], 'times': [], 'fluxes': [], 'ra': [], 'dec': [], 'index': sources['source_id'].values[i]}
                    for i in range(len(sources))}
        first_header = None
        
        for cube_slice in tqdm(cube_slices, desc="Processing cubes"):
            crops, time, header = process_cube_slice(cube_slice, sources, crop_radius)
            if first_header is None:

                if header['BMAJ'] > 0.0 and header['BMIN'] > 0.0:
                    # print(f"Beam info found in header: BMAJ={header['BMAJ']}, BMIN={header['BMIN']}")
                    first_header = header.copy()
                
            for i, crop, flux, ra, dec in crops:
        
                source_data[i]['cubes'].append(crop)
                source_data[i]['times'].append(time)
                source_data[i]['fluxes'].append(flux)
                source_data[i]['ra'].append(ra)
                source_data[i]['dec'].append(dec)

        print(f"Found {len(source_data)} sources in the data.")
            
        # Save all sources
        for i in tqdm(source_data, desc="Saving results"):
            if not source_data[i]['cubes']:
                continue
                
            # Create time cube
            time_cube = np.array(source_data[i]['cubes'])
            times = Time(source_data[i]['times'])
            
            # Update header
            hdr = first_header.copy()
            hdr['CRVAL1'] = source_data[i]['ra'][0]
            hdr['CRVAL2'] = source_data[i]['dec'][0]
            hdr['CRPIX1'] = crop_radius
            hdr['CRPIX2'] = crop_radius
            hdr['CTYPE3'] = 'TIME'
            hdr['CRPIX3'] = 1
            hdr['CDELT3'] = 1
            hdr['PV3_1'] = 1

            del hdr['CRVAL4']
            del hdr['CRPIX4']
            del hdr['CTYPE4']
            del hdr['CDELT4']  
            del hdr['CUNIT4']     

            col = fits.Column(name='TIME', format='D', unit='s', array=times.mjd)
            table = fits.BinTableHDU.from_columns([col])
            table.header['TUNIT1'] = 'mjd'
            table.header['TIMESYS'] = 'UTC'
            
            # Save FITS cube
            id = source_data[i]['index']

            # fits.PrimaryHDU(time_cube, header=hdr).writeto(f"{outdir}/source_{id+2}_cube.fits", overwrite=True)

            primary_hdu = fits.PrimaryHDU(time_cube, header=hdr)
            hdul = fits.HDUList([primary_hdu, table])
            hdul.writeto(f"{outdir}/source_{id+1}_cube.fits", overwrite=True)
            hdul.close()
            
            # Save time series plot
            # plt.plot(times.mjd, source_data[i]['fluxes'])
            # plt.title(f"Source {i+1} Time Series")
            # plt.xlabel("MJD")
            # plt.ylabel("Flux")
            # plt.savefig(f"{outdir}/source_{i+1}_time_series.png")
            # plt.close()

            fig = go.Figure()
            fluxes = np.array(source_data[i]['fluxes'])
            fluxes[fluxes > 100] = np.nan  # Set fluxes greater than 100 to NaN
            fluxes[fluxes < -100] = np.nan  # Set negative fluxes to NaN
            fig.add_trace(go.Scatter(x=times.mjd, y= fluxes, mode='lines+markers', name='Flux'))
            fig.update_layout(
                title=f"Source {id+1} Time Series",
                xaxis_title="MJD",
                yaxis_title="Flux",
                template="plotly_white"
            )
            # Save the plot as an interactive HTML file
            fig.write_html(f"{outdir}/source_{id+1}_time_series.html")
        #Process directory

    process_fits_directory(fits_directory, sigma, w, max_sep_arcsec)

#-----------------------
if __name__ == "__main__":
    main()
