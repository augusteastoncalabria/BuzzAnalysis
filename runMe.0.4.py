#!/usr/bin/env python3
"""
Runs all functions in baseFunctions.py on a dataset.
If the brood flag is used, it will also run functions in broodFunctions.py.
Brood processing functions have been moved to processBroodFunctions.py.
This version implements parallel processing and uses dtype/usecols for faster CSV reads,
as well as tqdm for progress reporting.
"""

__appname__ = 'runMe.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu), editor August Easton-Calabria (eastoncalabr@wisc.edu)'
__version__ = '0.0.4'

# Set to True to enable cProfile profiling (for debugging and bottleneck analysis)
ENABLE_CPROFILE = False

# Standard library and third-party imports
import pandas as pd
import numpy as np
import os
import sys
import argparse
from inspect import getmembers, isfunction
import baseFunctions
import processBroodFunctions
import broodFunctions
import warnings
import shapely
import data_cleaning
from multiprocessing import Pool
from tqdm import tqdm  # For progress reporting

# Suppress runtime and future warnings for clarity
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_opt():
    """
    Parse command-line options.
    Returns:
        args: Parsed command-line options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, default='testCSV',
                        help='Directory containing data. Defaults to current working directory.')
    parser.add_argument('--extension', '-e', type=str, default='.csv',
                        help='Suffix that identifies tracking data files. Defaults to ".csv".')
    parser.add_argument('--brood', '-b', type=str, default=None,
                        help='Path to brood data (if running brood functions).')
    parser.add_argument('--broodExtension', '-x', type=str, default='_nest_image.csv',
                        help='Suffix for brood data files. Defaults to "_nest_image.csv".')
    parser.add_argument('--whole', '-w', action='store_true',
                        help='Do not split frame into two when analyzing (process as a whole).')
    parser.add_argument('--bombus', '-z', action='store_true',
                        help='If set, use bombus-specific logic for data (e.g. MJPEG conversion).')
    parser.add_argument('--outFile', '-o', type=str, default='Analysis.csv',
                        help='Path to output file. Defaults to "Analysis.csv".')
    # Options added by August:
    parser.add_argument('--interpolate', '-i', action='store_true',
                        help='Enable interpolation of missing data.')
    parser.add_argument('--remove-jumps', '-rj', type=int, default=None,
                        help='Minimum number of pixels a tag can jump between frames.')
    parser.add_argument('--real-fps', '-rfps', type=float, default=None,
                        help='Frame rate used (required if interpolation is enabled).')
    parser.add_argument('--max-interpolation-seconds', '-mis', type=float, default=None,
                        help='Maximum number of seconds to interpolate between frames (required if interpolation is enabled).')
    parser.add_argument('--save-interpolation-data', '-sid', type=bool, default=False,
                        help='Save the interpolated data for debugging purposes.')
    parser.add_argument('--cores', '-c', type=int, default=1,
                        help='Number of CPU cores for parallel processing.')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit the number of files to process (for testing purposes).')
    return parser.parse_args()

def restructure_tracking_data(rawOneLR, opt, interpolated_path_name):
    """
    Rearranges and (optionally) interpolates the tracking data.
    
    Parameters:
        rawOneLR (DataFrame): Raw tracking data for one LR group.
        opt (dict): Parsed command-line options.
        interpolated_path_name (str): Path to save interpolated data (if enabled).
    
    Returns:
        DataFrame: Pivoted DataFrame with 'centroidX' and 'centroidY' for each tag.
    """
    # Remove duplicate rows based on 'ID' and 'frame'
    rawOneLR = rawOneLR.drop_duplicates(subset=['ID', 'frame'])
    
    if opt['interpolate'] == True:
        if opt.get('real_fps') is None or opt.get('max_interpolation_seconds') is None:
            raise ValueError("Interpolation is enabled but --real-fps and --max-interpolation-seconds must be provided.")
        max_seconds_gap = opt['max-interpolation-seconds']
        actual_frames_per_second = opt['real-fps']
        # Perform interpolation using data_cleaning.interpolate
        interpolated = data_cleaning.interpolate(rawOneLR, max_seconds_gap, actual_frames_per_second)
        if type(opt['remove_jumps']) == int:
            interpolated = data_cleaning.remove_jumps(interpolated)
        if opt['save-interpolation-data']:
            interpolated.to_csv(interpolated_path_name, index=False)
    else:
        if type(opt['remove_jumps']) == int:
            rawOneLR = data_cleaning.remove_jumps(rawOneLR)
        interpolated = rawOneLR
    
    # Pivot the data so that each tag's centroidX and centroidY appear as columns, keyed by 'frame'
    xs = interpolated.pivot(index="frame", columns='ID', values=['centroidX', 'centroidY'])
    return xs

def processBrood(base, oneLR, name, ext, broodSource):
    """
    Processes brood data for a file.
    
    Parameters:
        base (str): Basename of the file.
        oneLR (DataFrame): Processed tracking data for one LR group.
        name (str): Group name ('Left', 'Right', etc.).
        ext (str): Brood extension string (unused here).
        broodSource (str): Path to brood data.
    
    Returns:
        DataFrame: The tracking data augmented with brood distance metrics.
    """
    # Build the brood data file path using the basename and the brood extension.
    ext = '-nest_image.csv'
    print(ext)
    broodMapPath = os.path.join(broodSource, '_'.join(base.split('_')[0:2]).replace('-', '_') + ext)
    print(broodMapPath)
    
    # Read brood data; if desired, you could add dtype/usecols here as well (if structure is known)
    if os.path.exists(broodMapPath):
        full = pd.read_csv(broodMapPath)
    else:
        print('Missing nest image data, did you mean to run brood functions?')
        return oneLR
    
    # Split brood data based on LR values (using mean centroid as a threshold)
    if name == 'Left':
        fullLeft = full[full['x'] < np.nanmean(oneLR['centroidX'].to_numpy())]
    elif name == "Right":
        fullRight = full[full['x'] > np.nanmean(oneLR['centroidY'].to_numpy())]
    
    brood = full[full['label'] != 'Arena perimeter (polygon)']
    eggs = brood[brood['radius'].isna()]
    allbrood = brood.dropna(axis=0)
    
    # For each unique brood object, compute its centroid and append to the brood data.
    for i in set(eggs['object index']):
        try:
            egg = eggs[eggs['object index'] == i].reset_index()
            x = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.x
            y = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.y
            eggRow = pd.Series([i, egg.label[0], 'polygon', x, y, np.nan])
            eggRow.index = allbrood.columns
            allbrood = pd.concat([allbrood.T, eggRow], axis=1).T
        except Exception as e:
            print(e)
            with open('Error.csv', 'a') as errorFile:
                try:
                    errorFile.write(base + ', object ' + str(i) + ': ' + egg['label'][0] + '\n')
                    errorFile.write(str(e) + '\n')
                except Exception as e2:
                    errorFile.write('Cannot read label for ' + base + '\n')
            # Remove problematic rows from brood data
            if brood[brood['object index'] == i].shape[1] > 0:
                brood[brood['object index'] == i] = np.nan
                brood = brood.dropna(axis=0)
            if eggs[eggs['object index'] == i].shape[1] > 0:
                eggs[eggs['object index'] == i] = np.nan
                eggs = eggs.dropna(axis=0)
            if allbrood[allbrood['object index'] == i].shape[1] > 0:
                allbrood[allbrood['object index'] == i] = np.nan
                allbrood = allbrood.dropna(axis=0)
            continue
    allbrood = allbrood.reset_index()
    
    # Compute distance metrics using functions in processBroodFunctions
    distDF = processBroodFunctions.distanceFromCentroid(oneLR, allbrood)
    distDF2 = processBroodFunctions.minimumDistanceCircle(brood, oneLR)
    distDF3 = processBroodFunctions.minimumDistancePolygon(oneLR, eggs)
    return pd.concat([oneLR, distDF, distDF2, distDF3], axis=1)

def process_file(file_path, opt, funcs):
    """
    Process a single tracking file and return a DataFrame with analysis results.
    
    Steps:
      1. Read the CSV file using specified dtypes and usecols.
      2. Assign Left/Right (or Whole) labels.
      3. Restructure (and optionally interpolate) the tracking data.
      4. Optionally process brood data.
      5. Apply each analysis function from baseFunctions (and broodFunctions if enabled).
      6. Return a DataFrame including file metadata and analysis results.
    
    Returns:
        DataFrame: Analysis results for the file, or None if processing fails.
    """
    try:
        basename = os.path.basename(file_path)
        # For bombus data, use optimized CSV reading with dtype and usecols.
        if opt['bombus']:
            if 'mjpeg' in basename and os.path.exists(file_path.replace(".mjpeg", opt['extension'])):
                v = file_path
                #print('Analyzing:', v)
                # Parse the filename for metadata (first 5 parts: workerID, Date, Hours, Minutes, Seconds)
                workerID, Date, Hours, Minutes, Seconds = basename.split("_")[0:5]
                Time = Hours + "-" + Minutes + "-" + Seconds
                dtype_spec = {
                    "ID": "int64",
                    "frame": "int64",
                    "centroidX": "float64",
                    "centroidY": "float64"
                }
                usecols = ["ID", "frame", "centroidX", "centroidY"]
                trackingResults = pd.read_csv(file_path.replace(".mjpeg", opt['extension']),
                                              dtype=dtype_spec, usecols=usecols)
            else:
                return None
        else:
            # For non-bombus data, also use dtype and usecols for consistency.
            if opt['extension'] in basename:
                v = file_path
                #print('Analyzing:', basename)
                workerID, Date, Hours, Minutes, Seconds = basename.split("_")[0:5]
                Time = Hours + "-" + Minutes + "-" + Seconds
                dtype_spec = {
                    "ID": "int64",
                    "frame": "int64",
                    "centroidX": "float64",
                    "centroidY": "float64"
                }
                usecols = ["ID", "frame", "centroidX", "centroidY"]
                trackingResults = pd.read_csv(v, dtype=dtype_spec, usecols=usecols)
            else:
                return None
    except Exception as e:
        print('Error reading file', basename, ', skipping...')
        print(f"Error processing file {basename}: {e}")
        sys.stdout.flush()
        return None

    # Assign the LR label: if whole is enabled, set to "Whole"; otherwise, assign based on centroidX vs. mean.
    if opt['whole']:
        trackingResults['LR'] = "Whole"
    else:
        trackingResults['LR'] = (trackingResults['centroidX'] < np.nanmean(trackingResults['centroidX'].to_numpy()))
        trackingResults.loc[trackingResults['LR'], 'LR'] = "Left"
        trackingResults.loc[trackingResults['LR'] != "Left", 'LR'] = "Right"

    # Process data for each LR group (e.g., Left and Right)
    fullAnalysis = pd.DataFrame()
    datasets = trackingResults.groupby('LR')
    for name, rawOneLR in datasets:
        # Build a filename for saving interpolated data (if interpolation is enabled)
        data_path_name = os.path.join(os.path.dirname(v), os.path.splitext(basename)[0])
        interpolated_path_name = data_path_name + '_' + name + '_interpolated.csv'
        analysis = pd.DataFrame(index=rawOneLR.ID.unique())
        analysis['LR'] = name
        analysis['ID'] = analysis.index
        oneLR = restructure_tracking_data(rawOneLR, opt, interpolated_path_name)
        if opt['brood']:
            oneLR = processBrood(basename, oneLR, name, opt['broodExtension'], opt['brood'])
        # Apply each analysis function from baseFunctions (and broodFunctions if enabled)
        for test in funcs:
            try:
                analysis[test[0]] = test[1](oneLR)
            except Exception as e:
                print(test[0] + " cannot be run on " + v)
                analysis[test[0]] = None
        fullAnalysis = pd.concat([fullAnalysis, analysis], axis=0)
    
    # Create a DataFrame for the file that includes metadata (workerID, Date, Time) and analysis results.
    oneVid = pd.DataFrame(index=fullAnalysis.index)
    oneVid['pi_ID'] = workerID
    oneVid['bee_ID'] = oneVid.index
    oneVid['Date'] = Date
    oneVid['Time'] = Time
    oneVid = pd.concat([oneVid, fullAnalysis], axis=1)
    return oneVid

def main():
    """
    Main function:
      1. Parses command-line options.
      2. Loads existing output if available.
      3. Retrieves analysis functions from baseFunctions (and broodFunctions if enabled).
      4. Walks the source directory to build a list of files to process.
      5. Optionally limits the number of files (for testing).
      6. Processes files either in parallel or serially, reporting progress with tqdm.
      7. Combines all results and writes them to the output CSV.
    """
    opt = vars(parse_opt())
    
    # Load existing output analysis file if it exists to add to it
    if os.path.exists(opt['outFile']):
        print('Found existing', opt['outFile'], 'and will add to it.')
        output = pd.read_csv(opt['outFile'])
    else:
        output = pd.DataFrame()

    # Get analysis functions from baseFunctions and (if enabled) broodFunctions.
    funcs = [f for f in getmembers(baseFunctions) if isfunction(f[1]) and f[1].__module__ == 'baseFunctions']
    if opt['brood']:
        funcs += [f for f in getmembers(broodFunctions) if isfunction(f[1]) and f[1].__module__ == 'broodFunctions']

    # Walk the source directory to build a list of files to process.
    file_list = []
    for root, dirs, files in os.walk(opt['source']):
        for f in files:
            full_path = os.path.join(root, f)
            if opt['bombus']:
                if 'mjpeg' in f and os.path.exists(full_path.replace(".mjpeg", opt['extension'])):
                    file_list.append(full_path)
            else:
                if opt['extension'] in f:
                    file_list.append(full_path)
    print("Found", len(file_list), "files to process.")

    # If a limit is set, restrict the number of files processed (useful for testing).
    if opt.get('limit') is not None:
        file_list = file_list[:opt['limit']]

    # Process files using multiprocessing if more than 1 core is specified, with tqdm progress reporting.
    if opt['cores'] > 1:
        print("Processing files in parallel using", opt['cores'], "cores...")
        with Pool(processes=opt['cores']) as pool:
            # Use imap_unordered wrapped with tqdm to display progress. We use a lambda function because imap_unordered passes the arguments of the function as a tuple, but we need to pass multiple arguments to process_file.
            results = list(tqdm(
                            pool.imap_unordered(process_file_wrapper, [(file_path, opt, funcs) for file_path in file_list]),
                            total=len(file_list),
                            desc="Processing files"
                        ))

            successful_results = [r for r in results if r is not None]
            print(f"Successfully processed {len(successful_results)} out of {len(results)} files.")
    else:
        print("Processing files serially...")
        results = [process_file(file_path, opt, funcs) for file_path in tqdm(file_list, desc="Processing files")]
        successful_results = [r for r in results if r is not None]
        print(f"Successfully processed {len(successful_results)} out of {len(results)} files.")

    # Filter out None results and combine all successful ones.
    results = [res for res in results if res is not None]
    if results:
        combined = pd.concat(results, ignore_index=True, axis=0)
        output = pd.concat([output, combined], ignore_index=True, axis=0)
    else:
        print("No files processed successfully.")
    
    # Write the final combined output to the specified CSV file.
    output.to_csv(opt['outFile'], index=False)
    print("All done! Analysis saved to", opt['outFile'])
    return 0

def process_file_wrapper(args):
    return process_file(*args)    

if __name__ == "__main__":
    # If profiling is enabled, run with cProfile; otherwise, run normally.
    if ENABLE_CPROFILE:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        status = 0  # default exit status
        try:
            profiler.enable()
            status = main()
        except KeyboardInterrupt:
            print("Execution interrupted by user!")
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler).strip_dirs()
            print("\n--- Profiling Results (Top 20 by cumtime) ---")
            stats.sort_stats("cumtime").print_stats(20)
            print("\n--- Profiling Results (Top 20 by tottime) ---")
            stats.sort_stats("tottime").print_stats(20)
            sys.exit(status)
    else:
        start_time = pd.Timestamp.now()
        status = main()
        end_time = pd.Timestamp.now()
        print(f"Execution took {end_time - start_time}.")
        sys.exit(status)
