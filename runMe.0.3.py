#!/usr/bin/env python3

"""
Runs all functions in baseFunctions.py on dataset.
If the brood flag is used, will also run functions in broodFunctions.py.
Brood processing functions have been moved to processBroodFunctions.py.
This version implements parallel processing.
"""

__appname__ = 'runMe.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu), editor August Easton-Calabria (eastoncalabr@wisc.edu)'
__version__ = '0.0.1'

# Set to True to enable cProfile profiling, which allows you to see how long each function takes to run, ie. where the bottleneck in the code is. This is just for debugging purposes.
ENABLE_CPROFILE = False

# imports
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, default='testCSV',
                        help='Directory containing data. Defaults to current working directory.')
    parser.add_argument('--extension', '-e', type=str, default='.csv',
                        help='String at end of all data files from tracking. Defaults to ".csv".')
    parser.add_argument('--brood', '-b', type=str, default=None,
                        help='Provide path to brood data to run brood functions.')
    parser.add_argument('--broodExtension', '-x', type=str, default='_nest_image.csv',
                        help='String at end of all brood data files. Defaults to "_nest_image.csv".')
    parser.add_argument('--whole', '-w', action='store_true',
                        help='Do not split frame into two when analyzing.')
    parser.add_argument('--bombus', '-z', action='store_true',
                        help='Data is from rig, run alternative search for data files.')
    parser.add_argument('--outFile', '-o', type=str, default='Analysis.csv',
                        help='Path to output file. Defaults to "Analysis.csv".')
    # August added these options:
    parser.add_argument('--interpolate', '-i', action='store_true',
                        help='Enable interpolation of missing data.')
    parser.add_argument('--remove-jumps', '-rj', type=int, default=None,
                        help='Minimum number of pixels a tag can jump between frames.')
    parser.add_argument('--real-fps', '-rfps', type=float, default=None,
                        help='Enter the framerate you were using.')
    parser.add_argument('--max-interpolation-seconds', '-mis', type=float, default=None,
                        help='Maximum number of seconds to interpolate between frames.')
    parser.add_argument('--save-interpolation-data', '-sid', type=bool, default=False,
                        help='Choose whether to save the interpolated data for debugging purposes.')
    parser.add_argument('--cores', '-c', type=int, default=1,
                        help='Number of CPU cores to use for parallel processing.')
    parser.add_argument('--limit', '-l', type=int, default=None,
                    help='Limit the number of files to process (for testing purposes).')

    return parser.parse_args()

def restructure_tracking_data(rawOneLR, opt, interpolated_path_name):
    """Take centroid data from aruco-tracking structured output, rearrange and interpolate missing data."""
    rawOneLR = rawOneLR.drop_duplicates(subset=['ID', 'frame'])
    if opt['interpolate'] == True:
        if opt.get('real_fps') is None or opt.get('max_interpolation_seconds') is None:
            raise ValueError("Interpolation is enabled but --real-fps and --max-interpolation-seconds must be provided.")
        max_seconds_gap = opt['max_interpolation_seconds']
        actual_frames_per_second = opt['real_fps']
        interpolated = data_cleaning.interpolate(rawOneLR, max_seconds_gap, actual_frames_per_second)
        if type(opt['remove_jumps']) == int:
            interpolated = data_cleaning.remove_jumps(interpolated)
        if opt['save_interpolation_data']:
            interpolated.to_csv(interpolated_path_name, index=False)
    else:
        if type(opt['remove_jumps']) == int:
            rawOneLR = data_cleaning.remove_jumps(rawOneLR)
        interpolated = rawOneLR
    xs = interpolated.pivot(index="frame", columns='ID', values=['centroidX', 'centroidY'])
    return xs

def processBrood(base, oneLR, name, ext, broodSource):
    # Reformat filename to build brood data path
    ext = '-nest_image.csv'
    print(ext)
    broodMapPath = os.path.join(broodSource, '_'.join(base.split('_')[0:2]).replace('-', '_') + ext)
    print(broodMapPath)
    if os.path.exists(broodMapPath):
        full = pd.read_csv(broodMapPath)
    else:
        print('Missing nest image data, did you mean to run brood functions?')
        return oneLR
    # Split based on LR values
    if name == 'Left':
        fullLeft = full[full['x'] < np.nanmean(oneLR['centroidX'].to_numpy())]
    elif name == "Right":
        fullRight = full[full['x'] > np.nanmean(oneLR['centroidY'].to_numpy())]
    brood = full[full['label'] != 'Arena perimeter (polygon)']
    eggs = brood[brood['radius'].isna()]
    allbrood = brood.dropna(axis=0)
    for i in set(eggs['object index']):
        try:
            egg = eggs[eggs['object index'] == i].reset_index()
            x = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.x
            y = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.y
            eggRow = pd.Series([i, egg.label[0],'polygon',x,y,np.nan])
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
            # Remove problematic rows
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
    distDF = processBroodFunctions.distanceFromCentroid(oneLR, allbrood)
    distDF2 = processBroodFunctions.minimumDistanceCircle(brood, oneLR)
    distDF3 = processBroodFunctions.minimumDistancePolygon(oneLR, eggs)
    return pd.concat([oneLR, distDF, distDF2, distDF3], axis=1)

def process_file(file_path, opt, funcs):
    """
    Process a single file and return a DataFrame of analysis results.
    Returns None if the file cannot be processed.
    """
    try:
        basename = os.path.basename(file_path)
        #print(f"Processing file: {basename}")
        #sys.stdout.flush()  # Force flush the output
        # For bombus data:
        if opt['bombus']:
            if 'mjpeg' in basename and os.path.exists(file_path.replace(".mjpeg", opt['extension'])):
                v = file_path
                print('Analyzing:', v)
                #workerID, Date, Time = basename.split("_")
                #Following two lines added by August to deal with formatting issues in filenames
                workerID, Date, Hours, Minutes, Seconds = basename.split("_")[0:5]
                Time = Hours + "-" + Minutes + "-" + Seconds
                #Time = Time.replace(".mjpeg", "").replace("-", ":")

                dtype_spec = {
                                "ID": "int64",
                                "frame": "int64",
                                "centroidX": "float64",
                                "centroidY": "float64"
                            }
                usecols = ["ID", "frame", "centroidX", "centroidY"]
                trackingResults = pd.read_csv(v.replace(".mjpeg", opt['extension']), dtype=dtype_spec, usecols=usecols)
            else:
                return None
        else:
            if opt['extension'] in basename:
                v = file_path
                print('Analyzing:', basename)
                #workerID, Date, Time = basename.split("_")
                #Following two lines added by August to deal with formatting issues in filenames
                workerID, Date, Hours, Minutes, Seconds = basename.split("_")[0:5]
                Time = Hours + "-" + Minutes + "-" + Seconds
                #Time = Time.replace(".mjpeg", "").replace("-", ":")
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

    if opt['whole']:
        trackingResults['LR'] = "Whole"
    else:
        trackingResults['LR'] = (trackingResults['centroidX'] < np.nanmean(trackingResults['centroidX'].to_numpy()))
        trackingResults.loc[trackingResults['LR'], 'LR'] = "Left"
        trackingResults.loc[trackingResults['LR'] != "Left", 'LR'] = "Right"

    fullAnalysis = pd.DataFrame()
    datasets = trackingResults.groupby('LR')
    for name, rawOneLR in datasets:
        data_path_name = os.path.join(os.path.dirname(v), os.path.splitext(basename)[0])
        interpolated_path_name = data_path_name + '_' + name + '_interpolated.csv'
        analysis = pd.DataFrame(index=rawOneLR.ID.unique())
        analysis['LR'] = name
        analysis['ID'] = analysis.index
        oneLR = restructure_tracking_data(rawOneLR, opt, interpolated_path_name)
        if opt['brood']:
            oneLR = processBrood(basename, oneLR, name, opt['broodExtension'], opt['brood'])
        for test in funcs:
            try:
                analysis[test[0]] = test[1](oneLR)
            except Exception as e:
                print(test[0] + " cannot be run on " + v)
                analysis[test[0]] = None
        fullAnalysis = pd.concat([fullAnalysis, analysis], axis=0)
    oneVid = pd.DataFrame(index=fullAnalysis.index)
    oneVid['pi_ID'] = workerID
    oneVid['bee_ID'] = oneVid.index
    oneVid['Date'] = Date
    oneVid['Time'] = Time
    oneVid = pd.concat([oneVid, fullAnalysis], axis=1)
    return oneVid

def main():
    """Main entry point of the program. Gathers all files from the source and processes them."""
    opt = vars(parse_opt())
    if os.path.exists(opt['outFile']):
        print('Found existing', opt['outFile'], 'and will add to it.')
        output = pd.read_csv(opt['outFile'])
    else:
        output = pd.DataFrame()

    funcs = [f for f in getmembers(baseFunctions) if isfunction(f[1]) and f[1].__module__ == 'baseFunctions']
    if opt['brood']:
        funcs += [f for f in getmembers(broodFunctions) if isfunction(f[1]) and f[1].__module__ == 'broodFunctions']

    # Build list of files to process
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

    # Limit the number of files processed if --limit is provided.
    if 'limit' in opt and opt['limit'] is not None:
        file_list = file_list[:opt['limit']]

    # Process files in parallel if cores > 1
    if opt['cores'] > 1:
        print("Processing files in parallel using", opt['cores'], "cores...")
        with Pool(processes=opt['cores']) as pool:
            results = pool.starmap(process_file, [(file_path, opt, funcs) for file_path in file_list])
            successful_results = [r for r in results if r is not None]
            print(f"Successfully processed {len(successful_results)} out of {len(results)} files.")

    else:
        print("Processing files serially...")
        results = [process_file(file_path, opt, funcs) for file_path in file_list]
        successful_results = [r for r in results if r is not None]
        print(f"Successfully processed {len(successful_results)} out of {len(results)} files.")


    results = [res for res in results if res is not None]
    if results:
        combined = pd.concat(results, ignore_index=True, axis=0)
        output = pd.concat([output, combined], ignore_index=True, axis=0)
    else:
        print("No files processed successfully.")
    output.to_csv(opt['outFile'], index=False)
    print("All done! Analysis saved to", opt['outFile'])
    return 0

if __name__ == "__main__":

    if ENABLE_CPROFILE == True:

        import cProfile
        import pstats
        import sys

        profiler = cProfile.Profile()
        status = 0  # default exit status
        try:
            profiler.enable()
            status = main()  # run your main processing function
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
        status = main()
        sys.exit(status)