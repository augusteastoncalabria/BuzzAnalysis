#!/usr/bin/env python3

"""
Runs all functions in baseFunctions.py on dataset.
If the brood flag is used, will also run functions in broodFunctions.py
"""

__appname__ = 'runMe.py'
__author__ = 'Acacia Tang (ttang53@wisc.edu)'
__version__ = '0.0.1'

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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str, default='testCSV', help='Directory containing data. Defaults to current working directory.')
    parser.add_argument('--extension', '-e', type=str, default='.csv', help='String at end of all data files from tracking. Defaults to "_updated.csv".')
    parser.add_argument('--brood', '-b', type=str, default=None, help='Provide path to brood data to run brood functions.')
    parser.add_argument('--broodExtension', '-x', type=str, default='_nest_image.csv', help='String at end of all data files (must be CSVs) containing brood data. Defaults to "_nest_image.csv".')
    parser.add_argument('--whole', '-w', action='store_true', help='Do not split frame into two when analyzing.')
    parser.add_argument('--bombus', '-z', action='store_true', help='Data is from rig, run alternative search for data files.')
    parser.add_argument('--outFile', '-o', type=str, default='Analysis.csv', help='Path to output file. Defaults to "Analysis.csv".')
    #August added this
    parser.add_argument('--interpolate', '-i', type=bool, default=True, help='Choose whether to interpolate missing data.')
    parser.add_argument('--remove-jumps', '-rj', type=int, default=None, help='If you want to remove jumps in tag tracking between data, set the minimum number of pixels a tag can be from one frame to the next.')
    parser.add_argument('--real-fps', '-rfps', type=float or int, help='Enter the framerate you were using (with BumbleBox software using mp4 files, you have to test for the actual framerate!).')
    parser.add_argument('--max-interpolation-seconds', '-mis', type=float or int, help='Maximum number of seconds to interpolate between frames.')
    parser.add_argument('--save-interpolation-data', '-sid', type=bool, default=False, help='Choose whether to save the interpolated data for debugging purposes.')
    #End of August's addition
    return parser.parse_args()

def restructure_tracking_data(rawOneLR, opt, interpolated_path_name):
    """Take centroid data from aruco-tracking structured output, rearrange and interpolate missing data"""
    # Drop any duplicate rows
    rawOneLR = rawOneLR.drop_duplicates(subset=['ID', 'frame'])
    #August added this
    if opt['interpolate'] == True:
        max_seconds_gap = opt.get('max_interpolation_seconds')
        actual_frames_per_second = opt.get('real_fps')
        interpolated = data_cleaning.interpolate(rawOneLR, max_seconds_gap, actual_frames_per_second)
        if type(opt['remove_jumps']) == int:
            interpolated = data_cleaning.remove_jumps(interpolated)
        if opt['save_interpolation_data'] == True:
            interpolated.to_csv(interpolated_path_name, index=False)
    else:
        if type(opt['remove_jumps']) == int:
            rawOneLR = data_cleaning.remove_jumps(rawOneLR)
        interpolated = rawOneLR
    #End of August's addition
    xs = interpolated.pivot(index="frame", columns='ID', values=['centroidX', 'centroidY'])

    #August commented this out
    #return xs.interpolate(method='linear', limit=2, axis='index', limit_direction='both', limit_area='inside')
    return xs

def processBrood(base, oneLR, name, ext, broodSource):
    #Old
    ### 
    #os.path.join(broodSource, '_'.join(base.split('_')[0:2]) + ext
    #if os.path.exists(broodMapPath):
    ### 
    
    #Replaced with the following to deal with filename formatting issues
    ### 
    ext = '-nest_image.csv'
    print(ext)
    broodMapPath = os.path.join(broodSource, '_'.join(base.split('_')[0:2]).replace('-', '_') + ext)
    print(broodMapPath)
    
    if os.path.exists(broodMapPath):
    ###
    
        full = pd.read_csv(broodMapPath)
    else:
        print('Missing nest image data, did you mean to run brood functions?')
        return oneLR
    #split
    if name == 'Left':
        fullLeft = full[full['x'] < np.nanmean(oneLR['centroidX'].to_numpy())]
    elif name == "Right":
        fullRight = full[full['x'] > np.nanmean(oneLR['centroidY'].to_numpy())]

    #Don't care about areana
    brood = full[full['label'] != 'Arena perimeter (polygon)']

    #centroid: all
    eggs = brood[brood['radius'].isna()]
    allbrood = brood.dropna(axis =0)
    for i in set(eggs['object index']):
        try:
            egg = eggs[eggs['object index'] == i].reset_index()
            x = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.x
            y = shapely.Polygon(np.array(egg[['x', 'y']])).centroid.y
            eggRow = pd.Series([i, egg.label[0],'polygon',x,y,np.nan])
            eggRow.index = allbrood.columns
            allbrood = pd.concat([allbrood.T,eggRow],axis=1).T
        except Exception as e:
            print(e)
            errorFile = open('Error.csv', 'a')
            try:
                errorFile.write(base + ', object ' + str(i) + ': ' + egg['label'][0] + '\n')
                errorFile.write(str(e) + '\n')
            except:
                errorFile.write('Cannot read label, printing entire list of objects' + '\n')
                errorFile.write(egg)
                errorFile.write(str(e) + '\n')
                                
                
            errorFile.close()
            if brood[brood['object index'] == i].shape[1] > 0:
                brood[brood['object index'] == i] = np.nan
                brood = brood.dropna(axis = 0)
            if eggs[eggs['object index'] == i].shape[1] > 0:
                eggs[eggs['object index'] == i] = np.nan
                eggs = eggs.dropna(axis = 0)
            if allbrood[allbrood['object index'] == i].shape[1] > 0:
                allbrood[allbrood['object index'] == i] = np.nan
                allbrood = allbrood.dropna(axis = 0)
            continue


    allbrood = allbrood.reset_index()

    distDF = processBroodFunctions.distanceFromCentroid(oneLR, allbrood)
    distDF2 = processBroodFunctions.minimumDistanceCircle(brood, oneLR)
    distDF3 = processBroodFunctions.minimumDistancePolygon(oneLR, eggs)

    return pd.concat([oneLR, distDF, distDF2, distDF3], axis=1)

def main():
    """Main entry point of program. For takes in the path to a folder and a list of functions to run. Results will be written to Analysis.csv in the current directory."""
    opt = vars(parse_opt())
    if os.path.exists(opt['outFile']):
        print('I found a file named ' + opt['outFile'] + ' in the current working directory and will be adding to the file.')
        output = pd.read_csv(opt['outFile'])
    else:
        output = pd.DataFrame()
    funcs = [f for f in getmembers(baseFunctions) if isfunction(f[1]) and f[1].__module__ == 'baseFunctions']
    if opt['brood']:
        funcs = funcs + [f for f in getmembers(broodFunctions) if isfunction(f[1]) and f[1].__module__ == 'broodFunctions']

    for dir, subdir, files in os.walk(opt['source']):
        for f in files:
            try:
                if opt['bombus']:
                    
                    if 'mjpeg' in f and os.path.exists(os.path.join(dir, f).replace(".mjpeg", opt['extension'])):
                                v = os.path.join(dir, f)
                                print('Analyzing: ' + v)
                                workerID, Date, Time = f.split("_")
                                #Following two lines added by August to deal with formatting issues in filenames
                                #workerID, Date, Hours, Minutes, Seconds = f.split("_")[0:5]
                                #Time = Hours + "-" + Minutes + "-" + Seconds

                                Time = Time.replace(".mjpeg", "").replace("-", ":")
                                trackingResults = pd.read_csv(v.replace(".mjpeg", opt['extension']))
                    else:
                        continue
                else:
                    if opt['extension'] in f:
                        v = os.path.join(dir, f)
                        print('Analyzing: ' + f)
                        workerID, Date, Time = f.split("_")[0:3]

                        #Following two lines added by August to deal with formatting issues in filenames
                        #workerID, Date, Hours, Minutes, Seconds = f.split("_")[0:5]
                        #Time = Hours + "-" + Minutes + "-" + Seconds
                        
                        Time = Time.replace(opt['extension'], "").replace("-", ":")
                        trackingResults = pd.read_csv(v)
                    else:
                        continue
                    
            except Exception as e:
                print('Error reading file ' + f + ', skipping...')
                continue

            if opt['whole']:
                trackingResults['LR'] = "Whole"
            else:
                trackingResults['LR'] = (trackingResults['centroidX'] < np.nanmean(trackingResults['centroidX'].to_numpy()))
                trackingResults.loc[trackingResults['LR'], 'LR'] = "Left"
                trackingResults.loc[trackingResults['LR'] != "Left", 'LR'] = "Right"

            fullAnalysis = pd.DataFrame()
            datasets = trackingResults.groupby('LR')

            for name, rawOneLR in datasets:
                #August added this to allow us to save the interpolated data separately for debugging
                data_path_name = os.path.join(dir, f)
                data_path_name = os.path.splitext(data_path_name)[0]
                interpolated_path_name = data_path_name + '_' + name + '_interpolated.csv'
                #End of August's addition

                analysis = pd.DataFrame(index=rawOneLR.ID.unique())
                analysis['LR'] = name
                analysis['ID'] = analysis.index
                #August edited this line to include the opt values and the interpolated path name
                oneLR = restructure_tracking_data(rawOneLR, opt, interpolated_path_name)  # one video of one colony
                
                if opt['brood']:
                    oneLR = processBrood(f, oneLR, name, opt['broodExtension'], opt['brood'])
                    oneLR.to_csv('oneLR.csv')
                for test in funcs:
                    #try:
                        analysis[test[0]] = None
                        analysis[test[0]] = test[1](oneLR)
                    #except Exception as e:
                    #    print(test[0] + " cannot be run on " + v.replace(".mjpeg", opt['extension']))
                    #    analysis[test] = None
                    #    print(e)
                    #    continue
                fullAnalysis = pd.concat([fullAnalysis, analysis], axis=0)

            oneVid = pd.DataFrame(index=fullAnalysis.index)
            oneVid['pi_ID'] = workerID
            oneVid['bee_ID'] = oneVid.index
            oneVid['Date'] = Date
            oneVid['Time'] = Time
            oneVid = pd.concat([oneVid, fullAnalysis], axis=1)

            output = pd.concat([output, oneVid], ignore_index=True, axis=0)
            output.to_csv(path_or_buf=opt['outFile'])
            print('Done!')
    
        output.to_csv(path_or_buf=opt['outFile'])
    print("All done!")
    return 0


if __name__ == "__main__":
    """Makes sure the "main" function is called from the command line"""
    status = main()
    sys.exit(status)
