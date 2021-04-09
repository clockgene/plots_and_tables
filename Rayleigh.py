# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:40:21 2020

@author: Martin.Sladek
"""
import numpy  as np
import scipy as sp
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import PlotOptions as plo
import Bioluminescence as blu
import DecayingSinusoid as dsin
import CellularRecording as cr
import winsound
import glob
import matplotlib as mpl
import seaborn as sns
import math
import warnings

# for Lumi data set to False, for GridOverlay ImageJ data set to True
grid_overlay = True         

# How much plots of insidivual cells do you need? Set nth=1 for all, nth=10 for every 10th, ...
nth = 3

# input files from Lumi/Fiji rois/... need to be 2, id_signal and id_XY, from trackmate only 1 file.
# ID = LUMI for Lumicycle, FIJI for manual roi, SCNGRID for auto GridOverlay rois.
INPUT_FILES   = ['SCNGRID']

# if recording 1 frame/hour, set time_factor to 1, if 1 frame/0.25h, set to 1/4 (luminoskan), 1/6 for Lumicycle,...
time_factor = 1

# IN REAL HOURS, plots and analyze only data from this timepoint, settings for truncate_t variable - 
treatment = 0

# IN REAL HOURS or None (for whole dataset), plots and analyze only data to this timepoint, settings for end variable
end_h = 72
     
#
#
#                Code below this line should not be edited.
#

#inputs nms pre
PULL_FROM_IMAGEJ = False    # for Lumi data, use mock _XY.csv and _signal.csv in analysis folder
INPUT_DIR = 'data/'
INPUT_EXT = '.csv'

# for preanalyzed data from Lumi, use this and put signal and XY input files in analysis_output__ folder first
timestamp = '_'    

# supress annoying UserWarning: tight_layout: falling back to Agg renderer
def fxn():
    warnings.warn("tight_layout", UserWarning)

# list all the datasets
all_inputs=[]
for input_fi in INPUT_FILES:
    all_inputs.append(cr.generate_filenames_dict(INPUT_DIR, input_fi,
                                    PULL_FROM_IMAGEJ, input_ij_extension=INPUT_EXT))

#############################################################
#############################################################
####### FINAL PLOTS #########################################
#############################################################
#############################################################  

# to change CT to polar coordinates for polar plotting
# 1h = (2/24)*np.pi = (1/12)*np.pi,   circumference = 2*np.pi*radius
# use modulo to get remainder after integer division
def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x % 24)/12)*np.pi
    return r

# stackoverflow filter outliers - change m as needed (2 is default, 10 filters only most extreme)
def reject_outliers(data, m=10.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

# https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html
def grayscale_cmap(cmap):
    from matplotlib.colors import LinearSegmentedColormap
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

#cmap="viridis"
cmap="YlGnBu"
#cmap= grayscale_cmap(cmap)


#mydir = INPUT_DIR+f'analysis_output_{timestamp}/'
#mydir = f'{os.getcwd()}/{INPUT_DIR}analysis_output_{timestamp}/'
mydir = f'./{INPUT_DIR}analysis_output_{timestamp}/'

# LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])

### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)


#########################################################################
####### Calculate params #########################################
#########################################################################

# Use amplitude to filter out outliers or nans
#outlier_reindex = ~(np.isnan(reject_outliers(data[['Amplitude']])))['Amplitude']          # need series of bool values for indexing 
outlier_reindex = ~(np.isnan(data['Amplitude']))

data_filt = data[data.columns[:].tolist()][outlier_reindex]                                  # data w/o amp outliers

phaseseries = data_filt['Phase'].values.flatten()                                           # plot Phase
phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())                                     # plot R2 related number as width

# NAME
genes = data_filt['Unnamed: 0'].values.flatten().astype(int)                      # plot profile name as color
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))     # gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

# LENGTH (AMPLITUDE)
amp = data_filt['Amplitude'].values.flatten()                       # plot filtered Amplitude as length
#amp = 1                                                            # plot arbitrary number if Amp problematic

# POSITION (PHASE)
#phase = [polarphase(i) for i in phaseseries]                        # if phase in in hours (cosinor)
phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
#phase = [i for i in phaseseries]                                   # if phase is in radians already

# WIDTH (SD, SEM, R2, etc...)
#phase_sd = [polarphase(i) for i in phase_sdseries]                 # if using CI or SEM of phase, which is in hours
phase_sd = [i for i in phase_sdseries]                              # if using Rsq/R2, maybe adjust thickness 


###############################################################################################
####### Single Polar Histogram of frequency of phases #########################################
###############################################################################################

N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, N_bins))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle

axh = plt.subplot(111, projection='polar')                                                      #plot with polar projection
bars_h = axh.bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre

axh.set_yticklabels([])          # this deletes radial ticks
axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
axh.set_theta_direction(-1)      #reverse direction of theta increases
axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
axh.set_xlabel("Circadian phase (h)", fontsize=12)
#plt.title("Phase histogram", fontsize=14, fontstyle='italic')

# calculate vector sum of angles and plot "Rayleigh" vector
a_cos = map(lambda x: math.cos(x), phase)
a_sin = map(lambda x: math.sin(x), phase)
uv_x = sum(a_cos)/len(phase)
uv_y = sum(a_sin)/len(phase)
uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
uv_phase = np.angle(complex(uv_x, uv_y))

v_angle = uv_phase                  # maybe the same as np.mean(phase)?
v_length = uv_radius*max(phase_hist)  # because hist is not (0,1) but (0, N in largest bin), need to increase radius
axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(width=1, color='black')) #add arrow


### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}Histogram_Phase_Ray.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Histogram_Phase_Ray.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()