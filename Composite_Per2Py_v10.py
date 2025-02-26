"""
Created on Thu Mar 18 12:55:07 2021
@author: Martin.Sladek

Make composite figure from many individual heatmaps or histograms
v20250226 - Pie charts, PCA, Split violin KW_BoxenPlot_hue, Polar_scatterplot beta version. NO FILTERING or TraceHeatmaps or GIFS or TIFSEQUENCE for before_after_rhythm ! Exp-specific parameters
"""
# imports
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import glob, os, re
from tkinter import filedialog
from tkinter import *
from matplotlib import colors
import seaborn as sns
import scipy.stats as stats


# Choose type of experiment: decay, rhythm (without treatment), before_after_rhythm (i.e. with treatment)
experiment = 'before_after_rhythm'

# Choose to plot heatmap of K, Halflife, Trend, Phase, Amplitude or Period, or Phase_Histogram, Pie, Trace , Parameters, TraceHeatmaps, GIFS, TIFSEQUENCE, not working - Phase_ScatterPlot,
# Parameters do not need exp. specified and work on any n of folders, unless combineGroup = True. 
# Copies animated gif files if graphtype is set to GIFS. Pie chart shows number of rhythmic and nonrhythmic rois
graphtype = 'Parameters'

# Need arrow for treatment? Then add treatment time in frames (not hours, but usually 1 frame/h for 6 explants).
treatment_time = 77

# True - Plot all individual roi luminescence traces (TAKES a LONG TIME). False - Plot just median trace. 'select_parameter' - Plot rois filtered by parameter - only works for experiment = 'rhythm'
Plot_All_Traces = True
# Plot_All_Traces = 'select_rsq'
# Plot_All_Traces = 'select_amp'
# Plot_All_Traces = 'select_trend'
# Plot_All_Traces = 'select_decay'

# set arbitrary thresholds for filtering by R, Amp or other parameters - only works for experiment = 'rhythm'
rsq_threshold = 0.8
amp_threshold = 10    # different number for heatmap and for trace, but why? heatmap is norm after
trend_threshold = 17
decay_threshold = 0.000001

# cut first x hours before and leave y hours total (cutoff = 12, cutoff2 = None --- default)
cutoff = 0
cutoff2 = None

# set number of explants or how many individual L and R SCNs were analyzed.
# There will be Nr rows and 2 columns - before, after (subfolders of each explant) for experiment = 'before_after_rhythm'
Nr = 18

#For experiment = 'rhythm' or 'decay', need also Nc and Nw variables to control cols and rows
# no. of columns
Nc = 3
# no. of rows
Nw = 6

# Adjust spacing between before left-right and up-down heatmap and histogram plots, for 6 SCN try -0.9, for less or for big CP -0.6
wspace= -0.9   # -0.5
hspace= 0.5    # 0.5

# Same size heatmaps for all explants (True) or or adjust size to fit whole heatmap (False)?
# For explants with widely different size, sometimes True results in only partially plotted heatmap.
sharex = False
sharey = False

# Adjust how close and how big labels are in Phase_Histogram and Pie charts, depends on number of plots and wspace, for 6 rows try -13 and 4
pad = -8        # -13
fontsize = 1     # 4

# For Parameters csv and violin plots - set name and nameend variables, depends on length of folder names
name = -6               # -4 for nontreated CP, -5 for nontreated SCN, for treated CP -8, for treated SCN -12
nameend = None          # None, or try -4

# use iqr_value unless this is set to True - for Parameters violin plots with activated threshold, set to True
disable_parameters_iqr = False

# Amplitude specific settings 
# Adjust outlier filtering, try iqr_value 1, 2.22 or 8.88 or more (bigger iqr keeps bigger outliers) 
iqr_value = 8.88
# default False , but if iqr filter fails (bad rhythm) may use Log normalize to make usable Amp heatmaps
lognorm = False
# default False , but if iqr fails, may set integer index of 1 image for which normalization will be disabled
nonorm = False

# Make violinn plot for statistics combining different explants, True - experiment specific, need to adjust and finetune the script
combineGroup = False

# For TIFSEQUENCE - how many pictures u want? Set Nx, total pictures will be Nx^2, Nx = 6 looks good
Nx = 6
# For TIFSEQUENCE - choose from which image (which hour if 1 image/hour) to start the tablo, default Ns = 1
Ns = 120

# DO NOT EDIT BELOW

# stackoverflow filter outliers - change m as needed (2 is default, 10 filters only most extreme)
def reject_outliers(data, column=graphtype, m=10):
    data2 = data.loc[(data[column] == data[column])][column]
    d = np.abs(data[column] - np.median(data2))
    mdev = np.median(d[d == d])
    s = d / (mdev if mdev else 1.)
    data.loc[(data[column] == data[column]) & (s > m), column] = np.nan
    return data

# use modulo to get remainder after integer division
def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x % 24)/12)*np.pi
    return r

# polarhist(axh[i, j], mydirlist[counter][-9:], pad)
def polarhist(axh, title, pval_Rt_list, v_length_list, pad=-12):
    #axh = plt.subplot(projection='polar')                                                 #plot with polar projection, but must set for fig subplots
    axh.bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre
    
    axh.set_yticklabels([])          # this deletes radial ticks
    axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
    axh.set_theta_direction(-1)      #reverse direction of theta increases
    axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontsize=fontsize)  #set theta grids and labels, **kwargs for text properties
    #axh.set_thetagrids([])  # turns off labels in case of problems
    axh.yaxis.grid(False)   # turns off circles
    axh.xaxis.grid(False)  # turns off radial grids
    axh.tick_params(pad=pad)   # moves labels closer or further away from subplots, may need to adjust depending on number of subplots
    axh.set_xlabel(f'{title}', fontsize=fontsize, labelpad=1)   # place specific title and use labelpad to adjust proximity to subplot
    #plt.title("Phase histogram", fontsize=14, fontstyle='italic')
    
    # calculate vector sum of angles and plot "Rayleigh" vector
    a_cos = map(lambda x: math.cos(x), phase)
    a_sin = map(lambda x: math.sin(x), phase)
    uv_x = sum(a_cos)/len(phase)
    uv_y = sum(a_sin)/len(phase)
    uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
    uv_phase = np.angle(complex(uv_x, uv_y))
    # Alternative from http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf
    #v_angle = math.atan((uv_y/uv_radius)/(uv_x/uv_radius))    
    v_angle = uv_phase     # they are the same 
    v_length = uv_radius*max(phase_hist)  # because hist is not (0,1) but (0, N in largest bin), need to increase radius
    
    # Rayleigh test for non-uniformity of circular data https://github.com/circstat/pycircstat/blob/master/pycircstat/tests.py
    r_Rt = uv_radius
    n_Rt = len(phaseseries)
    R_Rt = n_Rt * r_Rt                              # compute Rayleigh's R (equ. 27.1)
    # z_Rt = R_Rt ** 2 / n_Rt                         # compute Rayleigh's z (equ. 27.2)
    pval_Rt = np.exp(np.sqrt(1 + 4 * n_Rt + 4 * (n_Rt ** 2 - R_Rt ** 2)) - (1 + 2 * n_Rt))     # compute p value using approxation in Zar, p. 617
    pval_Rt_list.append(pval_Rt)
    v_length_list.append(v_length)
    #add arrow and test rounded pvalue, needs update to adjust font and arrow size depending on number of plots
    # axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(width=1, color='black'))
    # axh.annotate(f'p={np.format_float_scientific(pval_Rt, precision=4)} {v_length}', xy=(v_angle, v_length))
    
    
    axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(linewidth=0.2, arrowstyle="-|>", color='black', mutation_scale=2)) #add arrow    
    
    return axh, pval_Rt_list, v_length_list

# old version that works with before_after_rhythm
def polarhist_old(axh, title, pad=-12):
    #axh = plt.subplot(projection='polar')                                                 #plot with polar projection, but must set for fig subplots
    axh.bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre
    
    axh.set_yticklabels([])          # this deletes radial ticks
    axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
    axh.set_theta_direction(-1)      #reverse direction of theta increases
    axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontsize=fontsize)  #set theta grids and labels, **kwargs for text properties
    #axh.set_thetagrids([])  # turns off labels in case of problems
    axh.yaxis.grid(False)   # turns off circles
    axh.xaxis.grid(False)  # turns off radial grids
    axh.tick_params(pad=pad)   # moves labels closer or further away from subplots, may need to adjust depending on number of subplots
    axh.set_xlabel(f'{title}', fontsize=fontsize, labelpad=1)   # place specific title and use labelpad to adjust proximity to subplot
    #plt.title("Phase histogram", fontsize=14, fontstyle='italic')
    
    # calculate vector sum of angles and plot "Rayleigh" vector
    a_cos = map(lambda x: math.cos(x), phase)
    a_sin = map(lambda x: math.sin(x), phase)
    uv_x = sum(a_cos)/len(phase)
    uv_y = sum(a_sin)/len(phase)
    uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
    uv_phase = np.angle(complex(uv_x, uv_y))
    # Alternative from http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf
    #v_angle = math.atan((uv_y/uv_radius)/(uv_x/uv_radius))    
    v_angle = uv_phase     # they are the same 
    v_length = uv_radius*max(phase_hist)  # because hist is not (0,1) but (0, N in largest bin), need to increase radius
    axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(linewidth=0.2, arrowstyle="-|>", color='black', mutation_scale=2)) #add arrow    
    
    return axh

# FOR POLAR SCATTER PLOT, create r or y or amp, need testing and manual adjustment
def create_second_list(original_list):
    second_list = []
    count_dict = {}

    for num in original_list:
        if num not in count_dict:
            count_dict[num] = 1
            second_list.append(1)
        else:
            count_dict[num] += 0.05
            second_list.append(1 + count_dict[num] - 1)

    return second_list

# FOR POLAR SCATTER PLOT, create r or y or amp, need testing and manual adjustment, NOT WORKING
def polarscatterplot(axh, title, phase_rounded, r_from_amp, pad=-12):
    axh.scatter(phase_rounded, r_from_amp, s=0.5**2, marker=".", linewidth=0, alpha=0.8) # color=colorcode,    
    axh.set_yticklabels([])          # this deletes radial ticks
    axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
    axh.set_theta_direction(-1)      #reverse direction of theta increases
    axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontsize=fontsize)  #set theta grids and labels, **kwargs for text properties
    #axh.set_thetagrids([])  # turns off labels in case of problems
    axh.yaxis.grid(False)   # turns off circles
    axh.xaxis.grid(False)  # turns off radial grids
    axh.tick_params(pad=pad)   # moves labels closer or further away from subplots, may need to adjust depending on number of subplots
    axh.set_xlabel(f'{title}', fontsize=fontsize, labelpad=1)   # place specific title and use labelpad to adjust proximity to subplot
    #plt.title("Phase histogram", fontsize=14, fontstyle='italic')
    
    # calculate vector sum of angles and plot "Rayleigh" vector
    a_cos = map(lambda x: math.cos(x), phase)
    a_sin = map(lambda x: math.sin(x), phase)
    uv_x = sum(a_cos)/len(phase)
    uv_y = sum(a_sin)/len(phase)
    uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
    uv_phase = np.angle(complex(uv_x, uv_y))
    # Alternative from http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf
    #v_angle = math.atan((uv_y/uv_radius)/(uv_x/uv_radius))    
    v_angle = uv_phase     # they are the same 
    v_length = uv_radius*max(r_from_amp)  # because hist is not (0,1) but (0, N in largest bin), need to increase radius
    axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(linewidth=0.2, arrowstyle="-|>", color='black', mutation_scale=2)) #add arrow    
    
    return axh

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

# Make images respond to changes in the norm of other images (e.g. via the
# "edit axis, curves and images parameters" GUI on Qt), but be careful not to
# recurse infinitely!
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())

# Violin plot from wide-format dataframe, for experiment specific figures
def violin_stat(data, title, ticks, remove_outliers=True, iqr_value = 2.22, test = 'ttest'):
    
    # FILTER outliers by iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    if remove_outliers == True:
        for col in data.columns.values:
            iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
            lim = np.abs((data[col] - data[col].median()) / iqr) < iqr_value
            data.loc[:, col] = data[col].where(lim, np.nan)
    
    title = title  
    
    # creating a dictionary with one specific color per group:
    my_pal = {data.columns[0]: "slateblue", data.columns[1]: "tomato"}
    
    fig, ax = plt.subplots(1, figsize=(2,4))          
    ax = sns.violinplot(data=data, palette=my_pal)
    # plt.title(title)
    ax.axes.xaxis.set_ticklabels(ticks)
    ax.set_xlabel('') 
    ax.set_ylabel(f'{title}') 
    ax.spines['top'].set_visible(False) # to turn off individual borders 
    ax.spines['right'].set_visible(False)
    # plt.xticks(rotation=90)
    
    ###### Calculate t test p values between hue_dat for separate categories in col_dat ######
    pvalues = []
    datax1 = data[data.columns[0]].dropna(how='any')
    datax2 = data[data.columns[1]].dropna(how='any')
    
    if test == 'ttest':
        t, p = stats.ttest_ind(datax1.values, datax2.values)
        pvalues = pvalues + [p]
        plt.annotate('t test \nP = ' + str(round(p, 10)), xy=(1, 1), xycoords='axes fraction', fontsize=10, 
                     xytext=(-5, 5), textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    else:
        t, p = stats.mannwhitneyu(datax1.values, datax2.values)    
        pvalues = pvalues + [p]
        plt.annotate('Mann-Whitney \nP = ' + str(round(p, 10)), xy=(1, 1), xycoords='axes fraction', fontsize=10, 
                     xytext=(-5, 5), textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    
    
    plt.savefig(f'{path}Violin_{title}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'{path}Violin_{title}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()


# UPDATE for 3 or more columns , for experiment specific figures       
def KW_BoxenPlot1(data, title, path, ticks, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Amplitude'):
    col_var = data.columns.values
    col_order = []
    for a in col_var:
        col_order.append(a)
        
    # FILTER outliers by iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    if remove_outliers == True:
        for col in data.columns.values:
            iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
            lim = np.abs((data[col] - data[col].median()) / iqr) < iqr_value
            data.loc[:, col] = data[col].where(lim, np.nan)
    
    title = title  
    
    # creating a dictionary with one specific color per group:
    my_pal = {data.columns[0]: "slateblue", data.columns[1]: "tomato", data.columns[2]: "orange"}

    f, ax = plt.subplots(figsize=(3, 6))  # figsize not working?
    
    if kind == 'boxstrip':    
        sns.boxplot(data=data, order=col_order, palette=my_pal) # , order=col_order
        sns.stripplot(data=data, order=col_order, palette=my_pal)
        ax.axes.xaxis.set_ticklabels(ticks)
        ax.set_xlabel('') 
        ax.set_ylabel(f'{title}') 
        ax.spines['top'].set_visible(False) # to turn off individual borders 
        ax.spines['right'].set_visible(False)
    else:
        sns.catplot(kind=kind, data=data, order=col_order, aspect=0.5, palette=my_pal) #g =  ..., order=col_order, 
        ax.axes.xaxis.set_ticklabels(ticks)
        ax.set_xlabel('') 
        ax.set_ylabel(f'{title}') 
        ax.spines['top'].set_visible(False) # to turn off individual borders 
        ax.spines['right'].set_visible(False)
        ax.set(ylim=ylim)

    # ##### Kruskal-Wallis H-test with auto-assigned data ######
    # alist = []
    # for i in col_order:
    #     alist.append(data[i].dropna(how='any'))
    # tano, pano = stats.kruskal(*alist) # asterisk is for *args - common idiom to allow arbitrary number of arguments to functions    
    # if pano < 0.0000000001:
    #     #plt.text(x_coord, y_coord, 'Kruskal-Wallis p < 1e-10' , fontsize=10)
    #     ax.annotate('Kruskal-Wallis p < 1e-10', xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
    #                 textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    # else:
    #     #plt.text(x_coord, y_coord, 'Kruskal-Wallis p = ' + str(round(pano, 8)), fontsize=10)
    #     ax.annotate('Kruskal-Wallis p = ' + str(round(pano, 8)), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
    #                 textcoords='offset points', horizontalalignment='right', verticalalignment='top')        

    plt.savefig(f'{path}KW_BoxenPlot_{name}_{kind}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'{path}KW_BoxenPlot_{name}_{kind}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()

    # ##### Post-hoc tests - Dunn's ###############
    # xx = data[x][(data[x] == data[x]) & (data[y] == data[y])].dropna(how='any')
    # yy = data[y][data[x] == data[x]].dropna(how='any')
    # df_stat = pd.DataFrame(xx)          #create new dataframe to avoid NaN problems
    # df_stat[y] = yy                     #add column with analysed data to new dataframe

    # #posthoc = sp.posthoc_dunn(df_stat.reset_index(drop=True), val_col=y, group_col=x)
    # pc = sp.posthoc_dunn(df_stat, val_col=y, group_col=x)
    # heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.79, 0.35, 0.035, 0.3]}  #
    # sp.sign_plot(pc, **heatmap_args)

    # plt.savefig(f'{mydir}' + '\\' + f'KW_BoxenPlot1_posthoc.png', format = 'png')
    # plt.savefig(f'{mydir}' + '\\' + f'KW_BoxenPlot1_posthoc.svg', format = 'svg')    
    # plt.clf()
    # plt.close()   


# UPDATE for 3 or more columns , for experiment specific figures       
def KW_BoxenPlot4(data, title, path, ticks, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Period'):
    col_var = data.columns.values
    col_order = []
    for a in col_var:
        col_order.append(a)
        
    # FILTER outliers by iqr filter: within 2.22 IQR (equiv. to z-score < 3)
    if remove_outliers == True:
        for col in data.columns.values:
            iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
            lim = np.abs((data[col] - data[col].median()) / iqr) < iqr_value
            data.loc[:, col] = data[col].where(lim, np.nan)
    
    title = title  
    
    # creating a dictionary with one specific color per group:
    my_pal = {data.columns[0]: "slateblue", data.columns[1]: "tomato", data.columns[2]: "orange", data.columns[3]: "green"}

    f, ax = plt.subplots(figsize=(3, 6))  # figsize not working?
    
    if kind == 'boxstrip':    
        sns.boxplot(data=data, order=col_order, palette=my_pal) # , order=col_order
        sns.stripplot(data=data, order=col_order, palette=my_pal)
        ax.axes.xaxis.set_ticklabels(ticks)
        ax.set_xlabel('') 
        ax.set_ylabel(f'{title}') 
        ax.spines['top'].set_visible(False) # to turn off individual borders 
        ax.spines['right'].set_visible(False)
    else:
        sns.catplot(kind=kind, data=data, order=col_order, aspect=0.5, palette=my_pal) #g =  ..., order=col_order, 
        ax.axes.xaxis.set_ticklabels(ticks)
        ax.set_xlabel('') 
        ax.set_ylabel(f'{title}') 
        ax.spines['top'].set_visible(False) # to turn off individual borders 
        ax.spines['right'].set_visible(False)
        ax.set(ylim=ylim)

    # ##### Kruskal-Wallis H-test with auto-assigned data ######
    # alist = []
    # for i in col_order:
    #     alist.append(data[i].dropna(how='any'))
    # tano, pano = stats.kruskal(*alist) # asterisk is for *args - common idiom to allow arbitrary number of arguments to functions    
    # if pano < 0.0000000001:
    #     #plt.text(x_coord, y_coord, 'Kruskal-Wallis p < 1e-10' , fontsize=10)
    #     ax.annotate('Kruskal-Wallis p < 1e-10', xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
    #                 textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    # else:
    #     #plt.text(x_coord, y_coord, 'Kruskal-Wallis p = ' + str(round(pano, 8)), fontsize=10)
    #     ax.annotate('Kruskal-Wallis p = ' + str(round(pano, 8)), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
    #                 textcoords='offset points', horizontalalignment='right', verticalalignment='top')        

    plt.savefig(f'{path}KW_BoxenPlot_{name}_{kind}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'{path}KW_BoxenPlot_{name}_{kind}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()

    # ##### Post-hoc tests - Dunn's ###############
    # xx = data[x][(data[x] == data[x]) & (data[y] == data[y])].dropna(how='any')
    # yy = data[y][data[x] == data[x]].dropna(how='any')
    # df_stat = pd.DataFrame(xx)          #create new dataframe to avoid NaN problems
    # df_stat[y] = yy                     #add column with analysed data to new dataframe

    # #posthoc = sp.posthoc_dunn(df_stat.reset_index(drop=True), val_col=y, group_col=x)
    # pc = sp.posthoc_dunn(df_stat, val_col=y, group_col=x)
    # heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.79, 0.35, 0.035, 0.3]}  #
    # sp.sign_plot(pc, **heatmap_args)

    # plt.savefig(f'{mydir}' + '\\' + f'KW_BoxenPlot1_posthoc.png', format = 'png')
    # plt.savefig(f'{mydir}' + '\\' + f'KW_BoxenPlot1_posthoc.svg', format = 'svg')    
    # plt.clf()
    # plt.close()   


def KW_BoxenPlot_hue(data, title, path, x, y, hue, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Period'):        
    
    title = title   
    f, ax = plt.subplots(figsize=(3, 6))           

    sns.catplot(kind=kind, data=data, x=x, y=y, hue=hue, split=True, aspect=0.5)   # order=col_order, 
    ax.set_xlabel('') 
    ax.set_ylabel(f'{title}') 
    ax.spines['top'].set_visible(False) # to turn off individual borders 
    ax.spines['right'].set_visible(False)
    ax.set(ylim=ylim)      

    plt.savefig(f'{path}KW_BoxenPlot_{name}_{kind}_{hue}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'{path}KW_BoxenPlot_{name}_{kind}_{hue}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
  
    # mpl.use('Qt5Agg')      
    sns.catplot(kind='violin', data=melted, x='Regime', y='Value', hue='Sex', split=True)   
    plt.savefig(f'{path}KW_BoxenPlot_SEX test.png', format = 'png', bbox_inches = 'tight')   



# custom colors for nice circular hues
circular_colors = np.array([[0.91510904, 0.55114749, 0.67037311],
   [0.91696411, 0.55081563, 0.66264366],
   [0.91870995, 0.55055664, 0.65485881],
   [0.92034498, 0.55037149, 0.64702356],
   [0.92186763, 0.55026107, 0.63914306],
   [0.92327636, 0.55022625, 0.63122259],
   [0.9245696 , 0.55026781, 0.62326754],
   [0.92574582, 0.5503865 , 0.6152834 ],
   [0.92680349, 0.55058299, 0.6072758 ],
   [0.92774112, 0.55085789, 0.59925045],
   [0.9285572 , 0.55121174, 0.59121319],
   [0.92925027, 0.551645  , 0.58316992],
   [0.92981889, 0.55215808, 0.57512667],
   [0.93026165, 0.55275127, 0.56708953],
   [0.93057716, 0.5534248 , 0.55906469],
   [0.93076407, 0.55417883, 0.55105838],
   [0.93082107, 0.55501339, 0.54307696],
   [0.93074689, 0.55592845, 0.53512681],
   [0.9305403 , 0.55692387, 0.52721438],
   [0.93020012, 0.55799943, 0.51934621],
   [0.92972523, 0.55915477, 0.51152885],
   [0.92911454, 0.56038948, 0.50376893],
   [0.92836703, 0.56170301, 0.49607312],
   [0.92748175, 0.56309471, 0.48844813],
   [0.9264578 , 0.56456383, 0.48090073],
   [0.92529434, 0.56610951, 0.47343769],
   [0.92399062, 0.56773078, 0.46606586],
   [0.92254595, 0.56942656, 0.45879209],
   [0.92095971, 0.57119566, 0.4516233 ],
   [0.91923137, 0.5730368 , 0.44456642],
   [0.91736048, 0.57494856, 0.4376284 ],
   [0.91534665, 0.57692945, 0.43081625],
   [0.91318962, 0.57897785, 0.42413698],
   [0.91088917, 0.58109205, 0.41759765],
   [0.90844521, 0.58327024, 0.41120533],
   [0.90585771, 0.58551053, 0.40496711],
   [0.90312676, 0.5878109 , 0.3988901 ],
   [0.90025252, 0.59016928, 0.39298143],
   [0.89723527, 0.5925835 , 0.38724821],
   [0.89407538, 0.59505131, 0.38169756],
   [0.89077331, 0.59757038, 0.37633658],
   [0.88732963, 0.60013832, 0.37117234],
   [0.88374501, 0.60275266, 0.36621186],
   [0.88002022, 0.6054109 , 0.36146209],
   [0.87615612, 0.60811044, 0.35692989],
   [0.87215369, 0.61084868, 0.352622  ],
   [0.86801401, 0.61362295, 0.34854502],
   [0.86373824, 0.61643054, 0.34470535],
   [0.85932766, 0.61926872, 0.3411092 ],
   [0.85478365, 0.62213474, 0.3377625 ],
   [0.85010767, 0.6250258 , 0.33467091],
   [0.84530131, 0.62793914, 0.3318397 ],
   [0.84036623, 0.63087193, 0.32927381],
   [0.8353042 , 0.63382139, 0.32697771],
   [0.83011708, 0.63678472, 0.32495541],
   [0.82480682, 0.63975913, 0.32321038],
   [0.81937548, 0.64274185, 0.32174556],
   [0.81382519, 0.64573011, 0.32056327],
   [0.80815818, 0.6487212 , 0.31966522],
   [0.80237677, 0.65171241, 0.31905244],
   [0.79648336, 0.65470106, 0.31872531],
   [0.79048044, 0.65768455, 0.31868352],
   [0.78437059, 0.66066026, 0.31892606],
   [0.77815645, 0.66362567, 0.31945124],
   [0.77184076, 0.66657827, 0.32025669],
   [0.76542634, 0.66951562, 0.3213394 ],
   [0.75891609, 0.67243534, 0.32269572],
   [0.75231298, 0.67533509, 0.32432138],
   [0.74562004, 0.6782126 , 0.32621159],
   [0.73884042, 0.68106567, 0.32836102],
   [0.73197731, 0.68389214, 0.33076388],
   [0.72503398, 0.68668995, 0.33341395],
   [0.7180138 , 0.68945708, 0.33630465],
   [0.71092018, 0.69219158, 0.33942908],
   [0.70375663, 0.69489159, 0.34278007],
   [0.69652673, 0.69755529, 0.34635023],
   [0.68923414, 0.70018097, 0.35013201],
   [0.6818826 , 0.70276695, 0.35411772],
   [0.67447591, 0.70531165, 0.3582996 ],
   [0.667018  , 0.70781354, 0.36266984],
   [0.65951284, 0.71027119, 0.36722061],
   [0.65196451, 0.71268322, 0.37194411],
   [0.64437719, 0.71504832, 0.37683259],
   [0.63675512, 0.71736525, 0.38187838],
   [0.62910269, 0.71963286, 0.38707389],
   [0.62142435, 0.72185004, 0.39241165],
   [0.61372469, 0.72401576, 0.39788432],
   [0.60600841, 0.72612907, 0.40348469],
   [0.59828032, 0.72818906, 0.40920573],
   [0.59054536, 0.73019489, 0.41504052],
   [0.58280863, 0.73214581, 0.42098233],
   [0.57507535, 0.7340411 , 0.42702461],
   [0.5673509 , 0.7358801 , 0.43316094],
   [0.55964082, 0.73766224, 0.43938511],
   [0.55195081, 0.73938697, 0.44569104],
   [0.54428677, 0.74105381, 0.45207286],
   [0.53665478, 0.74266235, 0.45852483],
   [0.52906111, 0.74421221, 0.4650414 ],
   [0.52151225, 0.74570306, 0.47161718],
   [0.5140149 , 0.74713464, 0.47824691],
   [0.506576  , 0.74850672, 0.48492552],
   [0.49920271, 0.74981912, 0.49164808],
   [0.49190247, 0.75107171, 0.4984098 ],
   [0.48468293, 0.75226438, 0.50520604],
   [0.47755205, 0.7533971 , 0.51203229],
   [0.47051802, 0.75446984, 0.5188842 ],
   [0.46358932, 0.75548263, 0.52575752],
   [0.45677469, 0.75643553, 0.53264815],
   [0.45008317, 0.75732863, 0.5395521 ],
   [0.44352403, 0.75816207, 0.54646551],
   [0.43710682, 0.758936  , 0.55338462],
   [0.43084133, 0.7596506 , 0.56030581],
   [0.42473758, 0.76030611, 0.56722555],
   [0.41880579, 0.76090275, 0.5741404 ],
   [0.41305637, 0.76144081, 0.58104704],
   [0.40749984, 0.76192057, 0.58794226],
   [0.40214685, 0.76234235, 0.59482292],
   [0.39700806, 0.7627065 , 0.60168598],
   [0.39209414, 0.76301337, 0.6085285 ],
   [0.38741566, 0.76326334, 0.6153476 ],
   [0.38298304, 0.76345681, 0.62214052],
   [0.37880647, 0.7635942 , 0.62890454],
   [0.37489579, 0.76367593, 0.63563704],
   [0.37126045, 0.76370246, 0.64233547],
   [0.36790936, 0.76367425, 0.64899736],
   [0.36485083, 0.76359176, 0.6556203 ],
   [0.36209245, 0.76345549, 0.66220193],
   [0.359641  , 0.76326594, 0.66873999],
   [0.35750235, 0.76302361, 0.67523226],
   [0.35568141, 0.76272903, 0.68167659],
   [0.35418202, 0.76238272, 0.68807086],
   [0.3530069 , 0.76198523, 0.69441305],
   [0.35215761, 0.7615371 , 0.70070115],
   [0.35163454, 0.76103888, 0.70693324],
   [0.35143685, 0.76049114, 0.71310742],
   [0.35156253, 0.75989444, 0.71922184],
   [0.35200839, 0.75924936, 0.72527472],
   [0.3527701 , 0.75855647, 0.73126429],
   [0.3538423 , 0.75781637, 0.73718884],
   [0.3552186 , 0.75702964, 0.7430467 ],
   [0.35689171, 0.75619688, 0.74883624],
   [0.35885353, 0.75531868, 0.75455584],
   [0.36109522, 0.75439565, 0.76020396],
   [0.36360734, 0.75342839, 0.76577905],
   [0.36637995, 0.75241752, 0.77127961],
   [0.3694027 , 0.75136364, 0.77670417],
   [0.37266493, 0.75026738, 0.7820513 ],
   [0.37615579, 0.74912934, 0.78731957],
   [0.37986429, 0.74795017, 0.79250759],
   [0.38377944, 0.74673047, 0.797614  ],
   [0.38789026, 0.74547088, 0.80263746],
   [0.3921859 , 0.74417203, 0.80757663],
   [0.39665568, 0.74283455, 0.81243022],
   [0.40128912, 0.74145908, 0.81719695],
   [0.406076  , 0.74004626, 0.82187554],
   [0.41100641, 0.73859673, 0.82646476],
   [0.41607073, 0.73711114, 0.83096336],
   [0.4212597 , 0.73559013, 0.83537014],
   [0.42656439, 0.73403435, 0.83968388],
   [0.43197625, 0.73244447, 0.8439034 ],
   [0.43748708, 0.73082114, 0.84802751],
   [0.44308905, 0.72916502, 0.85205505],
   [0.44877471, 0.72747678, 0.85598486],
   [0.45453694, 0.72575709, 0.85981579],
   [0.46036897, 0.72400662, 0.8635467 ],
   [0.4662644 , 0.72222606, 0.86717646],
   [0.47221713, 0.72041608, 0.87070395],
   [0.47822138, 0.71857738, 0.87412804],
   [0.4842717 , 0.71671065, 0.87744763],
   [0.4903629 , 0.71481659, 0.88066162],
   [0.49649009, 0.71289591, 0.8837689 ],
   [0.50264864, 0.71094931, 0.88676838],
   [0.50883417, 0.70897752, 0.88965898],
   [0.51504253, 0.70698127, 0.89243961],
   [0.52126981, 0.70496128, 0.8951092 ],
   [0.52751231, 0.70291829, 0.89766666],
   [0.53376652, 0.70085306, 0.90011093],
   [0.54002912, 0.69876633, 0.90244095],
   [0.54629699, 0.69665888, 0.90465565],
   [0.55256715, 0.69453147, 0.90675397],
   [0.55883679, 0.69238489, 0.90873487],
   [0.56510323, 0.69021993, 0.9105973 ],
   [0.57136396, 0.68803739, 0.91234022],
   [0.57761655, 0.68583808, 0.91396258],
   [0.58385872, 0.68362282, 0.91546336],
   [0.59008831, 0.68139246, 0.91684154],
   [0.59630323, 0.67914782, 0.9180961 ],
   [0.60250152, 0.67688977, 0.91922603],
   [0.60868128, 0.67461918, 0.92023033],
   [0.61484071, 0.67233692, 0.921108  ],
   [0.62097809, 0.67004388, 0.92185807],
   [0.62709176, 0.66774097, 0.92247957],
   [0.63318012, 0.66542911, 0.92297153],
   [0.63924166, 0.66310923, 0.92333301],
   [0.64527488, 0.66078227, 0.92356308],
   [0.65127837, 0.65844919, 0.92366082],
   [0.65725076, 0.65611096, 0.92362532],
   [0.66319071, 0.65376857, 0.92345572],
   [0.66909691, 0.65142302, 0.92315115],
   [0.67496813, 0.64907533, 0.92271076],
   [0.68080311, 0.64672651, 0.92213374],
   [0.68660068, 0.64437763, 0.92141929],
   [0.69235965, 0.64202973, 0.92056665],
   [0.69807888, 0.6396839 , 0.91957507],
   [0.70375724, 0.63734122, 0.91844386],
   [0.70939361, 0.63500279, 0.91717232],
   [0.7149869 , 0.63266974, 0.91575983],
   [0.72053602, 0.63034321, 0.91420578],
   [0.72603991, 0.62802433, 0.9125096 ],
   [0.7314975 , 0.62571429, 0.91067077],
   [0.73690773, 0.62341425, 0.9086888 ],
   [0.74226956, 0.62112542, 0.90656328],
   [0.74758193, 0.61884899, 0.90429382],
   [0.75284381, 0.6165862 , 0.90188009],
   [0.75805413, 0.61433829, 0.89932181],
   [0.76321187, 0.6121065 , 0.89661877],
   [0.76831596, 0.6098921 , 0.89377082],
   [0.77336536, 0.60769637, 0.89077786],
   [0.77835901, 0.6055206 , 0.88763988],
   [0.78329583, 0.6033661 , 0.88435693],
   [0.78817477, 0.60123418, 0.88092913],
   [0.79299473, 0.59912616, 0.87735668],
   [0.79775462, 0.59704339, 0.87363986],
   [0.80245335, 0.59498722, 0.86977904],
   [0.8070898 , 0.592959  , 0.86577468],
   [0.81166284, 0.5909601 , 0.86162732],
   [0.81617134, 0.5889919 , 0.8573376 ],
   [0.82061414, 0.58705579, 0.85290625],
   [0.82499007, 0.58515315, 0.84833413],
   [0.82929796, 0.58328538, 0.84362217],
   [0.83353661, 0.58145389, 0.83877142],
   [0.8377048 , 0.57966009, 0.83378306],
   [0.8418013 , 0.57790538, 0.82865836],
   [0.84582486, 0.57619119, 0.82339871],
   [0.84977422, 0.57451892, 0.81800565],
   [0.85364809, 0.57289   , 0.8124808 ],
   [0.85744519, 0.57130585, 0.80682595],
   [0.86116418, 0.56976788, 0.80104298],
   [0.86480373, 0.56827749, 0.79513394],
   [0.86836249, 0.56683612, 0.789101  ],
   [0.87183909, 0.56544515, 0.78294645],
   [0.87523214, 0.56410599, 0.77667274],
   [0.87854024, 0.56282002, 0.77028247],
   [0.88176195, 0.56158863, 0.76377835],
   [0.88489584, 0.56041319, 0.75716326],
   [0.88794045, 0.55929505, 0.75044023],
   [0.89089432, 0.55823556, 0.74361241],
   [0.89375596, 0.55723605, 0.73668312],
   [0.89652387, 0.55629781, 0.72965583],
   [0.89919653, 0.55542215, 0.72253414],
   [0.90177242, 0.55461033, 0.71532181],
   [0.90425   , 0.55386358, 0.70802274],
   [0.90662774, 0.55318313, 0.70064098],
   [0.90890408, 0.55257016, 0.69318073],
   [0.91107745, 0.55202582, 0.68564633],
   [0.91314629, 0.55155124, 0.67804225]])

if graphtype == 'Phase' or graphtype == 'Phase_Histogram':
    cmap = mpl.colors.ListedColormap(circular_colors)
if graphtype == 'Amplitude':
    cmap="YlGnBu"
if graphtype == 'Period':
    cmap="plasma"
if graphtype == 'Halflife':    
    cmap='inferno_r'
if graphtype == 'K':    
    cmap='coolwarm'
if graphtype == 'Trend':    
    cmap=grayscale_cmap('inferno')
if graphtype == 'TIFSEQUENCE':    
    # cmap=grayscale_cmap('inferno') 
    cmap='inferno'
#continous color maps
#cmap="viridis"
#cmap="YlGnBu"
#cmap= grayscale_cmap(cmap)
#other circular color maps
#cmap = mpl.colors.ListedColormap(sns.hls_palette(256))
#cmap = mpl.colors.ListedColormap(sns.husl_palette(256, .33, .85, .6))
#cmap = mpl.colors.ListedColormap(sns.husl_palette(256))

# for larger figures, need to make the lines thinner
mpl.rcParams['axes.linewidth'] = 0.1

### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)

##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button



####### FINAL PLOTS #########################################

# Specify FOLDER
root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = Button(text="Browse to folder", command=browse_button)
buttonBrowse.grid()
mainloop()
path = os.getcwd() + '\\'

depth = path.count(os.sep)
paths = [path]

if experiment == 'rhythm':
    treat = 0
    Ncc = 1
else:
    treat = 1
    Ncc = 2

mydirlist = []
for root, dirs, files in os.walk(path, topdown=False):
    for files in dirs:        
        folder = os.path.join(root, files)
        #mydir = os.path.abspath(directories)
        #if folder.count(os.sep) == depth + 1:
        if folder.count(os.sep) == depth + treat:
            mydirlist.append(folder)
   
if experiment == 'before_after_rhythm':
    
    if graphtype == 'Trace':   # create plots of Raw traces and cosines befor and after with arrow pointing to treatment time
        
        import matplotlib.patches as mpatches     
        
        fig, axs = plt.subplots(Nr, Ncc, figsize=(20,40))   # gridspec_kw={'width_ratios':[2,1]} This sets different size for left and right subplots.
        
        counter = 0
        images = []
        for i in range(Nr):
            for j in range(1, -1, -1):
                       
                mydir = f'{mydirlist[counter]}\\'
                print(mydir)
                data = pd.read_csv(glob.glob(f'{mydir}*cosine.csv')[0])
                datar = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])
                # datap = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
                title = mydirlist[counter][-9:]                                              
                                
                if Plot_All_Traces is True:                
                    for m in datar[cutoff:cutoff2].columns[1:]:
                        # axs[i, j].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2])   # data_filt['Rsq'] > data_filt['Rsq'].quantile(0.25)
                        if Nw == 1 or Nc == 1:
                            axs[counter].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)
                            axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                        else:
                            axs[i, j].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)  
                            axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                        
                else:
                    # axs[i, j].plot(datar.index[cutoff:cutoff2], datar[cutoff:cutoff2].median(axis=1)) 
                    if Nw == 1 or Nc == 1:
                        axs[counter].plot(datar.index[cutoff:cutoff2], datar[cutoff:cutoff2].median(axis=1), linewidth=0.1) 
                        axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                    else:
                        axs[i, j].plot(datar.index[cutoff:cutoff2], datar[cutoff:cutoff2].median(axis=1), linewidth=0.1) 
                        axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                    
                # Update 29.5.2023
                if Nw == 1 or Nc == 1:
                    axs[counter].label_outer()
                    axs[counter].set_yticklabels([])
                    axs[counter].set_xticklabels([]) 
                    axs[counter].set_xlabel(f'{title}', fontsize=round(int(288/Nr)), labelpad=-5) 
                    axs[counter].set_xticks([])
                    axs[counter].set_yticks([])
                    axs[counter].spines['top'].set_visible(False) # to turn off individual borders 
                    axs[counter].spines['right'].set_visible(False)
                    axs[counter].spines['bottom'].set_visible(False)
                    axs[counter].spines['left'].set_visible(False)
                    
                    # Arrow settings
                    x_tail = treatment_time
                    y_tail = datar.median(axis=1).max()
                    x_head = treatment_time
                    y_head = datar.median(axis=1).max()/2 - datar.median(axis=1).max()/6
                    arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head), mutation_scale=50)  #FancyArrowPatch
                    axs[counter].add_patch(arrow)
                    
                else:    
                    axs[i, j].label_outer()
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_xticklabels([]) 
                    axs[i, j].set_xlabel(f'{title}', fontsize=round(int(288/Nr)), labelpad=-5) 
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
                    axs[i, j].spines['right'].set_visible(False)
                    axs[i, j].spines['bottom'].set_visible(False)
                    axs[i, j].spines['left'].set_visible(False)

                    # Arrow settings
                    x_tail = treatment_time
                    y_tail = datar.median(axis=1).max()
                    x_head = treatment_time
                    y_head = datar.median(axis=1).max()/2 - datar.median(axis=1).max()/6
                    arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head), mutation_scale=50)  #FancyArrowPatch
                    axs[i, j].add_patch(arrow)
                
                counter += 1

        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Composite_Trace.png')
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}Composite_Trace.svg', format = 'svg')
        plt.clf()
        plt.close()
        
    
    if graphtype == 'Phase_Histogram':
        
        # this loop assumes folder structure with multiple SCN folders and 2 subfolders (before and after treatment, first is after due to name)             
        fig, axh = plt.subplots(Nr, Ncc, subplot_kw={'projection': 'polar'})   
        fig.subplots_adjust(hspace=hspace, wspace=wspace)            # negative wspace moves left and right close but there is empty space
        counter = 0
        for i in range(Nr):        
            for j in range(1, -1, -1):
                
                mydir = f'{mydirlist[counter]}\\'
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
                data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])
                
                N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
                colorcode = circular_colors[0::int(round(len(circular_colors) / N_bins, 0))]   # return every 5th item from circular_colors array to get cca. 47 distinct color similar to cmap  
                
                outlier_reindex = ~(np.isnan(data['Amplitude']))     
                data_filt = data[data.columns[:].tolist()][outlier_reindex]                                 # data w/o amp outliers                                  
                phaseseries = data_filt['Phase'].values.flatten()                                           # plot all Phase
                # phaseseries = data_filt.loc[(data_filt['Rsq'] > data_filt['Rsq'].quantile(0.25)) & (data_filt['Amplitude'] > data_filt['Amplitude'].quantile(0.25)),'Phase'].values.flatten() # plot quantile filtered phase
                
                phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())    
                phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
                
                phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
                theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
                width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle
                
                polarhist_old(axh[i, j], mydirlist[counter][-9:], pad)  # function(input data, name of data taken from last 9 chars in path)
                counter += 1
        
        
        #plt.show() 
        #fig.tight_layout()
        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Phase_Polar_Histogram.png', dpi=600)
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}Phase_Polar_Histogram.svg', format = 'svg', dpi=600)
        plt.clf()
        plt.close()


    if graphtype == 'Phase_ScatterPlot':
        
        # this loop assumes folder structure with multiple SCN folders and 2 subfolders (before and after treatment, first is after due to name)             
        fig, axh = plt.subplots(Nr, Ncc, subplot_kw={'projection': 'polar'})   
        fig.subplots_adjust(hspace=hspace, wspace=wspace)            # negative wspace moves left and right close but there is empty space
        counter = 0
        for i in range(Nr):        
            for j in range(1, -1, -1):
                
                mydir = f'{mydirlist[counter]}\\'
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
                data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])
                
                # N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
                # colorcode = circular_colors[0::int(round(len(circular_colors) / N_bins, 0))]   # return every 5th item from circular_colors array to get cca. 47 distinct color similar to cmap  
                              
                outlier_reindex = ~(np.isnan(data['Amplitude']))    
                data_filt = data[data.columns[:].tolist()][outlier_reindex]                                 # data w/o amp outliers    
                phaseseries = data_filt['Phase'].values.flatten()                                           # plot all Phase
                # phaseseries = data_filt.loc[(data_filt['Rsq'] > data_filt['Rsq'].quantile(0.25)) & (data_filt['Amplitude'] > data_filt['Amplitude'].quantile(0.25)),'Phase'].values.flatten() # plot quantile filtered phase
                
                
                phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())    
                phase = np.radians(phaseseries)                               # if phase is in degrees (per2py))
                phase_rounded = np.round(phase, 1)
                r_from_amp = create_second_list(phase_rounded)                
                polarscatterplot(axh[i, j], mydirlist[counter][-9:], pad)                
                counter += 1
        
        
        #plt.show() 
        #fig.tight_layout()
        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Phase_Polar_Scatterplot.png', dpi=600)
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}Phase_Polar_Scatterplot.svg', format = 'svg', dpi=600)
        plt.clf()
        plt.close()
        
        
    if graphtype == 'Pie':   # create plots of Raw traces and cosines    

        # no of columns (main folders)
        fig, axh = plt.subplots(Nr, Ncc)   
        fig.subplots_adjust(hspace=hspace, wspace=wspace)  # negative wspace moves left and right close but there is empty space
        counter = 0     
        for i in range(Nr):            
            for j in range(1, -1, -1):
                       
                mydir = f'{mydirlist[counter]}\\'
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])

                pie_data = [len(data.loc[data['Rsq'] > rsq_threshold]), len(data.loc[data['Rsq'] <= rsq_threshold]), len(data.loc[data['Rsq'] != data['Rsq'], 'Rsq'])]
                keys = [f'rhythmic Rsq>{rsq_threshold}', f'Rsq<={rsq_threshold}', 'arrhythmic']               
             
                axh[i, j].pie(pie_data, labels=keys, autopct='%.0f%%', textprops=dict(fontsize=fontsize))  
                axh[i, j].set_title(f'{mydirlist[counter][-9:]}', fontdict=dict(fontsize=fontsize))

                counter += 1

        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Pie chart.png', dpi=600)
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}Pie chart.svg', format = 'svg', dpi=600)
        plt.clf()
        plt.close()
        

    if graphtype == 'Phase' or graphtype == 'Amplitude' or graphtype == 'Period' or graphtype == 'Trend':
        
        ##### Taken frim Matplotlib Example multimage
        # no of columns
        #Nc = 2
        fig, axs = plt.subplots(Nr, Ncc, sharey=sharey, sharex=sharex)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)  # negative wspace moves left and right close but there is empty space
        #fig.suptitle('Phase heatmaps')
        counter = 0
        images = []
        #alldata = pd.read_csv(glob.glob(f'{mydirlist[0]}\\*oscillatory_params.csv')[0]) #combine all datasets for control of max values, etc.
        #checkdata = pd.DataFrame(columns = ['X', 'Y', 'Amplitude'])
        for i in range(Nr):
            for j in range(1, -1, -1):
        
                mydir = f'{mydirlist[counter]}\\'
                print(mydir)
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
                data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])

                title = mydirlist[counter][-9:]

       
                def PrepHeatmap(graphtype, decimals=2):
                 
                    # round values to 1 decimal
                    data_round = np.round(data[['X', 'Y', graphtype]], decimals=2)  #adjust decimals if trouble with pivoting table
                    
                    # FILTER outliers by interquantile range filter: within 2.22 IQR (equiv. to z-score < 3), but too stringent for LV200, try higher iqr_value
                    outlier_reindex = ~(np.isnan(data_round[graphtype]))    
                    data_filtered = data[data.columns[:].tolist()][outlier_reindex]  
                    # cols = data_filt.select_dtypes('number').columns   # pick only numeric columns
                    # cols = [graphtype]    # pick hand selected columns
                    df_sub = data.loc[:, graphtype]
                    iqr = df_sub.quantile(0.75) - df_sub.quantile(0.25)
                    lim = np.abs((df_sub - df_sub.median()) / iqr) < iqr_value
                    # replace outliers with nan, except for Phase, where it is not needed
                    if graphtype != 'Phase':
                        data_filtered.loc[:, graphtype] = df_sub.where(lim, np.nan)   
                    # replace outlier-caused nans with median values
                    # data_filtered[graphtype].fillna(data_filtered[graphtype].median(), inplace=True)                               
                    
                    #data_filtered = reject_outliers(data_round, column=graphtype, m=10)   # alternative way to filter outliers
                    # pivot and transpose for heatmap format 
                    df_heat = data_filtered.pivot(index='X', columns='Y', values=graphtype).transpose()
                                         
                    
                    images.append(axs[i, j].imshow(df_heat.to_numpy(), cmap=cmap, rasterized=True))
                    axs[i, j].label_outer()
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_xticklabels([]) 
                    axs[i, j].set_xlabel(f'{title}', fontsize=2, labelpad=1) 
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
                    axs[i, j].spines['right'].set_visible(False)
                    axs[i, j].spines['bottom'].set_visible(False)
                    axs[i, j].spines['left'].set_visible(False)
                    
                PrepHeatmap(graphtype)
                    
                counter += 1
            
        def SaveHeatmap(graphtype, fontsize=5, labelpad=5):
            if graphtype == 'Phase':
            
                # Find the min and max of all colors for use in setting the color scale.
                #vmin = min(image.get_array().min() for image in images)
                #vmax = max(image.get_array().max() for image in images)
                # for phase - set vmin and vmax to match full circadian cycle to nicely cover the phase space with circular colormap
                vmin = 0
                vmax = 360
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                for im in images:
                    im.set_norm(norm)
                
                cbar = fig.colorbar(images[0], ticks=[45, 90, 135, 180, 225, 270, 315], ax=axs)
                cbar.ax.set_ylabel('Phase (h)', rotation=270, fontsize=fontsize, labelpad=labelpad)
                cbar.ax.set_yticklabels((3, 6, 9, 12, 15, 18, 21), fontsize=fontsize)
                #cbar.ax.set_xlabel('Phase in °')
            
            else:
                # Update 25.5.2023    
                if nonorm != False:
                    del images[nonorm]
                    
                # Find the min and max of all colors for use in setting the color scale.
                vmin = min(image.get_array().min() for image in images)
                vmax = max(image.get_array().max() for image in images)
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                if lognorm == True:
                    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
                for im in images:
                    im.set_norm(norm)
                
                cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
                cbar.ax.set_ylabel(graphtype, rotation=270, fontsize=fontsize, labelpad=labelpad)

        
            for im in images:
                # im.callbacksSM.connect('changed', update)  # in older matplotlib
                im.callbacks.connect('changed', update)

            ### To save as bitmap png for easy viewing ###
            plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.png', dpi=600)
            ### To save as vector svg with fonts editable in Corel ###
            plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.svg', format = 'svg', dpi=600)
            #plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.pdf', format = 'pdf', dpi=300)
            plt.clf()
            plt.close()
        
        SaveHeatmap(graphtype)
 

if experiment == 'rhythm':
    
    if graphtype == 'TraceHeatmaps':
        
        fig, axs = plt.subplots(nrows=Nw, ncols=Nc, sharex=True, sharey=False, figsize=(6,12))  # , sharey=False does not WORK as in RNAseq, due to looping?
        
        counter = 0
        images = []
        for i in range(Nw):
            for j in range(Nc):                
                
                # Update 21.3.2023
                if Nw*Nc -1 == Nr:
                    if counter == Nw*Nc -1:
                        break
                    
                mydir = f'{mydirlist[counter]}\\'
                print(mydir)
                data = pd.read_csv(glob.glob(f'{mydir}*cosine.csv')[0])
                datar = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])
                title = mydirlist[counter][-9:]
                df = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])    
                
                if Plot_All_Traces is True:              

                    if Nw == 1 or Nc == 1:
                        datar.pop(' ')
                        df_heat_spec1 = datar.T.sort_values(by=[0])                        
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[counter], cmap='YlGnBu_r', zorder=-10)
                                        
                    else:
                        datar.pop(' ')
                        df_heat_spec1 = datar.T.sort_values(by=[0])           
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[i, j], cmap='YlGnBu_r', zorder=-10)  #cbar_ax=axs[1], tell sns which ax to use  #cmap='coolwarm'

                if Plot_All_Traces == 'select_rsq':
                    df_heat_list = []
                    phase_heat_list = []
                    for m in datar.columns[1:]:
                        phase_value = df['CircPeak'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)]  #  CircPeak looks better than Phase                      
                        rsq_value = df['Rsq'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                        if rsq_value > rsq_threshold:
                            df_heat_list.append(datar[m][cutoff:cutoff2].values)
                            phase_heat_list.append(phase_value)
                        
                    if Nw == 1 or Nc == 1:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase'])                       
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[counter], cmap='YlGnBu_r', zorder=-10) 
                                        
                    else:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase'])  
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[i, j], cmap='YlGnBu_r', zorder=-10)          # viridis       
                        # for testing and colorbar printing
                        # fig, axs = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [20, 1]}) 
                        # sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[0], cbar_ax=axs[1], cmap='YlGnBu')
                        # plt.savefig(f'{path}TraceHeatmaps______test.png')

                if Plot_All_Traces == 'select_amp':
                    df_heat_list = []
                    phase_heat_list = []
                    for m in datar[cutoff:cutoff2].columns[1:]:  
                        phase_value = df['CircPeak'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)]  #  CircPeak looks better than Phase    
                        amp_value = df['Amplitude'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.                                         
                        if amp_value > amp_threshold:
                            df_heat_list.append(datar[m][cutoff:cutoff2].values)
                            phase_heat_list.append(phase_value)
                        
                    if Nw == 1 or Nc == 1:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase'])                    
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[counter], cmap='YlGnBu_r', zorder=-10) 
                                        
                    else:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))                        
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase'])
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[i, j], cmap='YlGnBu_r', zorder=-10) 

                if Plot_All_Traces == 'select_trend':
                    df_heat_list = []
                    phase_heat_list = []
                    for m in datar[cutoff:cutoff2].columns[1:]:
                        phase_value = df['CircPeak'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)]  #  CircPeak looks better than Phase                      
                        trend_value = df['Trend'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                        if trend_value > trend_threshold:
                            df_heat_list.append(datar[m][cutoff:cutoff2].values)
                            phase_heat_list.append(phase_value)
                        
                    if Nw == 1 or Nc == 1:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase'])                      
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[counter], cmap='YlGnBu_r', zorder=-10) 
                                        
                    else:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase']) 
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[i, j], cmap='YlGnBu_r', zorder=-10) 

                if Plot_All_Traces == 'select_decay':
                    df_heat_list = []
                    phase_heat_list = []
                    for m in datar[cutoff:cutoff2].columns[1:]:
                        phase_value = df['CircPeak'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)]  #  CircPeak looks better than Phase                      
                        decay_value = df['Decay'].iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                        if decay_value > decay_threshold:
                            df_heat_list.append(datar[m][cutoff:cutoff2].values)
                            phase_heat_list.append(phase_value)
                        
                    if Nw == 1 or Nc == 1:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase'])                      
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[counter], cmap='YlGnBu_r', zorder=-10) 
                                        
                    else:
                        df_heat = pd.DataFrame(df_heat_list, columns = np.arange(0, len(datar[cutoff:cutoff2].index)))
                        df_heat['Phase'] = phase_heat_list                        
                        df_heat_spec1 = df_heat.sort_values(by=['Phase'])  
                        df_heat_spec1 = df_heat_spec1.drop(columns=['Phase'])
                        sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=False, ax=axs[i, j], cmap='YlGnBu_r', zorder=-10)                         

                if Nw == 1 or Nc == 1:
                    
                    axs[counter].set_xlabel(f'{title}', fontsize=round(int(100/Nr)), labelpad=5) 
                    axs[counter].set_rasterization_zorder(0)

                else:
                    
                    axs[i, j].set_xlabel(f'{title}', fontsize=round(int(100/Nr)), labelpad=5) 
                    axs[i, j].set_rasterization_zorder(0)
                
                counter += 1

        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}TraceHeatmaps.png')
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}TraceHeatmaps.svg', format = 'svg') # need zorder=-10 in plt and axs[i, j].set_rasterization_zorder(0), svg ignores dpi
        plt.savefig(f'{path}TraceHeatmaps.pdf', dpi=600)
        plt.clf()
        plt.close()
    

    if graphtype == 'Trace':   # create plots of Raw traces and cosines
        
        fig, axs = plt.subplots(Nw, Nc, figsize=(6,12))        # figsize=(20,40)
        
        counter = 0
        images = []
        for i in range(Nw):
            for j in range(Nc):                
                
                # Update 21.3.2023
                if Nw*Nc -1 == Nr:
                    if counter == Nw*Nc -1:
                        break
                    
                mydir = f'{mydirlist[counter]}\\'
                print(mydir)
                data = pd.read_csv(glob.glob(f'{mydir}*cosine.csv')[0])
                datar = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])
                title = mydirlist[counter][-9:]
                # add Rsq to signal column and filter only Rsq > 0.9
                """ ASKED AI
                I have one pandas dataframe called datar = pd.DataFrame({'Mean1': [1, 5, 3, 0, 3],  'Mean2': [1, 2, 3, 2, 1],
                   'Mean3': [5, 7, 9, 5, 0],
                   'Mean4': [1, 3, 5, 3, 1]}) .  I also have second dataframe called df = pd.DataFrame({'Rsq': [0.95, 0.8, 0.7, 0.91]}) - 
                df values match corresponding column from dataframe datar. I need to iterate over all columns from dataframe datar and plot all 
                their values, but only when the corresponding value from dataframe df is larger than 0.9.
                Replied with df...columns...get_loc(m) tip.
                """
                                
                df = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
                
                if Plot_All_Traces == 'select_rsq':

                    if Nw == 1 or Nc == 1:
                        axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                        
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            rsq_value = df['Rsq'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if rsq_value > rsq_threshold:                      
                                axs[counter].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)
              
                    else:
                        axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                    
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            rsq_value = df['Rsq'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if rsq_value > rsq_threshold:                      
                                axs[i, j].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1) 
                                                             
                if Plot_All_Traces == 'select_amp':

                    if Nw == 1 or Nc == 1:
                        axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                        
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            amp_value = df['Amplitude'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if amp_value > amp_threshold:                      
                                axs[counter].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)
              
                    else:
                        axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                    
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            amp_value = df['Amplitude'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if amp_value > amp_threshold:                      
                                axs[i, j].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)                            

                if Plot_All_Traces == 'select_trend':

                    if Nw == 1 or Nc == 1:
                        axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                        
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            trend_value = df['Trend'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if trend_value > trend_threshold:                      
                                axs[counter].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)
              
                    else:
                        axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                    
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            trend_value = df['Trend'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if trend_value > trend_threshold:                      
                                axs[i, j].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)                     

                if Plot_All_Traces == 'select_decay':

                    if Nw == 1 or Nc == 1:
                        axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                        
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            decay_value = df['Decay'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if decay_value > decay_threshold:                      
                                axs[counter].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)
              
                    else:
                        axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')                    
                        for m in datar[cutoff:cutoff2].columns[1:]:                            
                            decay_value = df['Decay'].T.iloc[datar[cutoff:cutoff2].columns[1:].get_loc(m)] # Get integer location for requested label.
                            if decay_value > decay_threshold:                      
                                axs[i, j].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)      

                if Plot_All_Traces is True:
                    
                    if Nw == 1 or Nc == 1:
                        axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                        for m in datar[cutoff:cutoff2].columns[1:]:
                            axs[counter].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1)

                    else:
                        axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                        for m in datar[cutoff:cutoff2].columns[1:]:
                            axs[i, j].plot(datar.index[cutoff:cutoff2], datar[m][cutoff:cutoff2], linewidth=0.1) 
                                                   
                else:

                    if Nw == 1 or Nc == 1:
                        axs[counter].plot(datar.index[cutoff:cutoff2], datar[cutoff:cutoff2].median(axis=1)) 
                        axs[counter].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                    else:
                        axs[i, j].plot(datar.index[cutoff:cutoff2], datar[cutoff:cutoff2].median(axis=1)) 
                        axs[i, j].plot(data.index[cutoff:cutoff2], data[cutoff:cutoff2].median(axis=1), color='r')
                              
                # Update 29.5.2023
                if Nw == 1 or Nc == 1:
                    axs[counter].label_outer()
                    axs[counter].set_yticklabels([])
                    axs[counter].set_xticklabels([]) 
                    axs[counter].set_xlabel(f'{title}', fontsize=round(int(100/Nr)), labelpad=-5) 
                    axs[counter].set_xticks([])
                    axs[counter].set_yticks([])
                    axs[counter].spines['top'].set_visible(False) # to turn off individual borders 
                    axs[counter].spines['right'].set_visible(False)
                    axs[counter].spines['bottom'].set_visible(False)
                    axs[counter].spines['left'].set_visible(False)
                    
                else:    
                    axs[i, j].label_outer()
                    # axs[i, j].set_yticklabels([])
                    # axs[i, j].set_xticklabels([]) 
                    axs[i, j].set_xlabel(f'{title}', fontsize=round(int(100/Nr)), labelpad=-5)  #288/Nr
                    # axs[i, j].set_xticks([])
                    # axs[i, j].set_yticks([])
                    axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
                    axs[i, j].spines['right'].set_visible(False)
                    # axs[i, j].spines['bottom'].set_visible(False)
                    # axs[i, j].spines['left'].set_visible(False)
                    # axs[i, j].set_rasterization_zorder(0)

                
                counter += 1

        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Composite_Trace.png')
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}Composite_Trace.svg', format = 'svg')
        plt.clf()
        plt.close()
        

    if graphtype == 'Pie':   # create plots of Raw traces and cosines
    

        # no of columns (main folders)
        fig, axh = plt.subplots(Nw, Nc)   
        fig.subplots_adjust(hspace=hspace, wspace=wspace)  # negative wspace moves left and right close but there is empty space
        counter = 0     
        for i in range(Nw):            
            for j in range(Nc):                
                
                # Update 21.3.2023
                if Nw*Nc -1 == Nr:
                    if counter == Nw*Nc -1:
                        break
                       
                mydir = f'{mydirlist[counter]}\\'
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])

                pie_data = [len(data.loc[data['Rsq'] > rsq_threshold]), len(data.loc[data['Rsq'] <= rsq_threshold]), len(data.loc[data['Rsq'] != data['Rsq'], 'Rsq'])]
                keys = [f'rhythmic Rsq>{rsq_threshold}', f'Rsq<={rsq_threshold}', 'arrhythmic']               
             
                if Nw == 1 or Nc == 1:
                    axh[counter].pie(pie_data, labels=keys, autopct='%.0f%%', textprops=dict(fontsize=fontsize))  
                    axh[counter].set_title(f'{mydirlist[counter][-9:]}', fontdict=dict(fontsize=fontsize))
                else:
                    axh[i, j].pie(pie_data, labels=keys, autopct='%.0f%%', textprops=dict(fontsize=fontsize))  
                    axh[i, j].set_title(f'{mydirlist[counter][-9:]}', fontdict=dict(fontsize=fontsize))

                counter += 1

        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Pie chart.png', dpi=600)
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}Pie chart.svg', format = 'svg', dpi=600)
        plt.clf()
        plt.close()
        


    
    if graphtype == 'Phase_Histogram':
            
        # no of columns (main folders)
        fig, axh = plt.subplots(Nw, Nc, subplot_kw={'projection': 'polar'})   
        fig.subplots_adjust(hspace=hspace, wspace=wspace)  # negative wspace moves left and right close but there is empty space
        #fig.suptitle('Phase heatmaps')
        counter = 0
        images = []
        pval_Rt_list = []
        v_length_list = []
        #alldata = pd.read_csv(glob.glob(f'{mydirlist[0]}\\*oscillatory_params.csv')[0]) #combine all datasets for control of max values, etc.
        #checkdata = pd.DataFrame(columns = ['X', 'Y', 'Amplitude'])        
        for i in range(Nw):            
            for j in range(Nc):                
                
                # Update 21.3.2023
                if Nw*Nc -1 == Nr:
                    if counter == Nw*Nc -1:
                        break
                       
                mydir = f'{mydirlist[counter]}\\'
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
                data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])
                
                N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
                colorcode = circular_colors[0::int(round(len(circular_colors) / N_bins, 0))]   # return every 5th item from circular_colors array to get cca. 47 distinct color similar to cmap  
                
                outlier_reindex = ~(np.isnan(data['Amplitude']))    
                data_filt = data[data.columns[:].tolist()][outlier_reindex]                                 # data w/o amp outliers    
                phaseseries = data_filt['Phase'].values.flatten()                                           # plot Phase
                phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())    
                phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
                
                datao = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
                rsqlist = list(datao['Rsq'])
                amplist = list(datao['Amplitude'])
                trendlist = list(datao['Trend'])
                decaylist = list(datao['Decay'])
                             
                if Plot_All_Traces == 'select_rsq':  # select_amp
                    phase_rsq = zip(phase, rsqlist)
                    phase_rsq = [p for p in phase_rsq if p[1] > rsq_threshold]                    
                    phase = list(zip(*phase_rsq))[0]                    
                    
                if Plot_All_Traces == 'select_amp':
                    phase_amp = zip(phase, amplist)
                    phase_amp = [p for p in phase_amp if p[1] > amp_threshold]                    
                    phase = list(zip(*phase_amp))[0]                       

                if Plot_All_Traces == 'select_trend':
                    phase_trend = zip(phase, trendlist)
                    phase_trend = [p for p in phase_trend if p[1] > trend_threshold]                    
                    phase = list(zip(*phase_trend))[0]    
                    
                if Plot_All_Traces == 'select_decay':
                    phase_decay = zip(phase, decaylist)
                    phase_decay = [p for p in phase_decay if p[1] < decay_threshold]                    
                    phase = list(zip(*phase_decay))[0]  
                
                phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
                theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
                width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle              
                             
                if Nw == 1 or Nc == 1:
                    polarhist(axh[counter], mydirlist[counter][-9:], pval_Rt_list, v_length_list, pad)
                else:
                    polarhist(axh[i, j], mydirlist[counter][-9:], pval_Rt_list, v_length_list, pad)
                
                # polarhist(axh[i, j], mydirlist[counter][-9:], pad)  # function(input data, name of data taken from last 9 chars in path)
                counter += 1
        print('p values of Rayleigh vector:')
        print(pval_Rt_list)
        print()
        print('Rayleigh vector lengths:')
        print(v_length_list)
        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Phase_Polar_Histogram.png', dpi=600)
        ### To save as vector svg with fonts editable in Corel ###
        plt.savefig(f'{path}Phase_Polar_Histogram.svg', format = 'svg', dpi=600)
        plt.clf()
        plt.close()
                        
        
    # Experimental plot, not tested yet, NOT WORKING
    if graphtype == 'Phase_ScatterPlot':
        
        # this loop assumes folder structure with multiple SCN folders and 2 subfolders (before and after treatment, first is after due to name)             
        fig, axh = plt.subplots(Nr, Ncc, subplot_kw={'projection': 'polar'})   
        fig.subplots_adjust(hspace=hspace, wspace=wspace)            # negative wspace moves left and right close but there is empty space
        counter = 0
        for i in range(Nr):        
            for j in range(1, -1, -1):
                
                mydir = f'{mydirlist[counter]}\\'
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
                data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])
                
                # N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
                # colorcode = circular_colors[0::int(round(len(circular_colors) / N_bins, 0))]   # return every 5th item from circular_colors array to get cca. 47 distinct color similar to cmap  
                              
                outlier_reindex = ~(np.isnan(data['Amplitude']))    
                data_filt = data[data.columns[:].tolist()][outlier_reindex]                                 # data w/o amp outliers    
                phaseseries = data_filt['Phase'].values.flatten()                                           # plot all Phase
                # phaseseries = data_filt.loc[(data_filt['Rsq'] > data_filt['Rsq'].quantile(0.25)) & (data_filt['Amplitude'] > data_filt['Amplitude'].quantile(0.25)),'Phase'].values.flatten() # plot quantile filtered phase
                
                
                phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())    
                phase = np.radians(phaseseries)                               # if phase is in degrees (per2py))
                
                datao = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])
                rsqlist = list(datao['Rsq'])
                amplist = list(datao['Amplitude'])
                trendlist = list(datao['Trend'])
                decaylist = list(datao['Decay'])
                             
                if Plot_All_Traces == 'select_rsq':  # select_amp
                    phase_rsq = zip(phase, rsqlist)
                    phase_rsq = [p for p in phase_rsq if p[1] > rsq_threshold]                    
                    phase = list(zip(*phase_rsq))[0]                    
                    
                if Plot_All_Traces == 'select_amp':
                    phase_amp = zip(phase, amplist)
                    phase_amp = [p for p in phase_amp if p[1] > amp_threshold]                    
                    phase = list(zip(*phase_amp))[0]                       

                if Plot_All_Traces == 'select_trend':
                    phase_trend = zip(phase, trendlist)
                    phase_trend = [p for p in phase_trend if p[1] > trend_threshold]                    
                    phase = list(zip(*phase_trend))[0]    
                    
                if Plot_All_Traces == 'select_decay':
                    phase_decay = zip(phase, decaylist)
                    phase_decay = [p for p in phase_decay if p[1] < decay_threshold]                    
                    phase = list(zip(*phase_decay))[0]  
                
                phase_rounded = np.round(phase, 1)
                r_from_amp = create_second_list(phase_rounded)                
                polarscatterplot(axh=axh[i, j], title=mydirlist[counter][-9:], phase_rounded=phase_rounded, r_from_amp=r_from_amp, pad=pad)                
                counter += 1
    
    
    if graphtype == 'Phase' or graphtype == 'Amplitude' or graphtype == 'Period' or graphtype == 'Trend':
           
        ##### Taken from Matplotlib Example multimage
        fig, axs = plt.subplots(Nw, Nc, sharey=sharey, sharex=sharex)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)  # negative wspace moves left and right close but there is empty space  
        #fig.suptitle('Phase heatmaps')
        counter = 0
        images = []
        #alldata = pd.read_csv(glob.glob(f'{mydirlist[0]}\\*oscillatory_params.csv')[0]) #combine all datasets for control of max values, etc.
        #checkdata = pd.DataFrame(columns = ['X', 'Y', 'Amplitude'])
        for i in range(Nw):
            for j in range(Nc):

                # Update 21.3.2023
                if Nw*Nc -1 == Nr:
                    if counter == Nw*Nc -1:
                        break
        
                mydir = f'{mydirlist[counter]}\\'
                print(mydir)
        
                # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
                try:
                    data_hl = pd.read_csv(glob.glob(f'{mydir}*decay_params.csv')[0])
                except IndexError:
                    data_hl = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
                
                title = mydirlist[counter][-9:]
                
                if Plot_All_Traces == 'select_rsq':
                    data_hl = data_hl[data_hl['Rsq'] > rsq_threshold]      
                
                if Plot_All_Traces == 'select_amp':
                    data_hl = data_hl[data_hl['Amplitude'] > amp_threshold]      
                
                if Plot_All_Traces == 'select_trend':
                    data_hl = data_hl[data_hl['Trend'] > trend_threshold]      
                    
                if Plot_All_Traces == 'select_decay':
                    data_hl = data_hl[data_hl['Decay'] > decay_threshold]      

                  
                data_h = np.round(data_hl[['X', 'Y', graphtype]], decimals=2)
                outlier_reindex = ~(np.isnan(data_h[graphtype]))    
                data_filtered = data_h[data_h.columns[:].tolist()][outlier_reindex]  
                
                if disable_parameters_iqr is not True:
                    # FILTER outliers by interquantile range filter: within 2.22 IQR (equiv. to z-score < 3), but too stringent for LV200, try higher iqr_value
                    df_sub = data_filtered.loc[:, graphtype]
                    iqr = df_sub.quantile(0.75) - df_sub.quantile(0.25)
                    lim = np.abs((df_sub - df_sub.median()) / iqr) < iqr_value
                    # replace outliers with nan, except for Phase, where it is not needed
                    if graphtype != 'Phase':
                        data_filtered.loc[:, graphtype] = df_sub.where(lim, np.nan)   
                    # replace outlier-caused nans with median values
                    # data_filtered[graphtype].fillna(data_filtered[graphtype].median(), inplace=True)                

             
                # pivot and transpose for heatmap format
                df_heat = data_filtered.pivot(index='X', columns='Y', values=graphtype).transpose()
                
                # Update 21.3.2023
                if Nw == 1 or Nc == 1:
                    images.append(axs[counter].imshow(df_heat.to_numpy(), cmap=cmap, rasterized=True))
                    axs[counter].label_outer()
                    axs[counter].set_yticklabels([])
                    axs[counter].set_xticklabels([]) 
                    axs[counter].set_xlabel(f'{title}', fontsize=6, labelpad=1) 
                    axs[counter].set_xticks([])
                    axs[counter].set_yticks([])
                    axs[counter].spines['top'].set_visible(False) # to turn off individual borders 
                    axs[counter].spines['right'].set_visible(False)
                    axs[counter].spines['bottom'].set_visible(False)
                    axs[counter].spines['left'].set_visible(False)             
                                                
                else:                                                                                              
                    images.append(axs[i, j].imshow(df_heat.to_numpy(), cmap=cmap, rasterized=True))
                    axs[i, j].label_outer()
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_xticklabels([]) 
                    axs[i, j].set_xlabel(f'{title}', fontsize=6, labelpad=1) 
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
                    axs[i, j].spines['right'].set_visible(False)
                    axs[i, j].spines['bottom'].set_visible(False)
                    axs[i, j].spines['left'].set_visible(False)
        
                counter += 1
        
        if graphtype == 'Phase':
    
            # for phase - set vmin and vmax to match full circadian cycle to nicely cover the phase space with circular colormap
            vmin = 0
            vmax = 360
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            
            cbar = fig.colorbar(images[0], ticks=[45, 90, 135, 180, 225, 270, 315], ax=axs)
            cbar.ax.set_ylabel('Phase (h)', rotation=270, labelpad=12) # fontsize=5, 
            cbar.ax.set_yticklabels((3, 6, 9, 12, 15, 18, 21)) #, fontsize=11       
            
        else:                        
            # Update 25.5.2023    
            if nonorm != False:
                del images[nonorm]
            
            # Find the min and max of all colors for use in setting the color scale.
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            if lognorm == True:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)                     
            for im in images:
                im.set_norm(norm)
            
            cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
            cbar.ax.set_ylabel(graphtype, rotation=270, labelpad=11)       
           
        for im in images:
            im.callbacks.connect('changed', update)
            
        ### To save as bitmap png for easy viewing ###
        plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.png', dpi=600) # 
        ### To save as vector svg with fonts editable in Corel ###
        # plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.svg', format = 'svg', rasterized = True, dpi=600)
        plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.svg', format = 'svg', dpi=600)
        # plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.pdf', dpi=600)
        plt.clf()
        plt.close() 



if experiment == 'decay':    
       
    ##### Taken frim Matplotlib Example multimage
    fig, axs = plt.subplots(Nw, Nc, sharey=sharey, sharex=sharex)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)  # negative wspace moves left and right close but there is empty space
    #fig.suptitle('Phase heatmaps')
    counter = 0
    images = []
    #alldata = pd.read_csv(glob.glob(f'{mydirlist[0]}\\*oscillatory_params.csv')[0]) #combine all datasets for control of max values, etc.
    #checkdata = pd.DataFrame(columns = ['X', 'Y', 'Amplitude'])
    for i in range(Nw):
        for j in range(Nc):            
            
            # Update 21.3.2023
            if Nw*Nc -1 == Nr:
                if counter == Nw*Nc -1:
                    break
                    
            mydir = f'{mydirlist[counter]}\\'
            print(mydir)
    
            # LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
            #data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
            #data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])
            data_hl = pd.read_csv(glob.glob(f'{mydir}*decay_params.csv')[0])      
            #data_raw = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])
            
            title = mydirlist[counter][-9:]
            
            data_h = np.round(data_hl[['X', 'Y', graphtype]], decimals=2)
            #data_filtered = reject_outliers(data_h, column=graphtype)
            # pivot and transpose for heatmap format
            df_heat = data_h.pivot(index='X', columns='Y', values=graphtype).transpose()
            
            # Update 21.3.2023
            if Nw == 1 or Nc == 1:
                images.append(axs[counter].imshow(df_heat.to_numpy(), cmap=cmap, rasterized=True))
                axs[counter].label_outer()
                axs[counter].set_yticklabels([])
                axs[counter].set_xticklabels([]) 
                axs[counter].set_xlabel(f'{title}', fontsize=6, labelpad=1) 
                axs[counter].set_xticks([])
                axs[counter].set_yticks([])
                axs[counter].spines['top'].set_visible(False) # to turn off individual borders 
                axs[counter].spines['right'].set_visible(False)
                axs[counter].spines['bottom'].set_visible(False)
                axs[counter].spines['left'].set_visible(False)             
                                            
            else:                                                                     
                images.append(axs[i, j].imshow(df_heat.to_numpy(), cmap=cmap, rasterized=True))
                axs[i, j].label_outer()
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([]) 
                axs[i, j].set_xlabel(f'{title}', fontsize=6, labelpad=1) 
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['bottom'].set_visible(False)
                axs[i, j].spines['left'].set_visible(False)
    
            counter += 1
    
    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    
    cbar = fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=.1)
    cbar.ax.set_ylabel(graphtype, rotation=270, labelpad=11)
    
    for im in images:
        im.callbacks.connect('changed', update)
    
    #plt.show() 
    #fig.tight_layout()
    ### To save as bitmap png for easy viewing ###
    plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.png', dpi=600)
    ### To save as vector svg with fonts editable in Corel ###
    plt.savefig(f'{path}Composite_Heatmap_XY_{graphtype}.svg', format = 'svg', dpi=600)
    plt.clf()
    plt.close()
    

# copy or move and rename animated gif files from all TIFF folders to anal folder
if graphtype == 'GIFS': 
    
    import shutil

    # Specify FOLDER
    root = Tk()
    folder_path = StringVar()
    lbl1 = Label(master=root, textvariable=folder_path)
    lbl1.grid(row=0, column=1)
    buttonBrowse = Button(text="Browse to TIFFs", command=browse_button)
    buttonBrowse.grid()
    mainloop()
    tiff_path = os.getcwd() + '\\'
    
    depth2 = tiff_path.count(os.sep)
    
    tiff_paths = []
    for root, dirs, files in os.walk(tiff_path, topdown=False):
        for files in dirs:        
            folder = os.path.join(root, files)
            if folder.count(os.sep) == depth2 + 2:      # if gif is in mod2 folder in time-stamped subfolder, otherwise adjust
                tiff_paths.append(folder)
    
    counter = 0
    for oldfolder in tiff_paths:
        for root, dirs, files in os.walk(oldfolder):
            for name in files:
                if name.endswith('gif'):
                    # newname = tiff_paths[counter][-30:-25]
                    newname = mydirlist[counter][-4:]   # if error, try the line above instead
                    # os.rename(os.path.join(root, name), f'{path}{newname}')
                    shutil.copy(os.path.join(root, name), f'{path}{newname}.gif')
        counter += 1
        
 
# create tablo from sequence of tiffs, choose how many (Nx) from set start (Ns), need to be square grid, e.g. 5x5 or 7x7
if graphtype == 'TIFSEQUENCE':         
        
    import shutil
    from skimage import io
    from skimage.transform import rescale
    import skimage

    # Specify FOLDER
    root = Tk()
    folder_path = StringVar()
    lbl1 = Label(master=root, textvariable=folder_path)
    lbl1.grid(row=0, column=1)
    buttonBrowse = Button(text="Browse to TIFFs", command=browse_button)
    buttonBrowse.grid()
    mainloop()
    tiff_path = os.getcwd() + '\\'
    
    depth2 = tiff_path.count(os.sep)
    
    tiff_paths = []
    for root, dirs, files in os.walk(tiff_path, topdown=False):
        for files in dirs:        
            folder = os.path.join(root, files)
            if folder.count(os.sep) == depth2 + 2:      # gif is in mod2 folder
                tiff_paths.append(folder) 
                
    ##### Taken from Matplotlib Example multimage
    fig, axs = plt.subplots(Nx, Nx, figsize=(Nx,Nx))  # sharey=sharey, sharex=sharex, 
    fig.subplots_adjust(hspace=0, wspace=-0.18)  # negative wspace moves left and right close but there is empty space
    counter = 0
    images = []    

    ######### Ceate LIST of image files in cwd and put their path to list
    files = []
    with os.scandir(tiff_path) as it:
        for entry in it:
            if entry.name.endswith(('.tif', '.png')) and entry.is_file():
                print(entry.name)                                           #, entry.path
                files.append(entry.path)
    
    ######### READ all sorted files and concatenate into list of arrays
    files.sort()
    # conc_list = []
    # for file in files[Ns-1:Nx*Nx+Ns]:
    #     print(file)
    #     img = io.imread(file)
    #     # rescaled = rescale(img, 0.6, anti_aliasing=False)  # downsize to 0.6 to avoid crashes doe  io.imread_collection io.concatenate_images(ic)
    #     # img = skimage.img_as_ubyte(rescaled)
    #     conc_list += [img] 
    conc_list = io.concatenate_images(io.imread_collection(files))  # alternative way, read all at once    

    # To display image stored as array>>> plt.imshow(conc_list[0]), then plt.show()                                                     
    for i in range(Nx):
        for j in range(Nx):                                                               
            images.append(axs[i, j].imshow(conc_list[counter], cmap=cmap))
            axs[i, j].label_outer()
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([]) 
            # axs[i, j].set_xlabel([]) 
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_visible(False)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].set_aspect('equal')

            counter += 1
    
            # Need to use this otherwise the grayscale is stretched
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            if lognorm == True:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)            
   
        for im in images:
            # im.callbacksSM.connect('changed', update)  # in older matplotlib
            im.callbacks.connect('changed', update)

    #plt.show() 
    #fig.tight_layout()
    ### To save as bitmap png for easy viewing ###
    plt.savefig(f'{path}_Tablo_{graphtype}.png', dpi=600, bbox_inches="tight")
    ### To save as vector svg with fonts editable in Corel ###
    plt.savefig(f'{path}_Tablo_{graphtype}.svg', format = 'svg', dpi=600, bbox_inches="tight")
    plt.clf()
    plt.close()                
                


if graphtype == 'Parameters':      
    
    # Preallocate memory for DataFrame df2
    num_rows = 5000  # Adjust this number based on your expected maximum number of rows, solves PROBLEM with extracting only some cells
    df2 = pd.DataFrame(index=range(num_rows))    
    # Initialize row index
    row_index = 0

    for mydir in mydirlist:
        data2 = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])  
        if Plot_All_Traces == 'select_rsq':
            data2 = data2[data2['Rsq'] > rsq_threshold]             
        if Plot_All_Traces == 'select_amp':
            data2 = data2[data2['Amplitude'] > amp_threshold]
        if Plot_All_Traces == 'select_trend':
            data2 = data2[data2['Trend'] > trend_threshold]             
        if Plot_All_Traces == 'select_decay':
            data2 = data2[data2['Decay'] > decay_threshold]    
        df2['Period' + str(mydir[name:nameend])] = data2['Period']    
    for mydir in mydirlist:
        data2 = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])         
        if Plot_All_Traces == 'select_rsq':
            data2 = data2[data2['Rsq'] > rsq_threshold]             
        if Plot_All_Traces == 'select_amp':
            data2 = data2[data2['Amplitude'] > amp_threshold]
        if Plot_All_Traces == 'select_trend':
            data2 = data2[data2['Trend'] > trend_threshold]             
        if Plot_All_Traces == 'select_decay':
            data2 = data2[data2['Decay'] > decay_threshold]    
        df2['Amplitude' + str(mydir[name:nameend])] = data2['Amplitude']
    for mydir in mydirlist:
        data2 = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])  
        if Plot_All_Traces == 'select_rsq':
            data2 = data2[data2['Rsq'] > rsq_threshold]             
        if Plot_All_Traces == 'select_amp':
            data2 = data2[data2['Amplitude'] > amp_threshold]
        if Plot_All_Traces == 'select_trend':
            data2 = data2[data2['Trend'] > trend_threshold]             
        if Plot_All_Traces == 'select_decay':
            data2 = data2[data2['Decay'] > decay_threshold]  
        df2['Phase' + str(mydir[name:nameend])] = data2['Phase']
    for mydir in mydirlist:
        data2 = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])  
        if Plot_All_Traces == 'select_rsq':
            data2 = data2[data2['Rsq'] > rsq_threshold]             
        if Plot_All_Traces == 'select_amp':
            data2 = data2[data2['Amplitude'] > amp_threshold]
        if Plot_All_Traces == 'select_trend':
            data2 = data2[data2['Trend'] > trend_threshold]             
        if Plot_All_Traces == 'select_decay':
            data2 = data2[data2['Decay'] > decay_threshold]  
        df2['Trend' + str(mydir[name:nameend])] = data2['Trend']
    for mydir in mydirlist:
        data2 = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])  
        if Plot_All_Traces == 'select_rsq':
            data2 = data2[data2['Rsq'] > rsq_threshold]             
        if Plot_All_Traces == 'select_amp':
            data2 = data2[data2['Amplitude'] > amp_threshold]
        if Plot_All_Traces == 'select_trend':
            data2 = data2[data2['Trend'] > trend_threshold]             
        if Plot_All_Traces == 'select_decay':
            data2 = data2[data2['Decay'] > decay_threshold]  
        df2['Decay' + str(mydir[name:nameend])] = data2['Decay']
    for mydir in mydirlist:
        data2 = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])  
        if Plot_All_Traces == 'select_rsq':
            data2 = data2[data2['Rsq'] > rsq_threshold]             
        if Plot_All_Traces == 'select_amp':
            data2 = data2[data2['Amplitude'] > amp_threshold]
        if Plot_All_Traces == 'select_trend':
            data2 = data2[data2['Trend'] > trend_threshold]             
        if Plot_All_Traces == 'select_decay':
            data2 = data2[data2['Decay'] > decay_threshold]  
        df2['Rsq' + str(mydir[name:nameend])] = data2['Rsq']
    
    df2 = df2.dropna(how='all') # remove empty rows
    df2.to_csv(f'{path}Composite_parameters.csv')    
    
    ticks = [i[name+1:nameend] for i in mydirlist]
        
    # Violin plot from wide-format dataframe 
    def violin(data2, title, ticks=ticks, remove_outliers=True):
        title = title               
        ax = sns.violinplot(data=data2)
        plt.title(title, fontsize=14)
        ax.axes.xaxis.set_ticklabels(ticks)
        plt.xticks(rotation=90)
        plt.savefig(f'{path}Violin_{title}.png', format = 'png', bbox_inches = 'tight')   
        plt.savefig(f'{path}Violin_{title}.svg', format = 'svg', bbox_inches = 'tight')
        plt.clf()
        plt.close()
    
    i = len(mydirlist)
    
    violin(df2.iloc[:, 2*i:3*i], 'Phase')
    
    # For other parameters, first replace outliers with nans
    if disable_parameters_iqr is not True:
        for col in df2.columns.values:
            # FILTER outliers by iqr filter: within 2.22 IQR (equiv. to z-score < 3)
            iqr = df2[col].quantile(0.75) - df2[col].quantile(0.25)
            lim = np.abs((df2[col] - df2[col].median()) / iqr) < iqr_value
            df2.loc[:, col] = df2[col].where(lim, np.nan)
    
    violin(df2.iloc[:, 0:i], 'Period')
    violin(df2.iloc[:, i:2*i], 'Amplitude')
    violin(df2.iloc[:, 3*i:4*i], 'Trend')
    violin(df2.iloc[:, 4*i:5*i], 'Decay')
    violin(df2.iloc[:, 5*i:6*i], 'Rsq')

    """Ask AI
    I have one pandas dataframe called ddf2 = pd.DataFrame({'Period\CHP1': [1, 5, 3, 0, 3,...],  'Period\CHP2': [1, 2, 3, 2, 1,...],  
                                                            'Period\CHP3': [5, 7, 9, 5, 0,...], 'Period\CHP4': [1, 3, 5, 3, 1,...], 
                                                            'Period\CHP5': [2, 3, 5, 4, 1,...], 'Period\CHP6': [0, 0, 0, 0, 1,...]}) . 
    I need to make another dataframe df with just two columns {'Period_CHP123': [1, 5, 3, 0, 3, ..., 1, 2, 3, 2, 1, ..., 5, 7, 9, 5, 0,...], 
                                                               'Period_CHP456': [1, 3, 5, 3, 1,..., 2, 3, 5, 4, 1,..., 0, 0, 0, 0, 1,...]}.    
    """
    
    # !!! THIS IS EXPERIMENT SPECIFIC
    if combineGroup == True:
        
        
        # FOR TTX DEX experiments e5-e9
        
        df3 = df2[['PeriodEX01\A', 'PeriodEX01\B', 'PeriodEX02\A', 'PeriodEX02\B', 'PeriodEX06\A', 'PeriodEX06\B', 'PeriodEX08\A', 'PeriodEX08\B', 'PeriodEX10\A', 'PeriodEX10\B',
                   'PeriodEH01\A', 'PeriodEH01\B', 'PeriodEH02\A', 'PeriodEH02\B', 'PeriodEH03\A', 'PeriodEH03\B', 'PeriodEH04\A', 'PeriodEH04\B', 'PeriodEH05\A', 'PeriodEH05\B',
                   'PeriodEH07\A', 'PeriodEH07\B', 'PeriodEH08\A', 'PeriodEH08\B', 'PeriodEH09\A', 'PeriodEH09\B']]
        # Rename columns
        df3.columns = ['Period_DEX01_A', 'Period_DEX01_B', 'Period_DEX02_A', 'Period_DEX02_B', 'Period_DEX06_A', 'Period_DEX06_B', 'Period_DEX08_A', 'Period_DEX08_B', 'Period_DEX10_A', 'Period_DEX10_B',
                   'Period_VEH01_A', 'Period_VEH01_B', 'Period_VEH02_A', 'Period_VEH02_B', 'Period_VEH03_A', 'Period_VEH03_B', 'Period_VEH04_A', 'Period_VEH04_B', 'Period_VEH05_A', 'Period_VEH05_B',
                   'Period_VEH07_A', 'Period_VEH07_B', 'Period_VEH08_A', 'Period_VEH08_B', 'Period_VEH09_A', 'Period_VEH09_B']        
        
        melted = df3.melt(value_vars=df3.columns, var_name='Var', value_name='Value')              
        # Extract the 'SCN' and period numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Period'] = melted['Var'].str.split('_').str[0]
        melted['Bef_Aft'] = melted['Var'].str.split('_').str[2]
        
        # Create normalized values After/Before
        DEX01 = pd.DataFrame()
        DEX01['Period'] = (melted.loc[(melted['Prefix'] == 'DEX01') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'DEX01') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        DEX01['Prefix'] = 'DEX01'
        DEX02 = pd.DataFrame()        
        DEX02['Period'] = (melted.loc[(melted['Prefix'] == 'DEX02') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'DEX02') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        DEX02['Prefix'] = 'DEX02'
        DEX06 = pd.DataFrame()
        DEX06['Period'] = (melted.loc[(melted['Prefix'] == 'DEX06') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'DEX06') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        DEX06['Prefix'] = 'DEX06'
        DEX08 = pd.DataFrame()
        DEX08['Period'] = (melted.loc[(melted['Prefix'] == 'DEX08') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'DEX08') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        DEX08['Prefix'] = 'DEX08'
        DEX10 = pd.DataFrame()
        DEX10['Period'] = (melted.loc[(melted['Prefix'] == 'DEX10') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'DEX10') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        DEX10['Prefix'] = 'DEX10'
        VEH01 = pd.DataFrame()
        VEH01['Period'] = (melted.loc[(melted['Prefix'] == 'VEH01') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH01') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH01['Prefix'] = 'VEH01'
        VEH02 = pd.DataFrame()
        VEH02['Period'] = (melted.loc[(melted['Prefix'] == 'VEH02') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH02') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH02['Prefix'] = 'VEH02'
        VEH03 = pd.DataFrame()
        VEH03['Period'] = (melted.loc[(melted['Prefix'] == 'VEH03') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH03') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH03['Prefix'] = 'VEH03'
        VEH04 = pd.DataFrame()
        VEH04['Period'] = (melted.loc[(melted['Prefix'] == 'VEH04') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH04') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH04['Prefix'] = 'VEH04'
        VEH05 = pd.DataFrame()
        VEH05['Period'] = (melted.loc[(melted['Prefix'] == 'VEH05') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH05') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH05['Prefix'] = 'VEH05'
        VEH07 = pd.DataFrame()
        VEH07['Period'] = (melted.loc[(melted['Prefix'] == 'VEH07') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH07') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH07['Prefix'] = 'VEH07'
        VEH08 = pd.DataFrame()
        VEH08['Period'] = (melted.loc[(melted['Prefix'] == 'VEH08') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH08') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH08['Prefix'] = 'VEH08'
        VEH09 = pd.DataFrame()
        VEH09['Period'] = (melted.loc[(melted['Prefix'] == 'VEH09') & (melted['Bef_Aft'] == 'A'), 'Value'].reset_index() / melted.loc[(melted['Prefix'] == 'VEH09') & (melted['Bef_Aft'] == 'B'), 'Value'].reset_index()).drop('index', axis=1)
        VEH09['Prefix'] = 'VEH09'
        
        result = pd.concat([DEX01, DEX02, DEX06, DEX08, DEX10, VEH01, VEH02, VEH03, VEH04, VEH05, VEH07, VEH08, VEH09]).reset_index()
        
        
        # to be continued ... maybe split again to get DEX VEH
        
        
        
        
        
        # Combine 'Prefix' and 'Period' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01') | (melted['Prefix'] == 'SCN02') | (melted['Prefix'] == 'SCN03') | (melted['Prefix'] == 'SCN04') | (melted['Prefix'] == 'SCN05') | (melted['Prefix'] == 'SCN06')), 'Prefix'] = 'LL M'
        melted.loc[((melted['Prefix'] == 'SCN07') | (melted['Prefix'] == 'SCN08') | (melted['Prefix'] == 'SCN09') | (melted['Prefix'] == 'SCN10') | (melted['Prefix'] == 'SCN11') | (melted['Prefix'] == 'SCN12')), 'Prefix'] = 'LL F'
        melted.loc[((melted['Prefix'] == 'SCN13') | (melted['Prefix'] == 'SCN14') | (melted['Prefix'] == 'SCN15') | (melted['Prefix'] == 'SCN16') | (melted['Prefix'] == 'SCN17') | (melted['Prefix'] == 'SCN18')), 'Prefix'] = 'LD M'
        melted.loc[((melted['Prefix'] == 'SCN19') | (melted['Prefix'] == 'SCN20') | (melted['Prefix'] == 'SCN21') | (melted['Prefix'] == 'SCN22') | (melted['Prefix'] == 'SCN23') | (melted['Prefix'] == 'SCN24')), 'Prefix'] = 'LD F'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Period COMBINE.csv')
        KW_BoxenPlot4(data=df, title= 'Period of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Period')
        KW_BoxenPlot4(data=df, title= 'Period of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Period')
        
        melted.loc[melted['Prefix'] == 'LL M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LL F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LD M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LD F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LL M', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LL F', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LD M', 'Regime'] = 'LD'
        melted.loc[melted['Prefix'] == 'LD F', 'Regime'] = 'LD'        
        KW_BoxenPlot_hue(data=melted, title= 'Period of all cells', path=path, x='Regime', y='Value', hue='Sex', ylim=(None, None), kind='violin', name='Period') 
        

      
        # FOR Katya LL vs Control Male vs Female SCN experiment
        """
        df3 = df2[['Period\SCN01', 'Period\SCN02', 'Period\SCN03', 'Period\SCN04', 'Period\SCN05', 'Period\SCN06', 'Period\SCN07', 'Period\SCN08', 'Period\SCN09', 'Period\SCN10', 'Period\SCN11', 'Period\SCN12',
                   'Period\SCN13', 'Period\SCN14', 'Period\SCN15', 'Period\SCN16', 'Period\SCN17', 'Period\SCN18', 'Period\SCN19', 'Period\SCN20', 'Period\SCN21', 'Period\SCN22', 'Period\SCN23', 'Period\SCN24']]
        # Rename columns
        df3.columns = ['Period_SCN01', 'Period_SCN02', 'Period_SCN03', 'Period_SCN04', 'Period_SCN05', 'Period_SCN06', 'Period_SCN07', 'Period_SCN08', 'Period_SCN09', 'Period_SCN10', 'Period_SCN11', 'Period_SCN12',
                       'Period_SCN13', 'Period_SCN14', 'Period_SCN15', 'Period_SCN16', 'Period_SCN17', 'Period_SCN18', 'Period_SCN19', 'Period_SCN20', 'Period_SCN21', 'Period_SCN22', 'Period_SCN23', 'Period_SCN24']        
        
        melted = df3.melt(value_vars=['Period_SCN01', 'Period_SCN02', 'Period_SCN03', 'Period_SCN04', 'Period_SCN05', 'Period_SCN06', 'Period_SCN07', 'Period_SCN08', 'Period_SCN09', 'Period_SCN10', 'Period_SCN11', 'Period_SCN12',
                       'Period_SCN13', 'Period_SCN14', 'Period_SCN15', 'Period_SCN16', 'Period_SCN17', 'Period_SCN18', 'Period_SCN19', 'Period_SCN20', 'Period_SCN21', 'Period_SCN22', 'Period_SCN23', 'Period_SCN24'],
                          var_name='Var', value_name='Value')              
        # Extract the 'SCN' and period numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Period'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Period' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01') | (melted['Prefix'] == 'SCN02') | (melted['Prefix'] == 'SCN03') | (melted['Prefix'] == 'SCN04') | (melted['Prefix'] == 'SCN05') | (melted['Prefix'] == 'SCN06')), 'Prefix'] = 'LL M'
        melted.loc[((melted['Prefix'] == 'SCN07') | (melted['Prefix'] == 'SCN08') | (melted['Prefix'] == 'SCN09') | (melted['Prefix'] == 'SCN10') | (melted['Prefix'] == 'SCN11') | (melted['Prefix'] == 'SCN12')), 'Prefix'] = 'LL F'
        melted.loc[((melted['Prefix'] == 'SCN13') | (melted['Prefix'] == 'SCN14') | (melted['Prefix'] == 'SCN15') | (melted['Prefix'] == 'SCN16') | (melted['Prefix'] == 'SCN17') | (melted['Prefix'] == 'SCN18')), 'Prefix'] = 'LD M'
        melted.loc[((melted['Prefix'] == 'SCN19') | (melted['Prefix'] == 'SCN20') | (melted['Prefix'] == 'SCN21') | (melted['Prefix'] == 'SCN22') | (melted['Prefix'] == 'SCN23') | (melted['Prefix'] == 'SCN24')), 'Prefix'] = 'LD F'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Period COMBINE.csv')
        KW_BoxenPlot4(data=df, title= 'Period of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Period')
        KW_BoxenPlot4(data=df, title= 'Period of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Period')
        
        melted.loc[melted['Prefix'] == 'LL M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LL F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LD M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LD F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LL M', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LL F', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LD M', 'Regime'] = 'LD'
        melted.loc[melted['Prefix'] == 'LD F', 'Regime'] = 'LD'        
        KW_BoxenPlot_hue(data=melted, title= 'Period of all cells', path=path, x='Regime', y='Value', hue='Sex', ylim=(None, None), kind='violin', name='Period') 
        
        df3 = df2[['Amplitude\SCN01', 'Amplitude\SCN02', 'Amplitude\SCN03', 'Amplitude\SCN04', 'Amplitude\SCN05', 'Amplitude\SCN06', 'Amplitude\SCN07', 'Amplitude\SCN08', 'Amplitude\SCN09', 'Amplitude\SCN10', 'Amplitude\SCN11', 'Amplitude\SCN12',
                   'Amplitude\SCN13', 'Amplitude\SCN14', 'Amplitude\SCN15', 'Amplitude\SCN16', 'Amplitude\SCN17', 'Amplitude\SCN18', 'Amplitude\SCN19', 'Amplitude\SCN20', 'Amplitude\SCN21', 'Amplitude\SCN22', 'Amplitude\SCN23', 'Amplitude\SCN24']]
        # Rename columns
        df3.columns = ['Amplitude_SCN01', 'Amplitude_SCN02', 'Amplitude_SCN03', 'Amplitude_SCN04', 'Amplitude_SCN05', 'Amplitude_SCN06', 'Amplitude_SCN07', 'Amplitude_SCN08', 'Amplitude_SCN09', 'Amplitude_SCN10', 'Amplitude_SCN11', 'Amplitude_SCN12',
                       'Amplitude_SCN13', 'Amplitude_SCN14', 'Amplitude_SCN15', 'Amplitude_SCN16', 'Amplitude_SCN17', 'Amplitude_SCN18', 'Amplitude_SCN19', 'Amplitude_SCN20', 'Amplitude_SCN21', 'Amplitude_SCN22', 'Amplitude_SCN23', 'Amplitude_SCN24']        
        
        melted = df3.melt(value_vars=['Amplitude_SCN01', 'Amplitude_SCN02', 'Amplitude_SCN03', 'Amplitude_SCN04', 'Amplitude_SCN05', 'Amplitude_SCN06', 'Amplitude_SCN07', 'Amplitude_SCN08', 'Amplitude_SCN09', 'Amplitude_SCN10', 'Amplitude_SCN11', 'Amplitude_SCN12',
                       'Amplitude_SCN13', 'Amplitude_SCN14', 'Amplitude_SCN15', 'Amplitude_SCN16', 'Amplitude_SCN17', 'Amplitude_SCN18', 'Amplitude_SCN19', 'Amplitude_SCN20', 'Amplitude_SCN21', 'Amplitude_SCN22', 'Amplitude_SCN23', 'Amplitude_SCN24'],
                          var_name='Var', value_name='Value')              
        # Extract the 'SCN' and Amplitude numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amplitude'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Amplitude' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01') | (melted['Prefix'] == 'SCN02') | (melted['Prefix'] == 'SCN03') | (melted['Prefix'] == 'SCN04') | (melted['Prefix'] == 'SCN05') | (melted['Prefix'] == 'SCN06')), 'Prefix'] = 'LL M'
        melted.loc[((melted['Prefix'] == 'SCN07') | (melted['Prefix'] == 'SCN08') | (melted['Prefix'] == 'SCN09') | (melted['Prefix'] == 'SCN10') | (melted['Prefix'] == 'SCN11') | (melted['Prefix'] == 'SCN12')), 'Prefix'] = 'LL F'
        melted.loc[((melted['Prefix'] == 'SCN13') | (melted['Prefix'] == 'SCN14') | (melted['Prefix'] == 'SCN15') | (melted['Prefix'] == 'SCN16') | (melted['Prefix'] == 'SCN17') | (melted['Prefix'] == 'SCN18')), 'Prefix'] = 'LD M'
        melted.loc[((melted['Prefix'] == 'SCN19') | (melted['Prefix'] == 'SCN20') | (melted['Prefix'] == 'SCN21') | (melted['Prefix'] == 'SCN22') | (melted['Prefix'] == 'SCN23') | (melted['Prefix'] == 'SCN24')), 'Prefix'] = 'LD F'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Amplitude COMBINE.csv')
        KW_BoxenPlot4(data=df, title= 'Amplitude of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Amplitude')
        KW_BoxenPlot4(data=df, title= 'Amplitude of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Amplitude')
        
        melted.loc[melted['Prefix'] == 'LL M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LL F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LD M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LD F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LL M', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LL F', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LD M', 'Regime'] = 'LD'
        melted.loc[melted['Prefix'] == 'LD F', 'Regime'] = 'LD'        
        KW_BoxenPlot_hue(data=melted, title= 'Amplitude of all cells', path=path, x='Regime', y='Value', hue='Sex', ylim=(None, None), kind='violin', name='Amplitude') 
        
        df3 = df2[['Trend\SCN01', 'Trend\SCN02', 'Trend\SCN03', 'Trend\SCN04', 'Trend\SCN05', 'Trend\SCN06', 'Trend\SCN07', 'Trend\SCN08', 'Trend\SCN09', 'Trend\SCN10', 'Trend\SCN11', 'Trend\SCN12',
                   'Trend\SCN13', 'Trend\SCN14', 'Trend\SCN15', 'Trend\SCN16', 'Trend\SCN17', 'Trend\SCN18', 'Trend\SCN19', 'Trend\SCN20', 'Trend\SCN21', 'Trend\SCN22', 'Trend\SCN23', 'Trend\SCN24']]
        # Rename columns
        df3.columns = ['Trend_SCN01', 'Trend_SCN02', 'Trend_SCN03', 'Trend_SCN04', 'Trend_SCN05', 'Trend_SCN06', 'Trend_SCN07', 'Trend_SCN08', 'Trend_SCN09', 'Trend_SCN10', 'Trend_SCN11', 'Trend_SCN12',
                       'Trend_SCN13', 'Trend_SCN14', 'Trend_SCN15', 'Trend_SCN16', 'Trend_SCN17', 'Trend_SCN18', 'Trend_SCN19', 'Trend_SCN20', 'Trend_SCN21', 'Trend_SCN22', 'Trend_SCN23', 'Trend_SCN24']        
        
        melted = df3.melt(value_vars=['Trend_SCN01', 'Trend_SCN02', 'Trend_SCN03', 'Trend_SCN04', 'Trend_SCN05', 'Trend_SCN06', 'Trend_SCN07', 'Trend_SCN08', 'Trend_SCN09', 'Trend_SCN10', 'Trend_SCN11', 'Trend_SCN12',
                       'Trend_SCN13', 'Trend_SCN14', 'Trend_SCN15', 'Trend_SCN16', 'Trend_SCN17', 'Trend_SCN18', 'Trend_SCN19', 'Trend_SCN20', 'Trend_SCN21', 'Trend_SCN22', 'Trend_SCN23', 'Trend_SCN24'],
                          var_name='Var', value_name='Value')              
        # Extract the 'SCN' and Trend numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Trend'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Trend' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01') | (melted['Prefix'] == 'SCN02') | (melted['Prefix'] == 'SCN03') | (melted['Prefix'] == 'SCN04') | (melted['Prefix'] == 'SCN05') | (melted['Prefix'] == 'SCN06')), 'Prefix'] = 'LL M'
        melted.loc[((melted['Prefix'] == 'SCN07') | (melted['Prefix'] == 'SCN08') | (melted['Prefix'] == 'SCN09') | (melted['Prefix'] == 'SCN10') | (melted['Prefix'] == 'SCN11') | (melted['Prefix'] == 'SCN12')), 'Prefix'] = 'LL F'
        melted.loc[((melted['Prefix'] == 'SCN13') | (melted['Prefix'] == 'SCN14') | (melted['Prefix'] == 'SCN15') | (melted['Prefix'] == 'SCN16') | (melted['Prefix'] == 'SCN17') | (melted['Prefix'] == 'SCN18')), 'Prefix'] = 'LD M'
        melted.loc[((melted['Prefix'] == 'SCN19') | (melted['Prefix'] == 'SCN20') | (melted['Prefix'] == 'SCN21') | (melted['Prefix'] == 'SCN22') | (melted['Prefix'] == 'SCN23') | (melted['Prefix'] == 'SCN24')), 'Prefix'] = 'LD F'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Trend COMBINE.csv')
        KW_BoxenPlot4(data=df, title= 'Trend of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Trend')
        KW_BoxenPlot4(data=df, title= 'Trend of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Trend')
        
        melted.loc[melted['Prefix'] == 'LL M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LL F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LD M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LD F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LL M', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LL F', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LD M', 'Regime'] = 'LD'
        melted.loc[melted['Prefix'] == 'LD F', 'Regime'] = 'LD'        
        KW_BoxenPlot_hue(data=melted, title= 'Trend of all cells', path=path, x='Regime', y='Value', hue='Sex', ylim=(None, None), kind='violin', name='Trend') 
        
        df3 = df2[['Decay\SCN01', 'Decay\SCN02', 'Decay\SCN03', 'Decay\SCN04', 'Decay\SCN05', 'Decay\SCN06', 'Decay\SCN07', 'Decay\SCN08', 'Decay\SCN09', 'Decay\SCN10', 'Decay\SCN11', 'Decay\SCN12',
                   'Decay\SCN13', 'Decay\SCN14', 'Decay\SCN15', 'Decay\SCN16', 'Decay\SCN17', 'Decay\SCN18', 'Decay\SCN19', 'Decay\SCN20', 'Decay\SCN21', 'Decay\SCN22', 'Decay\SCN23', 'Decay\SCN24']]
        # Rename columns
        df3.columns = ['Decay_SCN01', 'Decay_SCN02', 'Decay_SCN03', 'Decay_SCN04', 'Decay_SCN05', 'Decay_SCN06', 'Decay_SCN07', 'Decay_SCN08', 'Decay_SCN09', 'Decay_SCN10', 'Decay_SCN11', 'Decay_SCN12',
                       'Decay_SCN13', 'Decay_SCN14', 'Decay_SCN15', 'Decay_SCN16', 'Decay_SCN17', 'Decay_SCN18', 'Decay_SCN19', 'Decay_SCN20', 'Decay_SCN21', 'Decay_SCN22', 'Decay_SCN23', 'Decay_SCN24']        
        
        melted = df3.melt(value_vars=['Decay_SCN01', 'Decay_SCN02', 'Decay_SCN03', 'Decay_SCN04', 'Decay_SCN05', 'Decay_SCN06', 'Decay_SCN07', 'Decay_SCN08', 'Decay_SCN09', 'Decay_SCN10', 'Decay_SCN11', 'Decay_SCN12',
                       'Decay_SCN13', 'Decay_SCN14', 'Decay_SCN15', 'Decay_SCN16', 'Decay_SCN17', 'Decay_SCN18', 'Decay_SCN19', 'Decay_SCN20', 'Decay_SCN21', 'Decay_SCN22', 'Decay_SCN23', 'Decay_SCN24'],
                          var_name='Var', value_name='Value')              
        # Extract the 'SCN' and Decay numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Decay'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Decay' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01') | (melted['Prefix'] == 'SCN02') | (melted['Prefix'] == 'SCN03') | (melted['Prefix'] == 'SCN04') | (melted['Prefix'] == 'SCN05') | (melted['Prefix'] == 'SCN06')), 'Prefix'] = 'LL M'
        melted.loc[((melted['Prefix'] == 'SCN07') | (melted['Prefix'] == 'SCN08') | (melted['Prefix'] == 'SCN09') | (melted['Prefix'] == 'SCN10') | (melted['Prefix'] == 'SCN11') | (melted['Prefix'] == 'SCN12')), 'Prefix'] = 'LL F'
        melted.loc[((melted['Prefix'] == 'SCN13') | (melted['Prefix'] == 'SCN14') | (melted['Prefix'] == 'SCN15') | (melted['Prefix'] == 'SCN16') | (melted['Prefix'] == 'SCN17') | (melted['Prefix'] == 'SCN18')), 'Prefix'] = 'LD M'
        melted.loc[((melted['Prefix'] == 'SCN19') | (melted['Prefix'] == 'SCN20') | (melted['Prefix'] == 'SCN21') | (melted['Prefix'] == 'SCN22') | (melted['Prefix'] == 'SCN23') | (melted['Prefix'] == 'SCN24')), 'Prefix'] = 'LD F'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Decay COMBINE.csv')
        KW_BoxenPlot4(data=df, title= 'Decay of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Decay')
        KW_BoxenPlot4(data=df, title= 'Decay of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Decay')
        
        melted.loc[melted['Prefix'] == 'LL M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LL F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LD M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LD F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LL M', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LL F', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LD M', 'Regime'] = 'LD'
        melted.loc[melted['Prefix'] == 'LD F', 'Regime'] = 'LD'        
        KW_BoxenPlot_hue(data=melted, title= 'Decay of all cells', path=path, x='Regime', y='Value', hue='Sex', ylim=(None, None), kind='violin', name='Decay') 
        
        df3 = df2[['Rsq\SCN01', 'Rsq\SCN02', 'Rsq\SCN03', 'Rsq\SCN04', 'Rsq\SCN05', 'Rsq\SCN06', 'Rsq\SCN07', 'Rsq\SCN08', 'Rsq\SCN09', 'Rsq\SCN10', 'Rsq\SCN11', 'Rsq\SCN12',
                   'Rsq\SCN13', 'Rsq\SCN14', 'Rsq\SCN15', 'Rsq\SCN16', 'Rsq\SCN17', 'Rsq\SCN18', 'Rsq\SCN19', 'Rsq\SCN20', 'Rsq\SCN21', 'Rsq\SCN22', 'Rsq\SCN23', 'Rsq\SCN24']]
        # Rename columns
        df3.columns = ['Rsq_SCN01', 'Rsq_SCN02', 'Rsq_SCN03', 'Rsq_SCN04', 'Rsq_SCN05', 'Rsq_SCN06', 'Rsq_SCN07', 'Rsq_SCN08', 'Rsq_SCN09', 'Rsq_SCN10', 'Rsq_SCN11', 'Rsq_SCN12',
                       'Rsq_SCN13', 'Rsq_SCN14', 'Rsq_SCN15', 'Rsq_SCN16', 'Rsq_SCN17', 'Rsq_SCN18', 'Rsq_SCN19', 'Rsq_SCN20', 'Rsq_SCN21', 'Rsq_SCN22', 'Rsq_SCN23', 'Rsq_SCN24']        
        
        melted = df3.melt(value_vars=['Rsq_SCN01', 'Rsq_SCN02', 'Rsq_SCN03', 'Rsq_SCN04', 'Rsq_SCN05', 'Rsq_SCN06', 'Rsq_SCN07', 'Rsq_SCN08', 'Rsq_SCN09', 'Rsq_SCN10', 'Rsq_SCN11', 'Rsq_SCN12',
                       'Rsq_SCN13', 'Rsq_SCN14', 'Rsq_SCN15', 'Rsq_SCN16', 'Rsq_SCN17', 'Rsq_SCN18', 'Rsq_SCN19', 'Rsq_SCN20', 'Rsq_SCN21', 'Rsq_SCN22', 'Rsq_SCN23', 'Rsq_SCN24'],
                          var_name='Var', value_name='Value')              
        # Extract the 'SCN' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01') | (melted['Prefix'] == 'SCN02') | (melted['Prefix'] == 'SCN03') | (melted['Prefix'] == 'SCN04') | (melted['Prefix'] == 'SCN05') | (melted['Prefix'] == 'SCN06')), 'Prefix'] = 'LL M'
        melted.loc[((melted['Prefix'] == 'SCN07') | (melted['Prefix'] == 'SCN08') | (melted['Prefix'] == 'SCN09') | (melted['Prefix'] == 'SCN10') | (melted['Prefix'] == 'SCN11') | (melted['Prefix'] == 'SCN12')), 'Prefix'] = 'LL F'
        melted.loc[((melted['Prefix'] == 'SCN13') | (melted['Prefix'] == 'SCN14') | (melted['Prefix'] == 'SCN15') | (melted['Prefix'] == 'SCN16') | (melted['Prefix'] == 'SCN17') | (melted['Prefix'] == 'SCN18')), 'Prefix'] = 'LD M'
        melted.loc[((melted['Prefix'] == 'SCN19') | (melted['Prefix'] == 'SCN20') | (melted['Prefix'] == 'SCN21') | (melted['Prefix'] == 'SCN22') | (melted['Prefix'] == 'SCN23') | (melted['Prefix'] == 'SCN24')), 'Prefix'] = 'LD F'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Rsq COMBINE.csv')
        KW_BoxenPlot4(data=df, title= 'Rsq of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Rsq')
        KW_BoxenPlot4(data=df, title= 'Rsq of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Rsq')
        
        melted.loc[melted['Prefix'] == 'LL M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LL F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LD M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LD F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LL M', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LL F', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LD M', 'Regime'] = 'LD'
        melted.loc[melted['Prefix'] == 'LD F', 'Regime'] = 'LD'        
        KW_BoxenPlot_hue(data=melted, title= 'Rsq of all cells', path=path, x='Regime', y='Value', hue='Sex', ylim=(None, None), kind='violin', name='Rsq') 
        
        df3 = df2[['Phase\SCN01', 'Phase\SCN02', 'Phase\SCN03', 'Phase\SCN04', 'Phase\SCN05', 'Phase\SCN06', 'Phase\SCN07', 'Phase\SCN08', 'Phase\SCN09', 'Phase\SCN10', 'Phase\SCN11', 'Phase\SCN12',
                   'Phase\SCN13', 'Phase\SCN14', 'Phase\SCN15', 'Phase\SCN16', 'Phase\SCN17', 'Phase\SCN18', 'Phase\SCN19', 'Phase\SCN20', 'Phase\SCN21', 'Phase\SCN22', 'Phase\SCN23', 'Phase\SCN24']]
        # Rename columns
        df3.columns = ['Phase_SCN01', 'Phase_SCN02', 'Phase_SCN03', 'Phase_SCN04', 'Phase_SCN05', 'Phase_SCN06', 'Phase_SCN07', 'Phase_SCN08', 'Phase_SCN09', 'Phase_SCN10', 'Phase_SCN11', 'Phase_SCN12',
                       'Phase_SCN13', 'Phase_SCN14', 'Phase_SCN15', 'Phase_SCN16', 'Phase_SCN17', 'Phase_SCN18', 'Phase_SCN19', 'Phase_SCN20', 'Phase_SCN21', 'Phase_SCN22', 'Phase_SCN23', 'Phase_SCN24']        
        
        melted = df3.melt(value_vars=['Phase_SCN01', 'Phase_SCN02', 'Phase_SCN03', 'Phase_SCN04', 'Phase_SCN05', 'Phase_SCN06', 'Phase_SCN07', 'Phase_SCN08', 'Phase_SCN09', 'Phase_SCN10', 'Phase_SCN11', 'Phase_SCN12',
                       'Phase_SCN13', 'Phase_SCN14', 'Phase_SCN15', 'Phase_SCN16', 'Phase_SCN17', 'Phase_SCN18', 'Phase_SCN19', 'Phase_SCN20', 'Phase_SCN21', 'Phase_SCN22', 'Phase_SCN23', 'Phase_SCN24'],
                          var_name='Var', value_name='Value')              
        # Extract the 'SCN' and Phase numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Phase'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Phase' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01') | (melted['Prefix'] == 'SCN02') | (melted['Prefix'] == 'SCN03') | (melted['Prefix'] == 'SCN04') | (melted['Prefix'] == 'SCN05') | (melted['Prefix'] == 'SCN06')), 'Prefix'] = 'LL M'
        melted.loc[((melted['Prefix'] == 'SCN07') | (melted['Prefix'] == 'SCN08') | (melted['Prefix'] == 'SCN09') | (melted['Prefix'] == 'SCN10') | (melted['Prefix'] == 'SCN11') | (melted['Prefix'] == 'SCN12')), 'Prefix'] = 'LL F'
        melted.loc[((melted['Prefix'] == 'SCN13') | (melted['Prefix'] == 'SCN14') | (melted['Prefix'] == 'SCN15') | (melted['Prefix'] == 'SCN16') | (melted['Prefix'] == 'SCN17') | (melted['Prefix'] == 'SCN18')), 'Prefix'] = 'LD M'
        melted.loc[((melted['Prefix'] == 'SCN19') | (melted['Prefix'] == 'SCN20') | (melted['Prefix'] == 'SCN21') | (melted['Prefix'] == 'SCN22') | (melted['Prefix'] == 'SCN23') | (melted['Prefix'] == 'SCN24')), 'Prefix'] = 'LD F'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Phase COMBINE.csv')
        KW_BoxenPlot4(data=df, title= 'Phase of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Phase')
        KW_BoxenPlot4(data=df, title= 'Phase of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Phase')
       
        melted.loc[melted['Prefix'] == 'LL M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LL F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LD M', 'Sex'] = 'M'
        melted.loc[melted['Prefix'] == 'LD F', 'Sex'] = 'F'
        melted.loc[melted['Prefix'] == 'LL M', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LL F', 'Regime'] = 'LL'
        melted.loc[melted['Prefix'] == 'LD M', 'Regime'] = 'LD'
        melted.loc[melted['Prefix'] == 'LD F', 'Regime'] = 'LD'        
        KW_BoxenPlot_hue(data=melted, title= 'Phase of all cells', path=path, x='Regime', y='Value', hue='Sex', ylim=(None, None), kind='violin', name='Phase') 
        
        
        # PCA - table pivoting and plot
        from statsmodels.multivariate.pca import PCA
        
        def PCA_plot(data, columns, hue, name, mydir, pca_model, pc='PC1', annotate_scatter=False):

            # dta = z_score(data[columns])  # values for PCA should be standardized/normalized, 
            # pca_model = PCA(dta.dropna().T, standardize=False, demean=True)

            ## To save as vector svg with fonts editable in Corel ###
            import matplotlib as mpl                                                                       #import matplotlib as mpl
            new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
            mpl.rcParams.update(new_rc_params)
            
            # scree plot to show PC scores
            pca_model.plot_scree(log_scale=False)
            plt.savefig(f'{mydir}' + '\\' + f'PCA_scree_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
            plt.savefig(f'{mydir}' + '\\' + f'PCA_scree_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
            plt.clf()
            plt.close()
            
            # scatter plot of PC1 x PC2
            data['PC1'] = pca_model.loadings.iloc[:, 0] # pca_model.loadings['comp_0'] >> or 'comp_01'
            data['PC2'] = pca_model.loadings.iloc[:, 1] # pca_model.loadings['comp_1']
            plt.figure(figsize=(5, 5))
            sns.scatterplot(x="PC1", y="PC2", data=data, hue=hue, s=30)
            
            # add annotation from 2nd , i is 1st col as index, j is dataframe starting from 2nd col
            if annotate_scatter is True:
                for i,j in data.iterrows():
                    plt.annotate(j[0], xy=(j['PC1'], j['PC2']), fontsize=7)            
            
            plt.savefig(f'{mydir}' + '\\' + f'PCA_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
            plt.savefig(f'{mydir}' + '\\' + f'PCA_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
            plt.show()
            plt.clf()
            plt.close()

            # histogram of selected PC by hue
            sns.histplot(data=data, x=pc, hue=hue, kde=True, stat="density") 
            ###### Calculate p values between hue for separate categories in pc ######
            by_hue = data.groupby(data[hue])  # for ANOVA and labels, auto-create col_order
            hue_order = []
            for a, frame in by_hue:
                hue_order.append(a) 
            ##### ANOVA ######
            alist = []
            for i in range(len(hue_order)):
                alist.append(data[pc][(data[hue] == hue_order[i])].dropna(how='any'))
            F, p = stats.f_oneway(*alist) # asterisk is for *args - common idiom to allow arbitrary number of arguments to functions    
            if p < 0.0000000001:
                plt.annotate('ANOVA p < 1e-10', xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                            textcoords='offset points', horizontalalignment='right', verticalalignment='top')
            else:
                plt.annotate('ANOVA p = ' + str(round(p, 8)), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                            textcoords='offset points', horizontalalignment='right', verticalalignment='top')    
            plt.savefig(f'{mydir}' + '\\' + f'PCA_histo_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
            plt.savefig(f'{mydir}' + '\\' + f'PCA_histo_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
            plt.clf()
            plt.close()
            
            # description in txt file
            description = open(f'{mydir}\\PCA_{hue}_{name}.txt', 'w')
            description.write(f'{columns} \nANOVA F = {F}, p = {p}')
            description.close()   
            
            return data['PC1'], data['PC2']
        
        # Melt the DataFrame 
        df_melted = df2.melt(var_name='Metric_Sample', value_name='Value') # id_vars='...', 
        df_stacked = df2.stack().reset_index()
        df_stacked.columns = ['Row', 'Metric_Sample', 'Value']
        df_stacked[['Metric', 'Sample']] = df_stacked['Metric_Sample'].str.split('\\', expand=True)
        df3 = df_stacked[['Row', 'Sample', 'Metric', 'Value']]
        
        # with rois, not used for PCA, 3d
        df3_pivoted = df3.pivot(index=['Row', 'Sample'],columns = ['Metric'], values = 'Value')    
        
        # Pivot to get mean values of each metric for each explant (for PCA)
        df_result = df3.pivot_table(index=['Sample'], columns='Metric', values='Value', aggfunc='mean').reset_index()    
        df_result.loc[0:11, 'Regime'] = 'LL'
        df_result.loc[12:23, 'Regime'] = 'LD'
        df_result.loc[0:5, 'Sex'] = 'M'
        df_result.loc[6:11, 'Sex'] = 'F'
        df_result.loc[12:17, 'Sex'] = 'M'
        df_result.loc[18:23, 'Sex'] = 'F'

        # for PCA model, table must be transposed, but not for PCA plot later        
        pca_model = PCA(df_result.iloc[:, 1:7].T.dropna(how='all'), standardize=False, demean=True)
        # but PCA plots need analytes in columns again with hue in columns as well
        PCA_plot(df_result, df_result.columns[1:7], hue = 'Regime', name = 'Regime', mydir = path, pca_model=pca_model, pc='PC1', annotate_scatter=True)
        PCA_plot(df_result, df_result.columns[1:7], hue = 'Sex', name = 'Sex', mydir = path, pca_model=pca_model, pc='PC1', annotate_scatter=True)      
        
        # for PCA model, table must be transposed, but not for PCA plot later        
        pca_model = PCA(df_result.iloc[:, 1:7].T.dropna(how='all'), standardize=True, demean=True)
        # but PCA plots need analytes in columns again with hue in columns as well
        PCA_plot(df_result, df_result.columns[1:7], hue = 'Regime', name = 'Regime Standardized', mydir = path, pca_model=pca_model, pc='PC1', annotate_scatter=True)
        PCA_plot(df_result, df_result.columns[1:7], hue = 'Sex', name = 'Sex Standardized', mydir = path, pca_model=pca_model, pc='PC1', annotate_scatter=True)   
        
        # useless here
        # from matplotlib_venn import venn2        
        # def Venn(left, right, middle, labels, mydir):  # venn2(subsets = (980, 1197, 679), set_labels = ('Group A', 'Group B'))
        #     sns.set_context("paper", font_scale=1.4) 
        #     venn2(subsets = (left, right, middle), set_labels = labels)
        #     plt.savefig(f'{mydir}' + '\\' + f'Venn {labels}.png', format = 'png', bbox_inches = 'tight')
        #     plt.savefig(f'{mydir}' + '\\' + f'Venn {labels}.svg', format = 'svg', bbox_inches = 'tight')
        #     plt.clf()
        #     plt.close()        
        # Venn(left=len(df2['Period\\SCN01'].dropna()), right=len(df2['Period\\SCN01']), middle=len(df2['Period\\SCN01'].dropna()), labels=('SCN 1 rhythmic', ' all'), mydir=path)
        
        
       
        
        # Complete pie of all explants, NEED TO BE FINISHED!!!!
        LD_rhytmic_total = 0
        LD_arrhytmic_total = 0
        LL_rhytmic_total = 0
        LL_arrhytmic_total = 0
        counter = 0
        for mydir in mydirlist:        
            explant = f'{mydir[-5:]}'
            counter += 1
            if counter <=12:
                LL_rhytmic_total += len(df2[f'Period\\{explant}'].dropna())
                LL_arrhytmic_total += len(df2[f'Period\\{explant}']) - len(df2[f'Period\\{explant}'].dropna())
            if counter >12:
                LD_rhytmic_total += len(df2[f'Period\\{explant}'].dropna())
                LD_arrhytmic_total += len(df2[f'Period\\{explant}']) - len(df2[f'Period\\{explant}'].dropna())  

        
        # piechart for all LL explants together
        data = [LL_rhytmic_total, LL_arrhytmic_total]
        keys = ['rhythmic Rsq>0.8', 'Rsq<0.8 and arrhythmic'] 
        plt.pie(data, labels=keys, autopct='%.0f%%') 
        plt.savefig(f'{path}' + '\\' + 'Pie LL.png', format = 'png', bbox_inches = 'tight')
        plt.savefig(f'{path}' + '\\' + 'Pie LL.svg', format = 'svg', bbox_inches = 'tight')
        plt.clf()
        plt.close()
        
        # piechart for all LL explants together
        data = [LD_rhytmic_total, LD_arrhytmic_total]
        keys = ['rhythmic Rsq>0.8', 'Rsq<0.8 and arrhythmic'] 
        plt.pie(data, labels=keys, autopct='%.0f%%') 
        plt.savefig(f'{path}' + '\\' + 'Pie LD.png', format = 'png', bbox_inches = 'tight')
        plt.savefig(f'{path}' + '\\' + 'Pie LD.svg', format = 'svg', bbox_inches = 'tight')
        plt.clf()
        plt.close()

        
        for mydir in mydirlist:
            data2 = pd.read_csv(glob.glob(f'{mydir}\\*oscillatory_params.csv')[0])  
            if Plot_All_Traces == 'select_rsq':
                data2 = data2[data2['Rsq'] > rsq_threshold] 
                
                # add to list LD_rhytmic_total number of rhythmic rois, etc.., then move on to second explant
                # use similar as above but count REAL nonrhythmic
                
                
                
                
            # if Plot_All_Traces == 'select_amp':
            #     data2 = data2[data2['Amplitude'] > amp_threshold]
            # if Plot_All_Traces == 'select_trend':
            #     data2 = data2[data2['Trend'] > trend_threshold]             
            # if Plot_All_Traces == 'select_decay':
            #     data2 = data2[data2['Decay'] > decay_threshold]    
            df2['Period' + str(mydir[name:nameend])] = data2['Period']  
        
        """
        
        
        # FOR ALENA SRBR TALK - Dex vs Veh treated E17 SCN
        """
        # # all explants
        # df3 = df2[['AmplitudeSCN01L\\after', 'AmplitudeCN01L\\before',
        # 'AmplitudeSCN01R\\after', 'AmplitudeCN01R\\before',
        # 'AmplitudeSCN02L\\after', 'AmplitudeCN02L\\before',
        # 'AmplitudeSCN02R\\after', 'AmplitudeCN02R\\before',
        # 'AmplitudeSCN03L\\after', 'AmplitudeCN03L\\before',
        # 'AmplitudeSCN03R\\after', 'AmplitudeCN03R\\before',
        # 'AmplitudeSCN04L\\after', 'AmplitudeSCN04L\\befor',
        # 'AmplitudeSCN04R\\after', 'AmplitudeSCN04R\\befor',
        # 'AmplitudeSCN05L\\after', 'AmplitudeSCN05L\\befor',
        # 'AmplitudeSCN05R\\after', 'AmplitudeSCN05R\\befor',
        # 'Amplitudel\\SCN06L\\aft', 'Amplitudel\\SCN06L\\bef',
        # 'Amplitudel\\SCN06R\\aft', 'Amplitudel\\SCN06R\\bef',
        # 'Amplitudel\\SCN07L\\aft', 'Amplitudel\\SCN07L\\bef',
        # 'Amplitudel\\SCN07R\\aft', 'Amplitudel\\SCN07R\\bef',
        # 'Amplitudel\\SCN08L\\aft', 'Amplitudel\\SCN08L\\bef',
        # 'Amplitudel\\SCN08R\\aft', 'Amplitudel\\SCN08R\\bef',
        # 'Amplitudel\\SCN09L\\aft', 'Amplitudel\\SCN09L\\bef',
        # 'Amplitudel\\SCN09R\\aft', 'Amplitudel\\SCN09R\\bef',
        # 'Amplitudel\\SCN10L\\aft', 'Amplitudel\\SCN10L\\bef',
        # 'Amplitudel\\SCN10R\\aft', 'Amplitudel\\SCN10R\\bef',
        # 'Amplitudel\\SCN11L\\aft', 'Amplitudel\\SCN11L\\bef',
        # 'Amplitudel\\SCN11R\\aft', 'Amplitudel\\SCN11R\\bef',
        # 'Amplitudel\\SCN12L\\aft', 'Amplitudel\\SCN12L\\bef',
        # 'Amplitudel\\SCN12R\\aft', 'Amplitudel\\SCN12R\\bef',
        # 'Amplitudel\\SCN13L\\aft', 'Amplitudel\\SCN13L\\bef',
        # 'Amplitudel\\SCN13R\\aft', 'Amplitudel\\SCN13R\\bef',
        # 'Amplitudel\\SCN14L\\aft', 'Amplitudel\\SCN14L\\bef',
        # 'Amplitudel\\SCN14R\\aft', 'Amplitudel\\SCN14R\\bef']]
        # # Rename columns
        # df3.columns = ['Amplitude_SCN01L_aft', 'Amplitude_SCN01L_bef',
        # 'Amplitude_SCN01R_aft', 'Amplitude_SCN01R_bef',
        # 'Amplitude_SCN02L_aft', 'Amplitude_SCN02L_bef',
        # 'Amplitude_SCN02R_aft', 'Amplitude_SCN02R_bef',
        # 'Amplitude_SCN03L_aft', 'Amplitude_SCN03L_bef',
        # 'Amplitude_SCN03R_aft', 'Amplitude_SCN03R_bef',
        # 'Amplitude_SCN04L_aft', 'Amplitude_SCN04L_bef',
        # 'Amplitude_SCN04R_aft', 'Amplitude_SCN04R_bef',
        # 'Amplitude_SCN05L_aft', 'Amplitude_SCN05L_bef',
        # 'Amplitude_SCN05R_aft', 'Amplitude_SCN05R_bef',
        # 'Amplitude_SCN06L_aft', 'Amplitude_SCN06L_bef',
        # 'Amplitude_SCN06R_aft', 'Amplitude_SCN06R_bef',
        # 'Amplitude_SCN07L_aft', 'Amplitude_SCN07L_bef',
        # 'Amplitude_SCN07R_aft', 'Amplitude_SCN07R_bef',
        # 'Amplitude_SCN08L_aft', 'Amplitude_SCN08L_bef',
        # 'Amplitude_SCN08R_aft', 'Amplitude_SCN08R_bef',
        # 'Amplitude_SCN09L_aft', 'Amplitude_SCN09L_bef',
        # 'Amplitude_SCN09R_aft', 'Amplitude_SCN09R_bef',
        # 'Amplitude_SCN10L_aft', 'Amplitude_SCN10L_bef',
        # 'Amplitude_SCN10R_aft', 'Amplitude_SCN10R_bef',
        # 'Amplitude_SCN11L_aft', 'Amplitude_SCN11L_bef',
        # 'Amplitude_SCN11R_aft', 'Amplitude_SCN11R_bef',
        # 'Amplitude_SCN12L_aft', 'Amplitude_SCN12L_bef',
        # 'Amplitude_SCN12R_aft', 'Amplitude_SCN12R_bef',
        # 'Amplitude_SCN13L_aft', 'Amplitude_SCN13L_bef',
        # 'Amplitude_SCN13R_aft', 'Amplitude_SCN13R_bef',
        # 'Amplitude_SCN14L_aft', 'Amplitude_SCN14L_bef',
        # 'Amplitude_SCN14R_aft', 'Amplitude_SCN14R_bef']      
        # melted = df3.melt(value_vars=['Amplitude_SCN01L_aft', 'Amplitude_SCN01L_bef',
        # 'Amplitude_SCN01R_aft', 'Amplitude_SCN01R_bef',
        # 'Amplitude_SCN02L_aft', 'Amplitude_SCN02L_bef',
        # 'Amplitude_SCN02R_aft', 'Amplitude_SCN02R_bef',
        # 'Amplitude_SCN03L_aft', 'Amplitude_SCN03L_bef',
        # 'Amplitude_SCN03R_aft', 'Amplitude_SCN03R_bef',
        # 'Amplitude_SCN04L_aft', 'Amplitude_SCN04L_bef',
        # 'Amplitude_SCN04R_aft', 'Amplitude_SCN04R_bef',
        # 'Amplitude_SCN05L_aft', 'Amplitude_SCN05L_bef',
        # 'Amplitude_SCN05R_aft', 'Amplitude_SCN05R_bef',
        # 'Amplitude_SCN06L_aft', 'Amplitude_SCN06L_bef',
        # 'Amplitude_SCN06R_aft', 'Amplitude_SCN06R_bef',
        # 'Amplitude_SCN07L_aft', 'Amplitude_SCN07L_bef',
        # 'Amplitude_SCN07R_aft', 'Amplitude_SCN07R_bef',
        # 'Amplitude_SCN08L_aft', 'Amplitude_SCN08L_bef',
        # 'Amplitude_SCN08R_aft', 'Amplitude_SCN08R_bef',
        # 'Amplitude_SCN09L_aft', 'Amplitude_SCN09L_bef',
        # 'Amplitude_SCN09R_aft', 'Amplitude_SCN09R_bef',
        # 'Amplitude_SCN10L_aft', 'Amplitude_SCN10L_bef',
        # 'Amplitude_SCN10R_aft', 'Amplitude_SCN10R_bef',
        # 'Amplitude_SCN11L_aft', 'Amplitude_SCN11L_bef',
        # 'Amplitude_SCN11R_aft', 'Amplitude_SCN11R_bef',
        # 'Amplitude_SCN12L_aft', 'Amplitude_SCN12L_bef',
        # 'Amplitude_SCN12R_aft', 'Amplitude_SCN12R_bef',
        # 'Amplitude_SCN13L_aft', 'Amplitude_SCN13L_bef',
        # 'Amplitude_SCN13R_aft', 'Amplitude_SCN13R_bef',
        # 'Amplitude_SCN14L_aft', 'Amplitude_SCN14L_bef',
        # 'Amplitude_SCN14R_aft', 'Amplitude_SCN14R_bef'],
        #                   var_name='Var', value_name='Value')              
        # # Extract the 'CHP' and Rsq numbers from the 'Var' column
        # melted['Prefix'] = melted['Var'].str.split('_').str[1]
        # melted['Amp'] = melted['Var'].str.split('_').str[0]
        
        
        # after explants
        df3 = df2[['AmplitudeSCN01L\\after', 
        'AmplitudeSCN01R\\after', 
        'AmplitudeSCN02L\\after', 
        'AmplitudeSCN02R\\after', 
        'AmplitudeSCN03L\\after', 
        'AmplitudeSCN03R\\after', 
        'AmplitudeSCN04L\\after', 
        'AmplitudeSCN04R\\after', 
        'AmplitudeSCN05L\\after', 
        'AmplitudeSCN05R\\after', 
        'Amplitudel\\SCN06L\\aft', 
        'Amplitudel\\SCN06R\\aft', 
        'Amplitudel\\SCN07L\\aft',
        'Amplitudel\\SCN07R\\aft', 
        'Amplitudel\\SCN08L\\aft', 
        'Amplitudel\\SCN08R\\aft', 
        'Amplitudel\\SCN09L\\aft', 
        'Amplitudel\\SCN09R\\aft', 
        'Amplitudel\\SCN10L\\aft', 
        'Amplitudel\\SCN10R\\aft', 
        'Amplitudel\\SCN11L\\aft', 
        'Amplitudel\\SCN11R\\aft', 
        'Amplitudel\\SCN12L\\aft', 
        'Amplitudel\\SCN12R\\aft', 
        'Amplitudel\\SCN13L\\aft', 
        'Amplitudel\\SCN13R\\aft', 
        'Amplitudel\\SCN14L\\aft', 
        'Amplitudel\\SCN14R\\aft']]
        # Rename columns
        df3.columns = ['Amplitude_SCN01L_aft', 
        'Amplitude_SCN01R_aft', 
        'Amplitude_SCN02L_aft', 
        'Amplitude_SCN02R_aft', 
        'Amplitude_SCN03L_aft', 
        'Amplitude_SCN03R_aft', 
        'Amplitude_SCN04L_aft', 
        'Amplitude_SCN04R_aft', 
        'Amplitude_SCN05L_aft', 
        'Amplitude_SCN05R_aft', 
        'Amplitude_SCN06L_aft', 
        'Amplitude_SCN06R_aft', 
        'Amplitude_SCN07L_aft', 
        'Amplitude_SCN07R_aft', 
        'Amplitude_SCN08L_aft', 
        'Amplitude_SCN08R_aft', 
        'Amplitude_SCN09L_aft', 
        'Amplitude_SCN09R_aft', 
        'Amplitude_SCN10L_aft', 
        'Amplitude_SCN10R_aft', 
        'Amplitude_SCN11L_aft', 
        'Amplitude_SCN11R_aft', 
        'Amplitude_SCN12L_aft', 
        'Amplitude_SCN12R_aft', 
        'Amplitude_SCN13L_aft', 
        'Amplitude_SCN13R_aft', 
        'Amplitude_SCN14L_aft', 
        'Amplitude_SCN14R_aft']      
        melted = df3.melt(value_vars=['Amplitude_SCN01L_aft',
        'Amplitude_SCN01R_aft', 
        'Amplitude_SCN02L_aft', 
        'Amplitude_SCN02R_aft', 
        'Amplitude_SCN03L_aft', 
        'Amplitude_SCN03R_aft', 
        'Amplitude_SCN04L_aft', 
        'Amplitude_SCN04R_aft', 
        'Amplitude_SCN05L_aft', 
        'Amplitude_SCN05R_aft',
        'Amplitude_SCN06L_aft', 
        'Amplitude_SCN06R_aft', 
        'Amplitude_SCN07L_aft', 
        'Amplitude_SCN07R_aft', 
        'Amplitude_SCN08L_aft', 
        'Amplitude_SCN08R_aft', 
        'Amplitude_SCN09L_aft', 
        'Amplitude_SCN09R_aft', 
        'Amplitude_SCN10L_aft', 
        'Amplitude_SCN10R_aft', 
        'Amplitude_SCN11L_aft', 
        'Amplitude_SCN11R_aft', 
        'Amplitude_SCN12L_aft',
        'Amplitude_SCN12R_aft', 
        'Amplitude_SCN13L_aft', 
        'Amplitude_SCN13R_aft', 
        'Amplitude_SCN14L_aft', 
        'Amplitude_SCN14R_aft'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amp'] = melted['Var'].str.split('_').str[0]
       
        
       # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01L') | (melted['Prefix'] == 'SCN01R') | (melted['Prefix'] == 'SCN02L')
                   | (melted['Prefix'] == 'SCN02R')| (melted['Prefix'] == 'SCN03L')| (melted['Prefix'] == 'SCN03R')
                   | (melted['Prefix'] == 'SCN04L')| (melted['Prefix'] == 'SCN04R')| (melted['Prefix'] == 'SCN05L')
                   | (melted['Prefix'] == 'SCN05R') | (melted['Prefix'] == 'SCN06L')| (melted['Prefix'] == 'SCN06R')
                   | (melted['Prefix'] == 'SCN07L') | (melted['Prefix'] == 'SCN07R')), 'Prefix'] = 'DEX'
        
        melted.loc[((melted['Prefix'] == 'SCN08L') | (melted['Prefix'] == 'SCN08R') | (melted['Prefix'] == 'SCN09L')
                   | (melted['Prefix'] == 'SCN09R')| (melted['Prefix'] == 'SCN10L')| (melted['Prefix'] == 'SCN10R')
                   | (melted['Prefix'] == 'SCN11L')| (melted['Prefix'] == 'SCN11R')| (melted['Prefix'] == 'SCN12L')
                   | (melted['Prefix'] == 'SCN12R') | (melted['Prefix'] == 'SCN13L')| (melted['Prefix'] == 'SCN13R')
                   | (melted['Prefix'] == 'SCN14L') | (melted['Prefix'] == 'SCN14R')), 'Prefix'] = 'VEH'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('AMPLITUDE after COMBINE.csv')     
        

        # after explants
        df3 = df2[['TrendSCN01L\\after', 
        'TrendSCN01R\\after', 
        'TrendSCN02L\\after', 
        'TrendSCN02R\\after', 
        'TrendSCN03L\\after', 
        'TrendSCN03R\\after', 
        'TrendSCN04L\\after', 
        'TrendSCN04R\\after', 
        'TrendSCN05L\\after', 
        'TrendSCN05R\\after', 
        'Trendl\\SCN06L\\aft', 
        'Trendl\\SCN06R\\aft', 
        'Trendl\\SCN07L\\aft',
        'Trendl\\SCN07R\\aft', 
        'Trendl\\SCN08L\\aft', 
        'Trendl\\SCN08R\\aft', 
        'Trendl\\SCN09L\\aft', 
        'Trendl\\SCN09R\\aft', 
        'Trendl\\SCN10L\\aft', 
        'Trendl\\SCN10R\\aft', 
        'Trendl\\SCN11L\\aft', 
        'Trendl\\SCN11R\\aft', 
        'Trendl\\SCN12L\\aft', 
        'Trendl\\SCN12R\\aft', 
        'Trendl\\SCN13L\\aft', 
        'Trendl\\SCN13R\\aft', 
        'Trendl\\SCN14L\\aft', 
        'Trendl\\SCN14R\\aft']]
        # Rename columns
        df3.columns = ['Trend_SCN01L_aft', 
        'Trend_SCN01R_aft', 
        'Trend_SCN02L_aft', 
        'Trend_SCN02R_aft', 
        'Trend_SCN03L_aft', 
        'Trend_SCN03R_aft', 
        'Trend_SCN04L_aft', 
        'Trend_SCN04R_aft', 
        'Trend_SCN05L_aft', 
        'Trend_SCN05R_aft', 
        'Trend_SCN06L_aft', 
        'Trend_SCN06R_aft', 
        'Trend_SCN07L_aft', 
        'Trend_SCN07R_aft', 
        'Trend_SCN08L_aft', 
        'Trend_SCN08R_aft', 
        'Trend_SCN09L_aft', 
        'Trend_SCN09R_aft', 
        'Trend_SCN10L_aft', 
        'Trend_SCN10R_aft', 
        'Trend_SCN11L_aft', 
        'Trend_SCN11R_aft', 
        'Trend_SCN12L_aft', 
        'Trend_SCN12R_aft', 
        'Trend_SCN13L_aft', 
        'Trend_SCN13R_aft', 
        'Trend_SCN14L_aft', 
        'Trend_SCN14R_aft']      
        melted = df3.melt(value_vars=['Trend_SCN01L_aft',
        'Trend_SCN01R_aft', 
        'Trend_SCN02L_aft', 
        'Trend_SCN02R_aft', 
        'Trend_SCN03L_aft', 
        'Trend_SCN03R_aft', 
        'Trend_SCN04L_aft', 
        'Trend_SCN04R_aft', 
        'Trend_SCN05L_aft', 
        'Trend_SCN05R_aft',
        'Trend_SCN06L_aft', 
        'Trend_SCN06R_aft', 
        'Trend_SCN07L_aft', 
        'Trend_SCN07R_aft', 
        'Trend_SCN08L_aft', 
        'Trend_SCN08R_aft', 
        'Trend_SCN09L_aft', 
        'Trend_SCN09R_aft', 
        'Trend_SCN10L_aft', 
        'Trend_SCN10R_aft', 
        'Trend_SCN11L_aft', 
        'Trend_SCN11R_aft', 
        'Trend_SCN12L_aft',
        'Trend_SCN12R_aft', 
        'Trend_SCN13L_aft', 
        'Trend_SCN13R_aft', 
        'Trend_SCN14L_aft', 
        'Trend_SCN14R_aft'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amp'] = melted['Var'].str.split('_').str[0]
       
        
       # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01L') | (melted['Prefix'] == 'SCN01R') | (melted['Prefix'] == 'SCN02L')
                   | (melted['Prefix'] == 'SCN02R')| (melted['Prefix'] == 'SCN03L')| (melted['Prefix'] == 'SCN03R')
                   | (melted['Prefix'] == 'SCN04L')| (melted['Prefix'] == 'SCN04R')| (melted['Prefix'] == 'SCN05L')
                   | (melted['Prefix'] == 'SCN05R') | (melted['Prefix'] == 'SCN06L')| (melted['Prefix'] == 'SCN06R')
                   | (melted['Prefix'] == 'SCN07L') | (melted['Prefix'] == 'SCN07R')), 'Prefix'] = 'DEX'
        
        melted.loc[((melted['Prefix'] == 'SCN08L') | (melted['Prefix'] == 'SCN08R') | (melted['Prefix'] == 'SCN09L')
                   | (melted['Prefix'] == 'SCN09R')| (melted['Prefix'] == 'SCN10L')| (melted['Prefix'] == 'SCN10R')
                   | (melted['Prefix'] == 'SCN11L')| (melted['Prefix'] == 'SCN11R')| (melted['Prefix'] == 'SCN12L')
                   | (melted['Prefix'] == 'SCN12R') | (melted['Prefix'] == 'SCN13L')| (melted['Prefix'] == 'SCN13R')
                   | (melted['Prefix'] == 'SCN14L') | (melted['Prefix'] == 'SCN14R')), 'Prefix'] = 'VEH'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('Trend after COMBINE.csv')     
                
         
        # after explants
        df3 = df2[['PeriodSCN01L\\after', 
        'PeriodSCN01R\\after', 
        'PeriodSCN02L\\after', 
        'PeriodSCN02R\\after', 
        'PeriodSCN03L\\after', 
        'PeriodSCN03R\\after', 
        'PeriodSCN04L\\after', 
        'PeriodSCN04R\\after', 
        'PeriodSCN05L\\after', 
        'PeriodSCN05R\\after', 
        'Periodl\\SCN06L\\aft', 
        'Periodl\\SCN06R\\aft', 
        'Periodl\\SCN07L\\aft',
        'Periodl\\SCN07R\\aft', 
        'Periodl\\SCN08L\\aft', 
        'Periodl\\SCN08R\\aft', 
        'Periodl\\SCN09L\\aft', 
        'Periodl\\SCN09R\\aft', 
        'Periodl\\SCN10L\\aft', 
        'Periodl\\SCN10R\\aft', 
        'Periodl\\SCN11L\\aft', 
        'Periodl\\SCN11R\\aft', 
        'Periodl\\SCN12L\\aft', 
        'Periodl\\SCN12R\\aft', 
        'Periodl\\SCN13L\\aft', 
        'Periodl\\SCN13R\\aft', 
        'Periodl\\SCN14L\\aft', 
        'Periodl\\SCN14R\\aft']]
        # Rename columns
        df3.columns = ['Period_SCN01L_aft', 
        'Period_SCN01R_aft', 
        'Period_SCN02L_aft', 
        'Period_SCN02R_aft', 
        'Period_SCN03L_aft', 
        'Period_SCN03R_aft', 
        'Period_SCN04L_aft', 
        'Period_SCN04R_aft', 
        'Period_SCN05L_aft', 
        'Period_SCN05R_aft', 
        'Period_SCN06L_aft', 
        'Period_SCN06R_aft', 
        'Period_SCN07L_aft', 
        'Period_SCN07R_aft', 
        'Period_SCN08L_aft', 
        'Period_SCN08R_aft', 
        'Period_SCN09L_aft', 
        'Period_SCN09R_aft', 
        'Period_SCN10L_aft', 
        'Period_SCN10R_aft', 
        'Period_SCN11L_aft', 
        'Period_SCN11R_aft', 
        'Period_SCN12L_aft', 
        'Period_SCN12R_aft', 
        'Period_SCN13L_aft', 
        'Period_SCN13R_aft', 
        'Period_SCN14L_aft', 
        'Period_SCN14R_aft']      
        melted = df3.melt(value_vars=['Period_SCN01L_aft',
        'Period_SCN01R_aft', 
        'Period_SCN02L_aft', 
        'Period_SCN02R_aft', 
        'Period_SCN03L_aft', 
        'Period_SCN03R_aft', 
        'Period_SCN04L_aft', 
        'Period_SCN04R_aft', 
        'Period_SCN05L_aft', 
        'Period_SCN05R_aft',
        'Period_SCN06L_aft', 
        'Period_SCN06R_aft', 
        'Period_SCN07L_aft', 
        'Period_SCN07R_aft', 
        'Period_SCN08L_aft', 
        'Period_SCN08R_aft', 
        'Period_SCN09L_aft', 
        'Period_SCN09R_aft', 
        'Period_SCN10L_aft', 
        'Period_SCN10R_aft', 
        'Period_SCN11L_aft', 
        'Period_SCN11R_aft', 
        'Period_SCN12L_aft',
        'Period_SCN12R_aft', 
        'Period_SCN13L_aft', 
        'Period_SCN13R_aft', 
        'Period_SCN14L_aft', 
        'Period_SCN14R_aft'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amp'] = melted['Var'].str.split('_').str[0]
       
        
       # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01L') | (melted['Prefix'] == 'SCN01R') | (melted['Prefix'] == 'SCN02L')
                   | (melted['Prefix'] == 'SCN02R')| (melted['Prefix'] == 'SCN03L')| (melted['Prefix'] == 'SCN03R')
                   | (melted['Prefix'] == 'SCN04L')| (melted['Prefix'] == 'SCN04R')| (melted['Prefix'] == 'SCN05L')
                   | (melted['Prefix'] == 'SCN05R') | (melted['Prefix'] == 'SCN06L')| (melted['Prefix'] == 'SCN06R')
                   | (melted['Prefix'] == 'SCN07L') | (melted['Prefix'] == 'SCN07R')), 'Prefix'] = 'DEX'
        
        melted.loc[((melted['Prefix'] == 'SCN08L') | (melted['Prefix'] == 'SCN08R') | (melted['Prefix'] == 'SCN09L')
                   | (melted['Prefix'] == 'SCN09R')| (melted['Prefix'] == 'SCN10L')| (melted['Prefix'] == 'SCN10R')
                   | (melted['Prefix'] == 'SCN11L')| (melted['Prefix'] == 'SCN11R')| (melted['Prefix'] == 'SCN12L')
                   | (melted['Prefix'] == 'SCN12R') | (melted['Prefix'] == 'SCN13L')| (melted['Prefix'] == 'SCN13R')
                   | (melted['Prefix'] == 'SCN14L') | (melted['Prefix'] == 'SCN14R')), 'Prefix'] = 'VEH'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('Period after COMBINE.csv')  


        # after explants
        df3 = df2[['RsqSCN01L\\after', 
        'RsqSCN01R\\after', 
        'RsqSCN02L\\after', 
        'RsqSCN02R\\after', 
        'RsqSCN03L\\after', 
        'RsqSCN03R\\after', 
        'RsqSCN04L\\after', 
        'RsqSCN04R\\after', 
        'RsqSCN05L\\after', 
        'RsqSCN05R\\after', 
        'Rsql\\SCN06L\\aft', 
        'Rsql\\SCN06R\\aft', 
        'Rsql\\SCN07L\\aft',
        'Rsql\\SCN07R\\aft', 
        'Rsql\\SCN08L\\aft', 
        'Rsql\\SCN08R\\aft', 
        'Rsql\\SCN09L\\aft', 
        'Rsql\\SCN09R\\aft', 
        'Rsql\\SCN10L\\aft', 
        'Rsql\\SCN10R\\aft', 
        'Rsql\\SCN11L\\aft', 
        'Rsql\\SCN11R\\aft', 
        'Rsql\\SCN12L\\aft', 
        'Rsql\\SCN12R\\aft', 
        'Rsql\\SCN13L\\aft', 
        'Rsql\\SCN13R\\aft', 
        'Rsql\\SCN14L\\aft', 
        'Rsql\\SCN14R\\aft']]
        # Rename columns
        df3.columns = ['Rsq_SCN01L_aft', 
        'Rsq_SCN01R_aft', 
        'Rsq_SCN02L_aft', 
        'Rsq_SCN02R_aft', 
        'Rsq_SCN03L_aft', 
        'Rsq_SCN03R_aft', 
        'Rsq_SCN04L_aft', 
        'Rsq_SCN04R_aft', 
        'Rsq_SCN05L_aft', 
        'Rsq_SCN05R_aft', 
        'Rsq_SCN06L_aft', 
        'Rsq_SCN06R_aft', 
        'Rsq_SCN07L_aft', 
        'Rsq_SCN07R_aft', 
        'Rsq_SCN08L_aft', 
        'Rsq_SCN08R_aft', 
        'Rsq_SCN09L_aft', 
        'Rsq_SCN09R_aft', 
        'Rsq_SCN10L_aft', 
        'Rsq_SCN10R_aft', 
        'Rsq_SCN11L_aft', 
        'Rsq_SCN11R_aft', 
        'Rsq_SCN12L_aft', 
        'Rsq_SCN12R_aft', 
        'Rsq_SCN13L_aft', 
        'Rsq_SCN13R_aft', 
        'Rsq_SCN14L_aft', 
        'Rsq_SCN14R_aft']      
        melted = df3.melt(value_vars=['Rsq_SCN01L_aft',
        'Rsq_SCN01R_aft', 
        'Rsq_SCN02L_aft', 
        'Rsq_SCN02R_aft', 
        'Rsq_SCN03L_aft', 
        'Rsq_SCN03R_aft', 
        'Rsq_SCN04L_aft', 
        'Rsq_SCN04R_aft', 
        'Rsq_SCN05L_aft', 
        'Rsq_SCN05R_aft',
        'Rsq_SCN06L_aft', 
        'Rsq_SCN06R_aft', 
        'Rsq_SCN07L_aft', 
        'Rsq_SCN07R_aft', 
        'Rsq_SCN08L_aft', 
        'Rsq_SCN08R_aft', 
        'Rsq_SCN09L_aft', 
        'Rsq_SCN09R_aft', 
        'Rsq_SCN10L_aft', 
        'Rsq_SCN10R_aft', 
        'Rsq_SCN11L_aft', 
        'Rsq_SCN11R_aft', 
        'Rsq_SCN12L_aft',
        'Rsq_SCN12R_aft', 
        'Rsq_SCN13L_aft', 
        'Rsq_SCN13R_aft', 
        'Rsq_SCN14L_aft', 
        'Rsq_SCN14R_aft'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amp'] = melted['Var'].str.split('_').str[0]
       
        
       # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01L') | (melted['Prefix'] == 'SCN01R') | (melted['Prefix'] == 'SCN02L')
                   | (melted['Prefix'] == 'SCN02R')| (melted['Prefix'] == 'SCN03L')| (melted['Prefix'] == 'SCN03R')
                   | (melted['Prefix'] == 'SCN04L')| (melted['Prefix'] == 'SCN04R')| (melted['Prefix'] == 'SCN05L')
                   | (melted['Prefix'] == 'SCN05R') | (melted['Prefix'] == 'SCN06L')| (melted['Prefix'] == 'SCN06R')
                   | (melted['Prefix'] == 'SCN07L') | (melted['Prefix'] == 'SCN07R')), 'Prefix'] = 'DEX'
        
        melted.loc[((melted['Prefix'] == 'SCN08L') | (melted['Prefix'] == 'SCN08R') | (melted['Prefix'] == 'SCN09L')
                   | (melted['Prefix'] == 'SCN09R')| (melted['Prefix'] == 'SCN10L')| (melted['Prefix'] == 'SCN10R')
                   | (melted['Prefix'] == 'SCN11L')| (melted['Prefix'] == 'SCN11R')| (melted['Prefix'] == 'SCN12L')
                   | (melted['Prefix'] == 'SCN12R') | (melted['Prefix'] == 'SCN13L')| (melted['Prefix'] == 'SCN13R')
                   | (melted['Prefix'] == 'SCN14L') | (melted['Prefix'] == 'SCN14R')), 'Prefix'] = 'VEH'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('Rsq after COMBINE.csv') 


        # after explants
        df3 = df2[['DecaySCN01L\\after', 
        'DecaySCN01R\\after', 
        'DecaySCN02L\\after', 
        'DecaySCN02R\\after', 
        'DecaySCN03L\\after', 
        'DecaySCN03R\\after', 
        'DecaySCN04L\\after', 
        'DecaySCN04R\\after', 
        'DecaySCN05L\\after', 
        'DecaySCN05R\\after', 
        'Decayl\\SCN06L\\aft', 
        'Decayl\\SCN06R\\aft', 
        'Decayl\\SCN07L\\aft',
        'Decayl\\SCN07R\\aft', 
        'Decayl\\SCN08L\\aft', 
        'Decayl\\SCN08R\\aft', 
        'Decayl\\SCN09L\\aft', 
        'Decayl\\SCN09R\\aft', 
        'Decayl\\SCN10L\\aft', 
        'Decayl\\SCN10R\\aft', 
        'Decayl\\SCN11L\\aft', 
        'Decayl\\SCN11R\\aft', 
        'Decayl\\SCN12L\\aft', 
        'Decayl\\SCN12R\\aft', 
        'Decayl\\SCN13L\\aft', 
        'Decayl\\SCN13R\\aft', 
        'Decayl\\SCN14L\\aft', 
        'Decayl\\SCN14R\\aft']]
        # Rename columns
        df3.columns = ['Decay_SCN01L_aft', 
        'Decay_SCN01R_aft', 
        'Decay_SCN02L_aft', 
        'Decay_SCN02R_aft', 
        'Decay_SCN03L_aft', 
        'Decay_SCN03R_aft', 
        'Decay_SCN04L_aft', 
        'Decay_SCN04R_aft', 
        'Decay_SCN05L_aft', 
        'Decay_SCN05R_aft', 
        'Decay_SCN06L_aft', 
        'Decay_SCN06R_aft', 
        'Decay_SCN07L_aft', 
        'Decay_SCN07R_aft', 
        'Decay_SCN08L_aft', 
        'Decay_SCN08R_aft', 
        'Decay_SCN09L_aft', 
        'Decay_SCN09R_aft', 
        'Decay_SCN10L_aft', 
        'Decay_SCN10R_aft', 
        'Decay_SCN11L_aft', 
        'Decay_SCN11R_aft', 
        'Decay_SCN12L_aft', 
        'Decay_SCN12R_aft', 
        'Decay_SCN13L_aft', 
        'Decay_SCN13R_aft', 
        'Decay_SCN14L_aft', 
        'Decay_SCN14R_aft']      
        melted = df3.melt(value_vars=['Decay_SCN01L_aft',
        'Decay_SCN01R_aft', 
        'Decay_SCN02L_aft', 
        'Decay_SCN02R_aft', 
        'Decay_SCN03L_aft', 
        'Decay_SCN03R_aft', 
        'Decay_SCN04L_aft', 
        'Decay_SCN04R_aft', 
        'Decay_SCN05L_aft', 
        'Decay_SCN05R_aft',
        'Decay_SCN06L_aft', 
        'Decay_SCN06R_aft', 
        'Decay_SCN07L_aft', 
        'Decay_SCN07R_aft', 
        'Decay_SCN08L_aft', 
        'Decay_SCN08R_aft', 
        'Decay_SCN09L_aft', 
        'Decay_SCN09R_aft', 
        'Decay_SCN10L_aft', 
        'Decay_SCN10R_aft', 
        'Decay_SCN11L_aft', 
        'Decay_SCN11R_aft', 
        'Decay_SCN12L_aft',
        'Decay_SCN12R_aft', 
        'Decay_SCN13L_aft', 
        'Decay_SCN13R_aft', 
        'Decay_SCN14L_aft', 
        'Decay_SCN14R_aft'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amp'] = melted['Var'].str.split('_').str[0]
       
        
       # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'SCN01L') | (melted['Prefix'] == 'SCN01R') | (melted['Prefix'] == 'SCN02L')
                   | (melted['Prefix'] == 'SCN02R')| (melted['Prefix'] == 'SCN03L')| (melted['Prefix'] == 'SCN03R')
                   | (melted['Prefix'] == 'SCN04L')| (melted['Prefix'] == 'SCN04R')| (melted['Prefix'] == 'SCN05L')
                   | (melted['Prefix'] == 'SCN05R') | (melted['Prefix'] == 'SCN06L')| (melted['Prefix'] == 'SCN06R')
                   | (melted['Prefix'] == 'SCN07L') | (melted['Prefix'] == 'SCN07R')), 'Prefix'] = 'DEX'
        
        melted.loc[((melted['Prefix'] == 'SCN08L') | (melted['Prefix'] == 'SCN08R') | (melted['Prefix'] == 'SCN09L')
                   | (melted['Prefix'] == 'SCN09R')| (melted['Prefix'] == 'SCN10L')| (melted['Prefix'] == 'SCN10R')
                   | (melted['Prefix'] == 'SCN11L')| (melted['Prefix'] == 'SCN11R')| (melted['Prefix'] == 'SCN12L')
                   | (melted['Prefix'] == 'SCN12R') | (melted['Prefix'] == 'SCN13L')| (melted['Prefix'] == 'SCN13R')
                   | (melted['Prefix'] == 'SCN14L') | (melted['Prefix'] == 'SCN14R')), 'Prefix'] = 'VEH'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('Decay after COMBINE.csv') 
                  
        """
        
        # FOR CHRONODISRUPTION ChP paper (Milica) - here it separates up to 36 explants from 6 ChP experiment to 2-3 groups and combines them for statistics
        """
        df3 = df2[['RsqCP00A', 'RsqCP00B', 'RsqCP00C', 'RsqCP00D', 'Rsq\\CP01', 'Rsq\\CP02', 'Rsq\\CP03', 'Rsq\\CP04',
                   'Rsq\\CP07', 'Rsq\\CP08', 'Rsq\\CP09', 'Rsq\\CP10', 'Rsq\\CP11', 'Rsq\\CP12', 'Rsq\\CP13', 'Rsq\\CP14', 'Rsq\\CP15', 'Rsq\\CP16', 'Rsq\\CP17', 'Rsq\\CP18']]
        # Rename columns
        df3.columns = ['Rsq_CP00A', 'Rsq_CP00B', 'Rsq_CP00C', 'Rsq_CP00D', 'Rsq_CP01', 'Rsq_CP02', 'Rsq_CP03', 'Rsq_CP04',
                   'Rsq_CP07', 'Rsq_CP08', 'Rsq_CP09', 'Rsq_CP10', 'Rsq_CP11', 'Rsq_CP12', 'Rsq_CP13', 'Rsq_CP14', 'Rsq_CP15', 'Rsq_CP16', 'Rsq_CP17', 'Rsq_CP18']        
        melted = df3.melt(value_vars=['Rsq_CP00A', 'Rsq_CP00B', 'Rsq_CP00C', 'Rsq_CP00D', 'Rsq_CP01', 'Rsq_CP02', 'Rsq_CP03', 'Rsq_CP04',
                   'Rsq_CP07', 'Rsq_CP08', 'Rsq_CP09', 'Rsq_CP10', 'Rsq_CP11', 'Rsq_CP12', 'Rsq_CP13', 'Rsq_CP14', 'Rsq_CP15', 'Rsq_CP16', 'Rsq_CP17', 'Rsq_CP18'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')), 'Prefix'] = 'LD'
       
        melted.loc[((melted['Prefix'] == 'CP07') | (melted['Prefix'] == 'CP08') | (melted['Prefix'] == 'CP09')
                   | (melted['Prefix'] == 'CP10')| (melted['Prefix'] == 'CP11')| (melted['Prefix'] == 'CP12')
                   | (melted['Prefix'] == 'CP13') | (melted['Prefix'] == 'CP14') | (melted['Prefix'] == 'CP15') | (melted['Prefix'] == 'CP16') 
                   | (melted['Prefix'] == 'CP17') | (melted['Prefix'] == 'CP18')), 'Prefix'] = 'SHIFTS'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)               
        violin_stat(data=df, title='Nice Rsq of all cells LD vs SHIFTS', ticks=df.columns, remove_outliers=False, iqr_value=8.88, test='mann')  
        
        
        
        df3 = df2[['RsqCP00A', 'RsqCP00B', 'RsqCP00C', 'RsqCP00D', 'Rsq\\CP01', 'Rsq\\CP02', 'Rsq\\CP03', 'Rsq\\CP04',
                   'Rsq\\CP19', 'Rsq\\CP20', 'Rsq\\CP21', 'Rsq\\CP22', 'Rsq\\CP23', 'Rsq\\CP24', 'Rsq\\CP25', 'Rsq\\CP26', 'Rsq\\CP27', 'Rsq\\CP28', 'Rsq\\CP29', 'Rsq\\CP30']]
        # Rename columns
        df3.columns = ['Rsq_CP00A', 'Rsq_CP00B', 'Rsq_CP00C', 'Rsq_CP00D', 'Rsq_CP01', 'Rsq_CP02', 'Rsq_CP03', 'Rsq_CP04',
                   'Rsq_CP19', 'Rsq_CP20', 'Rsq_CP21', 'Rsq_CP22', 'Rsq_CP23', 'Rsq_CP24', 'Rsq_CP25', 'Rsq_CP26', 'Rsq_CP27', 'Rsq_CP28', 'Rsq_CP29', 'Rsq_CP30']        
        melted = df3.melt(value_vars=['Rsq_CP00A', 'Rsq_CP00B', 'Rsq_CP00C', 'Rsq_CP00D', 'Rsq_CP01', 'Rsq_CP02', 'Rsq_CP03', 'Rsq_CP04',
                   'Rsq_CP19', 'Rsq_CP20', 'Rsq_CP21', 'Rsq_CP22', 'Rsq_CP23', 'Rsq_CP24', 'Rsq_CP25', 'Rsq_CP26', 'Rsq_CP27', 'Rsq_CP28', 'Rsq_CP29', 'Rsq_CP30'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')), 'Prefix'] = 'LD'
       
        melted.loc[((melted['Prefix'] == 'CP19') | (melted['Prefix'] == 'CP20') | (melted['Prefix'] == 'CP21')
                   | (melted['Prefix'] == 'CP22')| (melted['Prefix'] == 'CP23')| (melted['Prefix'] == 'CP24')
                   | (melted['Prefix'] == 'CP25') | (melted['Prefix'] == 'CP26') | (melted['Prefix'] == 'CP27') | (melted['Prefix'] == 'CP28') 
                   | (melted['Prefix'] == 'CP29') | (melted['Prefix'] == 'CP30')), 'Prefix'] = 'LL'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)               
        violin_stat(data=df, title='Nice Rsq of all cells LD vs LL', ticks=df.columns, remove_outliers=True, iqr_value=8.88, test='mann')  
            
        
        df3 = df2[['AmplitudeCP00A', 'AmplitudeCP00B', 'AmplitudeCP00C', 'AmplitudeCP00D', 'Amplitude\\CP01', 'Amplitude\\CP02', 'Amplitude\\CP03',
        'Amplitude\\CP04', 
        'Amplitude\\CP07', 'Amplitude\\CP08', 'Amplitude\\CP09',
        'Amplitude\\CP10', 'Amplitude\\CP11', 'Amplitude\\CP12',
        'Amplitude\\CP13', 'Amplitude\\CP14', 'Amplitude\\CP15',
        'Amplitude\\CP16', 'Amplitude\\CP17', 'Amplitude\\CP18']]
        # Rename columns
        df3.columns = ['Amplitude_CP00A', 'Amplitude_CP00B', 'Amplitude_CP00C', 'Amplitude_CP00D', 'Amplitude_CP01', 'Amplitude_CP02', 'Amplitude_CP03',
        'Amplitude_CP04', 
        'Amplitude_CP07', 'Amplitude_CP08', 'Amplitude_CP09',
        'Amplitude_CP10', 'Amplitude_CP11', 'Amplitude_CP12',
        'Amplitude_CP13', 'Amplitude_CP14', 'Amplitude_CP15',
        'Amplitude_CP16', 'Amplitude_CP17', 'Amplitude_CP18']        
        melted = df3.melt(value_vars=['Amplitude_CP00A', 'Amplitude_CP00B', 'Amplitude_CP00C', 'Amplitude_CP00D', 'Amplitude_CP01', 'Amplitude_CP02', 'Amplitude_CP03',
        'Amplitude_CP04', 
        'Amplitude_CP07', 'Amplitude_CP08', 'Amplitude_CP09',
        'Amplitude_CP10', 'Amplitude_CP11', 'Amplitude_CP12',
        'Amplitude_CP13', 'Amplitude_CP14', 'Amplitude_CP15',
        'Amplitude_CP16', 'Amplitude_CP17', 'Amplitude_CP18'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')), 'Prefix'] = 'LD'
       
        melted.loc[((melted['Prefix'] == 'CP07') | (melted['Prefix'] == 'CP08') | (melted['Prefix'] == 'CP09')
                   | (melted['Prefix'] == 'CP10')| (melted['Prefix'] == 'CP11')| (melted['Prefix'] == 'CP12')
                   | (melted['Prefix'] == 'CP13') | (melted['Prefix'] == 'CP14') | (melted['Prefix'] == 'CP15') | (melted['Prefix'] == 'CP16') 
                   | (melted['Prefix'] == 'CP17') | (melted['Prefix'] == 'CP18')), 'Prefix'] = 'SHIFTS'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)               
        violin_stat(data=df, title='Amp of all cells LD vs SHIFTS', ticks=df.columns, remove_outliers=False, iqr_value=8.88, test='mann')  
        
        
        
        df3 = df2[['AmplitudeCP00A', 'AmplitudeCP00B', 'AmplitudeCP00C', 'AmplitudeCP00D', 'Amplitude\\CP01', 'Amplitude\\CP02', 'Amplitude\\CP03',
        'Amplitude\\CP04', 
        'Amplitude\\CP19', 'Amplitude\\CP20', 'Amplitude\\CP21',
        'Amplitude\\CP22', 'Amplitude\\CP23', 'Amplitude\\CP24',
        'Amplitude\\CP25', 'Amplitude\\CP26', 'Amplitude\\CP27',
        'Amplitude\\CP28', 'Amplitude\\CP29', 'Amplitude\\CP30']]
        # Rename columns
        df3.columns = ['Amplitude_CP00A', 'Amplitude_CP00B', 'Amplitude_CP00C', 'Amplitude_CP00D', 'Amplitude_CP01', 'Amplitude_CP02', 'Amplitude_CP03',
        'Amplitude_CP04', 
        'Amplitude_CP19', 'Amplitude_CP20', 'Amplitude_CP21',
        'Amplitude_CP22', 'Amplitude_CP23', 'Amplitude_CP24',
        'Amplitude_CP25', 'Amplitude_CP26', 'Amplitude_CP27',
        'Amplitude_CP28', 'Amplitude_CP29', 'Amplitude_CP30']        
        melted = df3.melt(value_vars=['Amplitude_CP00A', 'Amplitude_CP00B', 'Amplitude_CP00C', 'Amplitude_CP00D', 'Amplitude_CP01', 'Amplitude_CP02', 'Amplitude_CP03',
        'Amplitude_CP04', 
        'Amplitude_CP19', 'Amplitude_CP20', 'Amplitude_CP21',
        'Amplitude_CP22', 'Amplitude_CP23', 'Amplitude_CP24',
        'Amplitude_CP25', 'Amplitude_CP26', 'Amplitude_CP27',
        'Amplitude_CP28', 'Amplitude_CP29', 'Amplitude_CP30'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')), 'Prefix'] = 'LD'
       
        melted.loc[((melted['Prefix'] == 'CP19') | (melted['Prefix'] == 'CP20') | (melted['Prefix'] == 'CP21')
                   | (melted['Prefix'] == 'CP22')| (melted['Prefix'] == 'CP23')| (melted['Prefix'] == 'CP24')
                   | (melted['Prefix'] == 'CP25') | (melted['Prefix'] == 'CP26') | (melted['Prefix'] == 'CP27') | (melted['Prefix'] == 'CP28') 
                   | (melted['Prefix'] == 'CP29') | (melted['Prefix'] == 'CP30')), 'Prefix'] = 'LL'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)               
        violin_stat(data=df, title='Amp of all cells LD vs LL', ticks=df.columns, remove_outliers=True, iqr_value=8.88, test='mann')  
        
       
        
       
        # all three groups
        df3 = df2[['AmplitudeCP00A', 'AmplitudeCP00B', 'AmplitudeCP00C', 'AmplitudeCP00D', 'AmplitudeCP00E', 'AmplitudeCP00F',
        'Amplitude\\CP01', 'Amplitude\\CP02', 'Amplitude\\CP03',
        'Amplitude\\CP04', 'Amplitude\\CP05', 'Amplitude\\CP06',
        'Amplitude\\CP07', 'Amplitude\\CP08', 'Amplitude\\CP09',
        'Amplitude\\CP10', 'Amplitude\\CP11', 'Amplitude\\CP12',
        'Amplitude\\CP13', 'Amplitude\\CP14', 'Amplitude\\CP15',
        'Amplitude\\CP16', 'Amplitude\\CP17', 'Amplitude\\CP18',        
        'Amplitude\\CP19', 'Amplitude\\CP20', 'Amplitude\\CP21',
        'Amplitude\\CP22', 'Amplitude\\CP23', 'Amplitude\\CP24',
        'Amplitude\\CP25', 'Amplitude\\CP26', 'Amplitude\\CP27',
        'Amplitude\\CP28', 'Amplitude\\CP29', 'Amplitude\\CP30']]
        # Rename columns
        df3.columns = ['Amplitude_CP00A', 'Amplitude_CP00B', 'Amplitude_CP00C', 'Amplitude_CP00D', 'Amplitude_CP00E', 'Amplitude_CP00F',
        'Amplitude_CP01', 'Amplitude_CP02', 'Amplitude_CP03',
        'Amplitude_CP04', 'Amplitude_CP05', 'Amplitude_CP06',
        'Amplitude_CP07', 'Amplitude_CP08', 'Amplitude_CP09',
        'Amplitude_CP10', 'Amplitude_CP11', 'Amplitude_CP12',
        'Amplitude_CP13', 'Amplitude_CP14', 'Amplitude_CP15',
        'Amplitude_CP16', 'Amplitude_CP17', 'Amplitude_CP18',        
        'Amplitude_CP19', 'Amplitude_CP20', 'Amplitude_CP21',
        'Amplitude_CP22', 'Amplitude_CP23', 'Amplitude_CP24',
        'Amplitude_CP25', 'Amplitude_CP26', 'Amplitude_CP27',
        'Amplitude_CP28', 'Amplitude_CP29', 'Amplitude_CP30']      
        melted = df3.melt(value_vars=['Amplitude_CP00A', 'Amplitude_CP00B', 'Amplitude_CP00C', 'Amplitude_CP00D', 'Amplitude_CP00E', 'Amplitude_CP00F',
        'Amplitude_CP01', 'Amplitude_CP02', 'Amplitude_CP03',
        'Amplitude_CP04', 'Amplitude_CP05', 'Amplitude_CP06',
        'Amplitude_CP07', 'Amplitude_CP08', 'Amplitude_CP09',
        'Amplitude_CP10', 'Amplitude_CP11', 'Amplitude_CP12',
        'Amplitude_CP13', 'Amplitude_CP14', 'Amplitude_CP15',
        'Amplitude_CP16', 'Amplitude_CP17', 'Amplitude_CP18',        
        'Amplitude_CP19', 'Amplitude_CP20', 'Amplitude_CP21',
        'Amplitude_CP22', 'Amplitude_CP23', 'Amplitude_CP24',
        'Amplitude_CP25', 'Amplitude_CP26', 'Amplitude_CP27',
        'Amplitude_CP28', 'Amplitude_CP29', 'Amplitude_CP30'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amp'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP00E')| (melted['Prefix'] == 'CP00F')
                   | (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')
                   | (melted['Prefix'] == 'CP05') | (melted['Prefix'] == 'CP06')), 'Prefix'] = 'LD'
        
        melted.loc[((melted['Prefix'] == 'CP07') | (melted['Prefix'] == 'CP08') | (melted['Prefix'] == 'CP09')
                   | (melted['Prefix'] == 'CP10')| (melted['Prefix'] == 'CP11')| (melted['Prefix'] == 'CP12')
                   | (melted['Prefix'] == 'CP13') | (melted['Prefix'] == 'CP14') | (melted['Prefix'] == 'CP15') | (melted['Prefix'] == 'CP16') 
                   | (melted['Prefix'] == 'CP17') | (melted['Prefix'] == 'CP18')), 'Prefix'] = 'SHIFTS'
       
        melted.loc[((melted['Prefix'] == 'CP19') | (melted['Prefix'] == 'CP20') | (melted['Prefix'] == 'CP21')
                   | (melted['Prefix'] == 'CP22')| (melted['Prefix'] == 'CP23')| (melted['Prefix'] == 'CP24')
                   | (melted['Prefix'] == 'CP25') | (melted['Prefix'] == 'CP26') | (melted['Prefix'] == 'CP27') | (melted['Prefix'] == 'CP28') 
                   | (melted['Prefix'] == 'CP29') | (melted['Prefix'] == 'CP30')), 'Prefix'] = 'xLL'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('AMPLITUDE COMBINE.csv')
        
        KW_BoxenPlot1(data=df, path=path, kind='boxen', name='Amplitude')
        
        # all three groups
        df3 = df2[['PeriodCP00A', 'PeriodCP00B', 'PeriodCP00C', 'PeriodCP00D', 'PeriodCP00E', 'PeriodCP00F',
        'Period\\CP01', 'Period\\CP02', 'Period\\CP03',
        'Period\\CP04', 'Period\\CP05', 'Period\\CP06',
        'Period\\CP07', 'Period\\CP08', 'Period\\CP09',
        'Period\\CP10', 'Period\\CP11', 'Period\\CP12',
        'Period\\CP13', 'Period\\CP14', 'Period\\CP15',
        'Period\\CP16', 'Period\\CP17', 'Period\\CP18',        
        'Period\\CP19', 'Period\\CP20', 'Period\\CP21',
        'Period\\CP22', 'Period\\CP23', 'Period\\CP24',
        'Period\\CP25', 'Period\\CP26', 'Period\\CP27',
        'Period\\CP28', 'Period\\CP29', 'Period\\CP30']]
        # Rename columns
        df3.columns = ['Period_CP00A', 'Period_CP00B', 'Period_CP00C', 'Period_CP00D', 'Period_CP00E', 'Period_CP00F',
        'Period_CP01', 'Period_CP02', 'Period_CP03',
        'Period_CP04', 'Period_CP05', 'Period_CP06',
        'Period_CP07', 'Period_CP08', 'Period_CP09',
        'Period_CP10', 'Period_CP11', 'Period_CP12',
        'Period_CP13', 'Period_CP14', 'Period_CP15',
        'Period_CP16', 'Period_CP17', 'Period_CP18',        
        'Period_CP19', 'Period_CP20', 'Period_CP21',
        'Period_CP22', 'Period_CP23', 'Period_CP24',
        'Period_CP25', 'Period_CP26', 'Period_CP27',
        'Period_CP28', 'Period_CP29', 'Period_CP30']      
        melted = df3.melt(value_vars=['Period_CP00A', 'Period_CP00B', 'Period_CP00C', 'Period_CP00D', 'Period_CP00E', 'Period_CP00F',
        'Period_CP01', 'Period_CP02', 'Period_CP03',
        'Period_CP04', 'Period_CP05', 'Period_CP06',
        'Period_CP07', 'Period_CP08', 'Period_CP09',
        'Period_CP10', 'Period_CP11', 'Period_CP12',
        'Period_CP13', 'Period_CP14', 'Period_CP15',
        'Period_CP16', 'Period_CP17', 'Period_CP18',        
        'Period_CP19', 'Period_CP20', 'Period_CP21',
        'Period_CP22', 'Period_CP23', 'Period_CP24',
        'Period_CP25', 'Period_CP26', 'Period_CP27',
        'Period_CP28', 'Period_CP29', 'Period_CP30'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP00E')| (melted['Prefix'] == 'CP00F')
                   | (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')
                   | (melted['Prefix'] == 'CP05') | (melted['Prefix'] == 'CP06')), 'Prefix'] = 'LD'
        
        melted.loc[((melted['Prefix'] == 'CP07') | (melted['Prefix'] == 'CP08') | (melted['Prefix'] == 'CP09')
                   | (melted['Prefix'] == 'CP10')| (melted['Prefix'] == 'CP11')| (melted['Prefix'] == 'CP12')
                   | (melted['Prefix'] == 'CP13') | (melted['Prefix'] == 'CP14') | (melted['Prefix'] == 'CP15') | (melted['Prefix'] == 'CP16') 
                   | (melted['Prefix'] == 'CP17') | (melted['Prefix'] == 'CP18')), 'Prefix'] = 'SHIFTS'
       
        melted.loc[((melted['Prefix'] == 'CP19') | (melted['Prefix'] == 'CP20') | (melted['Prefix'] == 'CP21')
                   | (melted['Prefix'] == 'CP22')| (melted['Prefix'] == 'CP23')| (melted['Prefix'] == 'CP24')
                   | (melted['Prefix'] == 'CP25') | (melted['Prefix'] == 'CP26') | (melted['Prefix'] == 'CP27') | (melted['Prefix'] == 'CP28') 
                   | (melted['Prefix'] == 'CP29') | (melted['Prefix'] == 'CP30')), 'Prefix'] = 'xLL'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('PERIOD COMBINE.csv')
        
        KW_BoxenPlot1(data=df, path=path, kind='boxen', name='Period')
        
        
        # all three groups
        df3 = df2[['RsqCP00A', 'RsqCP00B', 'RsqCP00C', 'RsqCP00D', 'RsqCP00E', 'RsqCP00F',
        'Rsq\\CP01', 'Rsq\\CP02', 'Rsq\\CP03',
        'Rsq\\CP04', 'Rsq\\CP05', 'Rsq\\CP06',
        'Rsq\\CP07', 'Rsq\\CP08', 'Rsq\\CP09',
        'Rsq\\CP10', 'Rsq\\CP11', 'Rsq\\CP12',
        'Rsq\\CP13', 'Rsq\\CP14', 'Rsq\\CP15',
        'Rsq\\CP16', 'Rsq\\CP17', 'Rsq\\CP18',        
        'Rsq\\CP19', 'Rsq\\CP20', 'Rsq\\CP21',
        'Rsq\\CP22', 'Rsq\\CP23', 'Rsq\\CP24',
        'Rsq\\CP25', 'Rsq\\CP26', 'Rsq\\CP27',
        'Rsq\\CP28', 'Rsq\\CP29', 'Rsq\\CP30']]
        # Rename columns
        df3.columns = ['Rsq_CP00A', 'Rsq_CP00B', 'Rsq_CP00C', 'Rsq_CP00D', 'Rsq_CP00E', 'Rsq_CP00F',
        'Rsq_CP01', 'Rsq_CP02', 'Rsq_CP03',
        'Rsq_CP04', 'Rsq_CP05', 'Rsq_CP06',
        'Rsq_CP07', 'Rsq_CP08', 'Rsq_CP09',
        'Rsq_CP10', 'Rsq_CP11', 'Rsq_CP12',
        'Rsq_CP13', 'Rsq_CP14', 'Rsq_CP15',
        'Rsq_CP16', 'Rsq_CP17', 'Rsq_CP18',        
        'Rsq_CP19', 'Rsq_CP20', 'Rsq_CP21',
        'Rsq_CP22', 'Rsq_CP23', 'Rsq_CP24',
        'Rsq_CP25', 'Rsq_CP26', 'Rsq_CP27',
        'Rsq_CP28', 'Rsq_CP29', 'Rsq_CP30']      
        melted = df3.melt(value_vars=['Rsq_CP00A', 'Rsq_CP00B', 'Rsq_CP00C', 'Rsq_CP00D', 'Rsq_CP00E', 'Rsq_CP00F',
        'Rsq_CP01', 'Rsq_CP02', 'Rsq_CP03',
        'Rsq_CP04', 'Rsq_CP05', 'Rsq_CP06',
        'Rsq_CP07', 'Rsq_CP08', 'Rsq_CP09',
        'Rsq_CP10', 'Rsq_CP11', 'Rsq_CP12',
        'Rsq_CP13', 'Rsq_CP14', 'Rsq_CP15',
        'Rsq_CP16', 'Rsq_CP17', 'Rsq_CP18',        
        'Rsq_CP19', 'Rsq_CP20', 'Rsq_CP21',
        'Rsq_CP22', 'Rsq_CP23', 'Rsq_CP24',
        'Rsq_CP25', 'Rsq_CP26', 'Rsq_CP27',
        'Rsq_CP28', 'Rsq_CP29', 'Rsq_CP30'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP00E')| (melted['Prefix'] == 'CP00F')
                   | (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')
                   | (melted['Prefix'] == 'CP05') | (melted['Prefix'] == 'CP06')), 'Prefix'] = 'LD'
        
        melted.loc[((melted['Prefix'] == 'CP07') | (melted['Prefix'] == 'CP08') | (melted['Prefix'] == 'CP09')
                   | (melted['Prefix'] == 'CP10')| (melted['Prefix'] == 'CP11')| (melted['Prefix'] == 'CP12')
                   | (melted['Prefix'] == 'CP13') | (melted['Prefix'] == 'CP14') | (melted['Prefix'] == 'CP15') | (melted['Prefix'] == 'CP16') 
                   | (melted['Prefix'] == 'CP17') | (melted['Prefix'] == 'CP18')), 'Prefix'] = 'SHIFTS'
       
        melted.loc[((melted['Prefix'] == 'CP19') | (melted['Prefix'] == 'CP20') | (melted['Prefix'] == 'CP21')
                   | (melted['Prefix'] == 'CP22')| (melted['Prefix'] == 'CP23')| (melted['Prefix'] == 'CP24')
                   | (melted['Prefix'] == 'CP25') | (melted['Prefix'] == 'CP26') | (melted['Prefix'] == 'CP27') | (melted['Prefix'] == 'CP28') 
                   | (melted['Prefix'] == 'CP29') | (melted['Prefix'] == 'CP30')), 'Prefix'] = 'xLL'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('Rsq COMBINE.csv')
        
        KW_BoxenPlot1(data=df, path=path, kind='boxen', name='Rsq')        


        # all three groups
        df3 = df2[['TrendCP00A', 'TrendCP00B', 'TrendCP00C', 'TrendCP00D', 'TrendCP00E', 'TrendCP00F',
        'Trend\\CP01', 'Trend\\CP02', 'Trend\\CP03',
        'Trend\\CP04', 'Trend\\CP05', 'Trend\\CP06',
        'Trend\\CP07', 'Trend\\CP08', 'Trend\\CP09',
        'Trend\\CP10', 'Trend\\CP11', 'Trend\\CP12',
        'Trend\\CP13', 'Trend\\CP14', 'Trend\\CP15',
        'Trend\\CP16', 'Trend\\CP17', 'Trend\\CP18',        
        'Trend\\CP19', 'Trend\\CP20', 'Trend\\CP21',
        'Trend\\CP22', 'Trend\\CP23', 'Trend\\CP24',
        'Trend\\CP25', 'Trend\\CP26', 'Trend\\CP27',
        'Trend\\CP28', 'Trend\\CP29', 'Trend\\CP30']]
        # Rename columns
        df3.columns = ['Trend_CP00A', 'Trend_CP00B', 'Trend_CP00C', 'Trend_CP00D', 'Trend_CP00E', 'Trend_CP00F',
        'Trend_CP01', 'Trend_CP02', 'Trend_CP03',
        'Trend_CP04', 'Trend_CP05', 'Trend_CP06',
        'Trend_CP07', 'Trend_CP08', 'Trend_CP09',
        'Trend_CP10', 'Trend_CP11', 'Trend_CP12',
        'Trend_CP13', 'Trend_CP14', 'Trend_CP15',
        'Trend_CP16', 'Trend_CP17', 'Trend_CP18',        
        'Trend_CP19', 'Trend_CP20', 'Trend_CP21',
        'Trend_CP22', 'Trend_CP23', 'Trend_CP24',
        'Trend_CP25', 'Trend_CP26', 'Trend_CP27',
        'Trend_CP28', 'Trend_CP29', 'Trend_CP30']      
        melted = df3.melt(value_vars=['Trend_CP00A', 'Trend_CP00B', 'Trend_CP00C', 'Trend_CP00D', 'Trend_CP00E', 'Trend_CP00F',
        'Trend_CP01', 'Trend_CP02', 'Trend_CP03',
        'Trend_CP04', 'Trend_CP05', 'Trend_CP06',
        'Trend_CP07', 'Trend_CP08', 'Trend_CP09',
        'Trend_CP10', 'Trend_CP11', 'Trend_CP12',
        'Trend_CP13', 'Trend_CP14', 'Trend_CP15',
        'Trend_CP16', 'Trend_CP17', 'Trend_CP18',        
        'Trend_CP19', 'Trend_CP20', 'Trend_CP21',
        'Trend_CP22', 'Trend_CP23', 'Trend_CP24',
        'Trend_CP25', 'Trend_CP26', 'Trend_CP27',
        'Trend_CP28', 'Trend_CP29', 'Trend_CP30'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Trend numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Trend'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Trend' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP00E')| (melted['Prefix'] == 'CP00F')
                   | (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')
                   | (melted['Prefix'] == 'CP05') | (melted['Prefix'] == 'CP06')), 'Prefix'] = 'LD'
        
        melted.loc[((melted['Prefix'] == 'CP07') | (melted['Prefix'] == 'CP08') | (melted['Prefix'] == 'CP09')
                   | (melted['Prefix'] == 'CP10')| (melted['Prefix'] == 'CP11')| (melted['Prefix'] == 'CP12')
                   | (melted['Prefix'] == 'CP13') | (melted['Prefix'] == 'CP14') | (melted['Prefix'] == 'CP15') | (melted['Prefix'] == 'CP16') 
                   | (melted['Prefix'] == 'CP17') | (melted['Prefix'] == 'CP18')), 'Prefix'] = 'SHIFTS'
       
        melted.loc[((melted['Prefix'] == 'CP19') | (melted['Prefix'] == 'CP20') | (melted['Prefix'] == 'CP21')
                   | (melted['Prefix'] == 'CP22')| (melted['Prefix'] == 'CP23')| (melted['Prefix'] == 'CP24')
                   | (melted['Prefix'] == 'CP25') | (melted['Prefix'] == 'CP26') | (melted['Prefix'] == 'CP27') | (melted['Prefix'] == 'CP28') 
                   | (melted['Prefix'] == 'CP29') | (melted['Prefix'] == 'CP30')), 'Prefix'] = 'xLL'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('Trend COMBINE.csv')
        
        KW_BoxenPlot1(data=df, path=path, kind='boxen', name='Trend')
        
        
        # all three groups
        df3 = df2[['DecayCP00A', 'DecayCP00B', 'DecayCP00C', 'DecayCP00D', 'DecayCP00E', 'DecayCP00F',
        'Decay\\CP01', 'Decay\\CP02', 'Decay\\CP03',
        'Decay\\CP04', 'Decay\\CP05', 'Decay\\CP06',
        'Decay\\CP07', 'Decay\\CP08', 'Decay\\CP09',
        'Decay\\CP10', 'Decay\\CP11', 'Decay\\CP12',
        'Decay\\CP13', 'Decay\\CP14', 'Decay\\CP15',
        'Decay\\CP16', 'Decay\\CP17', 'Decay\\CP18',        
        'Decay\\CP19', 'Decay\\CP20', 'Decay\\CP21',
        'Decay\\CP22', 'Decay\\CP23', 'Decay\\CP24',
        'Decay\\CP25', 'Decay\\CP26', 'Decay\\CP27',
        'Decay\\CP28', 'Decay\\CP29', 'Decay\\CP30']]
        # Rename columns
        df3.columns = ['Decay_CP00A', 'Decay_CP00B', 'Decay_CP00C', 'Decay_CP00D', 'Decay_CP00E', 'Decay_CP00F',
        'Decay_CP01', 'Decay_CP02', 'Decay_CP03',
        'Decay_CP04', 'Decay_CP05', 'Decay_CP06',
        'Decay_CP07', 'Decay_CP08', 'Decay_CP09',
        'Decay_CP10', 'Decay_CP11', 'Decay_CP12',
        'Decay_CP13', 'Decay_CP14', 'Decay_CP15',
        'Decay_CP16', 'Decay_CP17', 'Decay_CP18',        
        'Decay_CP19', 'Decay_CP20', 'Decay_CP21',
        'Decay_CP22', 'Decay_CP23', 'Decay_CP24',
        'Decay_CP25', 'Decay_CP26', 'Decay_CP27',
        'Decay_CP28', 'Decay_CP29', 'Decay_CP30']      
        melted = df3.melt(value_vars=['Decay_CP00A', 'Decay_CP00B', 'Decay_CP00C', 'Decay_CP00D', 'Decay_CP00E', 'Decay_CP00F',
        'Decay_CP01', 'Decay_CP02', 'Decay_CP03',
        'Decay_CP04', 'Decay_CP05', 'Decay_CP06',
        'Decay_CP07', 'Decay_CP08', 'Decay_CP09',
        'Decay_CP10', 'Decay_CP11', 'Decay_CP12',
        'Decay_CP13', 'Decay_CP14', 'Decay_CP15',
        'Decay_CP16', 'Decay_CP17', 'Decay_CP18',        
        'Decay_CP19', 'Decay_CP20', 'Decay_CP21',
        'Decay_CP22', 'Decay_CP23', 'Decay_CP24',
        'Decay_CP25', 'Decay_CP26', 'Decay_CP27',
        'Decay_CP28', 'Decay_CP29', 'Decay_CP30'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Decay numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Decay'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Decay' to create new column names       
        melted.loc[((melted['Prefix'] == 'CP00A') | (melted['Prefix'] == 'CP00B') | (melted['Prefix'] == 'CP00C')
                   | (melted['Prefix'] == 'CP00D')| (melted['Prefix'] == 'CP00E')| (melted['Prefix'] == 'CP00F')
                   | (melted['Prefix'] == 'CP01')| (melted['Prefix'] == 'CP02')
                   | (melted['Prefix'] == 'CP03') | (melted['Prefix'] == 'CP04')
                   | (melted['Prefix'] == 'CP05') | (melted['Prefix'] == 'CP06')), 'Prefix'] = 'LD'
        
        melted.loc[((melted['Prefix'] == 'CP07') | (melted['Prefix'] == 'CP08') | (melted['Prefix'] == 'CP09')
                   | (melted['Prefix'] == 'CP10')| (melted['Prefix'] == 'CP11')| (melted['Prefix'] == 'CP12')
                   | (melted['Prefix'] == 'CP13') | (melted['Prefix'] == 'CP14') | (melted['Prefix'] == 'CP15') | (melted['Prefix'] == 'CP16') 
                   | (melted['Prefix'] == 'CP17') | (melted['Prefix'] == 'CP18')), 'Prefix'] = 'SHIFTS'
       
        melted.loc[((melted['Prefix'] == 'CP19') | (melted['Prefix'] == 'CP20') | (melted['Prefix'] == 'CP21')
                   | (melted['Prefix'] == 'CP22')| (melted['Prefix'] == 'CP23')| (melted['Prefix'] == 'CP24')
                   | (melted['Prefix'] == 'CP25') | (melted['Prefix'] == 'CP26') | (melted['Prefix'] == 'CP27') | (melted['Prefix'] == 'CP28') 
                   | (melted['Prefix'] == 'CP29') | (melted['Prefix'] == 'CP30')), 'Prefix'] = 'xLL'

        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)  
        df.to_csv('Decay COMBINE.csv')
        
        KW_BoxenPlot1(data=df, path=path, kind='boxen', name='Decay')  
        """
        
        
        # FOR RNAseq ChP paper v3 - here it separates explants to 3 groups and combines them for statistics
        """
        df3 = df2[['Period\CHP1', 'Period\CHP2', 'Period\CHP3', 'Period\CHP4', 'Period\CHP5', 'Period\CHP6', 'Period\CHP7', 'Period\CHP8', 'Period\CHP9', 'Period\CHPa', 'Period\CHPb', 'Period\CHPc']]
        # Rename columns
        df3.columns = ['Period_CHP1', 'Period_CHP2', 'Period_CHP3', 'Period_CHP4', 'Period_CHP5', 'Period_CHP6', 'Period_CHP7', 'Period_CHP8', 'Period_CHP9', 'Period_CHPa', 'Period_CHPb', 'Period_CHPc']        
        melted = df3.melt(value_vars=['Period_CHP1', 'Period_CHP2', 'Period_CHP3', 'Period_CHP4', 'Period_CHP5', 'Period_CHP6', 'Period_CHP7', 'Period_CHP8', 'Period_CHP9', 'Period_CHPa', 'Period_CHPb', 'Period_CHPc'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and period numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Period'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Period' to create new column names       
        melted.loc[((melted['Prefix'] == 'CHP1') | (melted['Prefix'] == 'CHP2') | (melted['Prefix'] == 'CHP3')), 'Prefix'] = 'Control'
        melted.loc[((melted['Prefix'] == 'CHP4') | (melted['Prefix'] == 'CHP5') | (melted['Prefix'] == 'CHP6')), 'Prefix'] = 'SCNx'
        melted.loc[((melted['Prefix'] == 'CHP7') | (melted['Prefix'] == 'CHP8') | (melted['Prefix'] == 'CHP9') | (melted['Prefix'] == 'CHPa') | (melted['Prefix'] == 'CHPb') | (melted['Prefix'] == 'CHPc')), 'Prefix'] = 'SCNx+Dex'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Period COMBINE.csv')
        KW_BoxenPlot1(data=df, title= 'Period of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Period')
        KW_BoxenPlot1(data=df, title= 'Period of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Period')
        
        
        df3 = df2[['Amplitude\CHP1', 'Amplitude\CHP2', 'Amplitude\CHP3', 'Amplitude\CHP4', 'Amplitude\CHP5', 'Amplitude\CHP6', 'Amplitude\CHP7', 'Amplitude\CHP8', 'Amplitude\CHP9', 'Amplitude\CHPa', 'Amplitude\CHPb', 'Amplitude\CHPc']]
        # Rename columns
        df3.columns = ['Amplitude_CHP1', 'Amplitude_CHP2', 'Amplitude_CHP3', 'Amplitude_CHP4', 'Amplitude_CHP5', 'Amplitude_CHP6', 'Amplitude_CHP7', 'Amplitude_CHP8', 'Amplitude_CHP9', 'Amplitude_CHPa', 'Amplitude_CHPb', 'Amplitude_CHPc']        
        melted = df3.melt(value_vars=['Amplitude_CHP1', 'Amplitude_CHP2', 'Amplitude_CHP3', 'Amplitude_CHP4', 'Amplitude_CHP5', 'Amplitude_CHP6', 'Amplitude_CHP7', 'Amplitude_CHP8', 'Amplitude_CHP9', 'Amplitude_CHPa', 'Amplitude_CHPb', 'Amplitude_CHPc'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Amplitude numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Amplitude'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Amplitude' to create new column names       
        melted.loc[((melted['Prefix'] == 'CHP1') | (melted['Prefix'] == 'CHP2') | (melted['Prefix'] == 'CHP3')), 'Prefix'] = 'Control'
        melted.loc[((melted['Prefix'] == 'CHP4') | (melted['Prefix'] == 'CHP5') | (melted['Prefix'] == 'CHP6')), 'Prefix'] = 'SCNx'
        melted.loc[((melted['Prefix'] == 'CHP7') | (melted['Prefix'] == 'CHP8') | (melted['Prefix'] == 'CHP9') | (melted['Prefix'] == 'CHPa') | (melted['Prefix'] == 'CHPb') | (melted['Prefix'] == 'CHPc')), 'Prefix'] = 'SCNx+Dex'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Amplitude COMBINE.csv')
        KW_BoxenPlot1(data=df, title= 'Amplitude of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Amplitude')
        KW_BoxenPlot1(data=df, title= 'Amplitude of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Amplitude')
        
        
        df3 = df2[['Phase\CHP1', 'Phase\CHP2', 'Phase\CHP3', 'Phase\CHP4', 'Phase\CHP5', 'Phase\CHP6', 'Phase\CHP7', 'Phase\CHP8', 'Phase\CHP9', 'Phase\CHPa', 'Phase\CHPb', 'Phase\CHPc']]
        # Rename columns
        df3.columns = ['Phase_CHP1', 'Phase_CHP2', 'Phase_CHP3', 'Phase_CHP4', 'Phase_CHP5', 'Phase_CHP6', 'Phase_CHP7', 'Phase_CHP8', 'Phase_CHP9', 'Phase_CHPa', 'Phase_CHPb', 'Phase_CHPc']        
        melted = df3.melt(value_vars=['Phase_CHP1', 'Phase_CHP2', 'Phase_CHP3', 'Phase_CHP4', 'Phase_CHP5', 'Phase_CHP6', 'Phase_CHP7', 'Phase_CHP8', 'Phase_CHP9', 'Phase_CHPa', 'Phase_CHPb', 'Phase_CHPc'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Phase numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Phase'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Phase' to create new column names       
        melted.loc[((melted['Prefix'] == 'CHP1') | (melted['Prefix'] == 'CHP2') | (melted['Prefix'] == 'CHP3')), 'Prefix'] = 'Control'
        melted.loc[((melted['Prefix'] == 'CHP4') | (melted['Prefix'] == 'CHP5') | (melted['Prefix'] == 'CHP6')), 'Prefix'] = 'SCNx'
        melted.loc[((melted['Prefix'] == 'CHP7') | (melted['Prefix'] == 'CHP8') | (melted['Prefix'] == 'CHP9') | (melted['Prefix'] == 'CHPa') | (melted['Prefix'] == 'CHPb') | (melted['Prefix'] == 'CHPc')), 'Prefix'] = 'SCNx+Dex'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Phase COMBINE.csv')
        KW_BoxenPlot1(data=df, title= 'Phase of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Phase')
        KW_BoxenPlot1(data=df, title= 'Phase of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Phase')
        
        
        df3 = df2[['Trend\CHP1', 'Trend\CHP2', 'Trend\CHP3', 'Trend\CHP4', 'Trend\CHP5', 'Trend\CHP6', 'Trend\CHP7', 'Trend\CHP8', 'Trend\CHP9', 'Trend\CHPa', 'Trend\CHPb', 'Trend\CHPc']]
        # Rename columns
        df3.columns = ['Trend_CHP1', 'Trend_CHP2', 'Trend_CHP3', 'Trend_CHP4', 'Trend_CHP5', 'Trend_CHP6', 'Trend_CHP7', 'Trend_CHP8', 'Trend_CHP9', 'Trend_CHPa', 'Trend_CHPb', 'Trend_CHPc']        
        melted = df3.melt(value_vars=['Trend_CHP1', 'Trend_CHP2', 'Trend_CHP3', 'Trend_CHP4', 'Trend_CHP5', 'Trend_CHP6', 'Trend_CHP7', 'Trend_CHP8', 'Trend_CHP9', 'Trend_CHPa', 'Trend_CHPb', 'Trend_CHPc'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Trend numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Trend'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Trend' to create new column names       
        melted.loc[((melted['Prefix'] == 'CHP1') | (melted['Prefix'] == 'CHP2') | (melted['Prefix'] == 'CHP3')), 'Prefix'] = 'Control'
        melted.loc[((melted['Prefix'] == 'CHP4') | (melted['Prefix'] == 'CHP5') | (melted['Prefix'] == 'CHP6')), 'Prefix'] = 'SCNx'
        melted.loc[((melted['Prefix'] == 'CHP7') | (melted['Prefix'] == 'CHP8') | (melted['Prefix'] == 'CHP9') | (melted['Prefix'] == 'CHPa') | (melted['Prefix'] == 'CHPb') | (melted['Prefix'] == 'CHPc')), 'Prefix'] = 'SCNx+Dex'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Trend COMBINE.csv')
        KW_BoxenPlot1(data=df, title= 'Mesor of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Trend')
        KW_BoxenPlot1(data=df, title= 'Mesor of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Trend')
        
        
        df3 = df2[['Rsq\CHP1', 'Rsq\CHP2', 'Rsq\CHP3', 'Rsq\CHP4', 'Rsq\CHP5', 'Rsq\CHP6', 'Rsq\CHP7', 'Rsq\CHP8', 'Rsq\CHP9', 'Rsq\CHPa', 'Rsq\CHPb', 'Rsq\CHPc']]
        # Rename columns
        df3.columns = ['Rsq_CHP1', 'Rsq_CHP2', 'Rsq_CHP3', 'Rsq_CHP4', 'Rsq_CHP5', 'Rsq_CHP6', 'Rsq_CHP7', 'Rsq_CHP8', 'Rsq_CHP9', 'Rsq_CHPa', 'Rsq_CHPb', 'Rsq_CHPc']        
        melted = df3.melt(value_vars=['Rsq_CHP1', 'Rsq_CHP2', 'Rsq_CHP3', 'Rsq_CHP4', 'Rsq_CHP5', 'Rsq_CHP6', 'Rsq_CHP7', 'Rsq_CHP8', 'Rsq_CHP9', 'Rsq_CHPa', 'Rsq_CHPb', 'Rsq_CHPc'],
                          var_name='Var', value_name='Value')              
        # Extract the 'CHP' and Rsq numbers from the 'Var' column
        melted['Prefix'] = melted['Var'].str.split('_').str[1]
        melted['Rsq'] = melted['Var'].str.split('_').str[0]
        # Combine 'Prefix' and 'Rsq' to create new column names       
        melted.loc[((melted['Prefix'] == 'CHP1') | (melted['Prefix'] == 'CHP2') | (melted['Prefix'] == 'CHP3')), 'Prefix'] = 'Control'
        melted.loc[((melted['Prefix'] == 'CHP4') | (melted['Prefix'] == 'CHP5') | (melted['Prefix'] == 'CHP6')), 'Prefix'] = 'SCNx'
        melted.loc[((melted['Prefix'] == 'CHP7') | (melted['Prefix'] == 'CHP8') | (melted['Prefix'] == 'CHP9') | (melted['Prefix'] == 'CHPa') | (melted['Prefix'] == 'CHPb') | (melted['Prefix'] == 'CHPc')), 'Prefix'] = 'SCNx+Dex'
        melted = melted[['Prefix', 'Value']].dropna()        
        # Pivot the dataframe and create the new dataframe 'df'
        df = melted.pivot(columns='Prefix', values='Value')  #.reset_index(drop=True)       
        df.to_csv('Rsq COMBINE.csv')
        KW_BoxenPlot1(data=df, title= 'Rsq of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='violin', name='Rsq')
        KW_BoxenPlot1(data=df, title= 'Rsq of all cells', path=path, ticks=df.columns, remove_outliers=True, iqr_value = 2.22, ylim=(None, None), kind='boxen', name='Rsq') 
        
        """        