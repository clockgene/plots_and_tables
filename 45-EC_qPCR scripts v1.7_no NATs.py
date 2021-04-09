import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from tkinter import filedialog
from tkinter import *

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.api import anova_lm
from statsmodels.graphics.api import interaction_plot, abline_plot

####### Define functions  ###########
#####################################

#to manipulate data (normalize to max value of profil)
def normalize(profilvalues):                                           
    return profilvalues/profilvalues.max()

#to fit with scipy, period=24, degrees of freedom is cca. for adr it is (n - parameters) = 34-3=31
def cosinor(x, baseline, amplitude, phase):                      
    return baseline + amplitude*np.cos(2*np.pi*(x-phase)/24)

"""    
#alternative cosinor or other periodic function, does not work so far
def cosinor_v2(x, baseline, amplitude, phase, trend, period):   # , damp , period
    #return baseline + amplitude*(np.exp(-damp*x))*np.cos(2*np.pi*(x-phase)/24) + trend*x
    return baseline + amplitude*np.cos(2*np.pi*(x-phase)/period) + trend*x

#horizontal or other basic function for statistics, not used, instead manual calc of mean value line
def horizontal(x):
    return np.mean(x) + 0*x
"""

# to remove negative phase after FFT
def phase_correct(x):    
    if x < 0:
        return x + 24
    else:
        return x

# to solve FFT cosinor sometimes returning (phase-period/2), compares max mean values vs phase/phase+12 and choose , still NOT WORKING 100% maybe use find peaks function?
def phase_correct_2(phase, phase_12, maxv):                         
    if maxv > 12 and phase < 12:
        comp1 = maxv - (phase + 24)
        if np.absolute(comp1) > np.absolute(maxv - phase_12):
            return phase_12
        else:
            return phase
    if maxv < 12 and phase > 12:
        comp2 = (maxv + 24) - phase
        if np.absolute(comp2) > np.absolute(maxv - phase_12):
            return phase_12
        else:
            return phase
    if maxv < 12 and phase_12 > 12:
        comp3 = (maxv + 24) - phase_12
        if np.absolute(comp3) < np.absolute(maxv - phase):
            return phase_12
        else:
            return phase
    else:
        if np.absolute(maxv - phase) > np.absolute(maxv - phase_12):
            return phase_12
        else:
            return phase

# to change CT to polar coordinates for polar plotting
# 1h = (2/24)*np.pi = (1/12)*np.pi,   circumference = 2*np.pi*radius
def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x-24)/12)*np.pi   
    return r

def array_to_number(array_like):
    if len(array_like) == 0:
        return float(np.nan)
    else:
        return float(array_like)


##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button

root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = Button(text="Browse folder", command=browse_button)
buttonBrowse.grid()
mainloop()

#############################################################################################################################################

#############################################
######### Original Data Loading #############
#############################################
df_orig = pd.read_csv('.\EC_RTqPCR.csv', delimiter = ',')
df0 = df_orig.fillna(value=0) #problem with nans, solved by zeros for normalization and by filtering-out zeros for cosin/plot
#os.chdir('.\output') #save all to output subdir if needed

dfi = df0.set_index(['tissue', 'food', 'CT', 'sample']).sort_index() #create indexed dataframe for easy calling via loc
df1 = df0.copy()  #copy for different normalization type

################################################
############ Dataframes processing #############
################################################
gene = list(df0.columns[4:].values)   #create list of raw genes for iteration over them
food_regime = ['LD', 'RF']
bytissue = df0.groupby('tissue')
tissues = []
for tissue_group, frame in bytissue:
    tissues = tissues + [tissue_group]    # alternative way to create list of subindexes for iteration
    

######### df0 - normalized to max LD and to max RF values ############################################
for tissue_group in tissues:
    for food in food_regime:
        for gen in gene:
            try:
                df0.loc[(df0.tissue == tissue_group) & (df0.food == food), f'{gen}_rel'] = normalize(dfi.loc[(tissue_group, food),[gen]].values)
            except:
                KeyError
         
######### df1 - normalized to max of LD+RF values ###########################################################
for tissue_group in tissues:
    for gen in gene:
        df1.loc[df1.tissue == tissue_group, f'{gen}_rel'] = normalize(dfi.loc[[tissue_group],[gen]].values)    #working,  but not with nans, uses multi-indexed dfi 
           


# both df0 and df1 contain zeroes from normal od NaN values, otherwise there should not be any zero - drop them for mean calculations only
df0_i = df0.set_index(['tissue', 'food', 'CT', 'sample']).sort_index()   #set index to avoid dropping CT0..., sort_index() for performance gain?
df00 = df0_i[df0_i != 0].reset_index().sort_index()                                  # alt>>> df0_i[df0_i.astype(bool)].reset_index() 
df1_i = df1.set_index(['tissue', 'food', 'CT', 'sample']).sort_index()
df11 = df1_i[df1_i != 0].reset_index().sort_index()


######### df_mean - mean/SD/SEM values for plotting ######################################################
gene_all = list(df0.columns[4:].values)     #create list of raw+norm genes for iteration over them
dflist = []
dflist1 = []
for i in gene_all:   
    dflist.append(pd.DataFrame(df00.groupby(['tissue', 'food', 'CT']).agg(['mean', 'std', 'sem'])[[i]].reset_index()))
    dflist1.append(pd.DataFrame(df11.groupby(['tissue', 'food', 'CT']).agg(['mean', 'std', 'sem'])[[i]].reset_index()))

df_m = dflist[0]
df_m1 = dflist1[0]
x_index = 1
for x in range(len(gene_all[1:])):  #x will be 0, 1, ...50     
    df_m = pd.merge(df_m, dflist[x_index], on=['tissue', 'food', 'CT']).set_index(['tissue', 'food', 'CT'])
    df_m1 = pd.merge(df_m1, dflist1[x_index], on=['tissue', 'food', 'CT']).set_index(['tissue', 'food', 'CT'])
    x_index += 1

df_mean = df_m[df_m != 0]   #drop all zeros from data, that needed to be there for normalization   NOT NECESSARY ?
df_mean1 = df_m1[df_m1 != 0] 

print('df_mean is this:')
print(df_mean.head())
print()


####### df_heat - dataframes for heatmap plots ####################################
df_mean_stack = df_mean.stack(level=0)           #stacks level 0 column labels (=gene names) to a new unnamed column
#df_mean_stack = df_mean1.stack(level=0)        # choose this instead of the above if you want values norm to max of LD+RF

df_mean_stack_r = df_mean_stack.reset_index().rename(columns={'level_3':'Gene'})    #renames the new column to Gene

# this selects only those rows that have _rel in string, no need for regex, then sets new indexes #
df_heat = df_mean_stack_r[df_mean_stack_r['Gene'].str.contains("_rel")].reset_index().set_index(['tissue', 'food', 'Gene', 'CT']).sort_index()   

print('df_heat is this:')
print(df_heat.head())
print()


####### df_mean_cat - Adding Gene Categories ##############
dfh = df_mean.stack([0, 1]).reset_index().rename(columns={'level_3':'Gene', 'level_4':'Aggr', 0:'Expression'})   #alternative
#dfh = df_mean1.stack([0, 1]).reset_index().rename(columns={'level_3':'Gene', 'level_4':'Aggr', 0:'Expression'}) #norm to max of LD+RF
"""
dfh.loc[dfh.Gene.str.contains("Arntl"), 'function'] = 'Clock'
dfh.loc[dfh.Gene.str.contains("Nr1d1"), 'function'] = 'Clock'
dfh.loc[dfh.Gene.str.contains("Per2"), 'function'] = 'Clock'
dfh.loc[dfh.Gene.str.contains("Cnr"), 'function'] = 'Receptor'
dfh.loc[dfh.Gene.str.contains("Trpv"), 'function'] = 'Receptor'
dfh.loc[dfh.Gene.str.contains("Ppar"), 'function'] = 'Receptor'
dfh.loc[dfh.Gene.str.contains("Gpr"), 'function'] = 'Receptor'
dfh.loc[dfh.Gene.str.contains("Nape"), 'function'] = 'Synthesis'
dfh.loc[dfh.Gene.str.contains("Abhd4"), 'function'] = 'Synthesis'
dfh.loc[dfh.Gene.str.contains("Abhd6"), 'function'] = 'Degradation'
dfh.loc[dfh.Gene.str.contains("Dagl"), 'function'] = 'Synthesis'
dfh.loc[dfh.Gene.str.contains("Gde"), 'function'] = 'Synthesis'
dfh.loc[dfh.Gene.str.contains("Nat"), 'function'] = 'NATs'
dfh.loc[dfh.Gene.str.contains("Naa60"), 'function'] = 'NATs'
dfh.loc[dfh.Gene.str.contains("Mgl"), 'function'] = 'Degradation'
dfh.loc[dfh.Gene.str.contains("Faah"), 'function'] = 'Degradation'
dfh.loc[dfh.Gene.str.contains("Thop"), 'function'] = 'Transport/Other'  
dfh.loc[dfh.Gene.str.contains("Fabp"), 'function'] = 'Transport/Other'
"""
dfh.loc[dfh.Gene.str.contains("Arntl"), 'function'] = '1.Clock'
dfh.loc[dfh.Gene.str.contains("Nr1d1"), 'function'] = '1.Clock'
dfh.loc[dfh.Gene.str.contains("Per2"), 'function'] = '1.Clock'
dfh.loc[dfh.Gene.str.contains("Cnr"), 'function'] = '2.Receptor'
dfh.loc[dfh.Gene.str.contains("Trpv"), 'function'] = '2.Receptor'
dfh.loc[dfh.Gene.str.contains("Ppar"), 'function'] = '2.Receptor'
dfh.loc[dfh.Gene.str.contains("Gpr"), 'function'] = '2.Receptor'
dfh.loc[dfh.Gene.str.contains("Nape"), 'function'] = '3.Synthesis'
dfh.loc[dfh.Gene.str.contains("Abhd4"), 'function'] = '3.Synthesis'
dfh.loc[dfh.Gene.str.contains("Abhd6"), 'function'] = '4.Catabolism'
dfh.loc[dfh.Gene.str.contains("Dagl"), 'function'] = '3.Synthesis'
dfh.loc[dfh.Gene.str.contains("Gde"), 'function'] = '3.Synthesis'
dfh.loc[dfh.Gene.str.contains("Nat"), 'function'] = '6.NATs'
dfh.loc[dfh.Gene.str.contains("Naa60"), 'function'] = '6.NATs'
dfh.loc[dfh.Gene.str.contains("Mgl"), 'function'] = '4.Catabolism'
dfh.loc[dfh.Gene.str.contains("Faah"), 'function'] = '4.Catabolism'
dfh.loc[dfh.Gene.str.contains("Thop"), 'function'] = '5.Transport'  
dfh.loc[dfh.Gene.str.contains("Fabp"), 'function'] = '5.Transport'

df_mean_cat = dfh.set_index(['tissue', 'food', 'function', 'Gene', 'Aggr', 'CT']).sort_index()     #alternative

#alternative
#first set row multiindex, than unstack rows func and gene to column-multiindex, then get rid of unnecessary 'index' columns
#df_mean_cat = dfh.set_index(['function', 'Gene', 'tissue', 'food', 'CT']).unstack(['function', 'Gene']).drop('index', axis=1, level=0)

###############################################


###############################################
##### curve fits for all profiles #############
###############################################


######## Initial Guess Parameters #######
p0=[0.5, 0.5, 4.0]     # for cosinor(baseline, amplitude, phase)  ,trend was removed
#p0=[0.5, 0.5, 4.0, 0.0, 24]     # for cosinor_v2 (baseline, amplitude, phase, trend, period/damp)

####### Prepare lists for data and choose which data to fit #############
df = df1.copy() # choose df0 if want fit to data norm separately to LD and to RF
df2 = df_mean_cat.reset_index().copy()    # for getting max value for each profile

norm_gene = list(df.columns[30:].values)
df_cos = pd.DataFrame()
profile = []
Mesor = []
Mesor_SD = []
Amplitude = []
Amp_SD = []
Phase = []
Phase_SD = []
Phase_SEM = []
R2 = []
missing_samples = []
not_expressed = []
horizont = []
Ftest_res = []
pValue = []
maxValue = []
tissue_id = []
food_id = []
norm_gen_id = []
Peaks = []

####### Fit curve and calculate parameters and compare cos vs horiz by F test ##########
for i in tissues:    
    for food in food_regime:        
        for norm_gen in norm_gene:
            xdata0 = df.loc[(df['tissue'] == i) & (df['food'] == food), ['CT']].values.flatten()    #try - instead of values, put CT outside []
            ydata0 = df.loc[(df['tissue'] == i) & (df['food'] == food) , [norm_gen]].values.flatten()
            newindex = ~((ydata0 == 0))     #remove zeroes, the were substitute for nan, but would change result of cosinor
            xdata = xdata0[newindex]            
            ydata = ydata0[newindex]
            maxValue.append(df2.loc[(df2['Expression'] == (df2.loc[(df2['tissue'] == i) & (df2['food'] == food) & (df2['Gene'] == norm_gen) & (df2['Aggr'] == 'mean'), 'Expression'].max())), ['tissue', 'food', 'function', 'Gene', 'Aggr', 'CT', 'Expression']])  #create list of df with CT of max mean expression
            tissue_id.append(i)                 # for MultiIndex
            food_id.append(food)                # for MultiIndex
            norm_gen_id.append(norm_gen)        # for MultiIndex
            try:                                                    #strange problem when using if/else, this is common workaround to get through NaN
                profile.append(f'{i}_{food}_{norm_gen}')
                fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor, xdata, ydata, p0, full_output=1, maxfev=1000000)                            #v1 cosinor curve
                #fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor_v2, xdata, ydata, p0, full_output=1, maxfev=1000000)                        #v2
                ss_res = sum((ydata - cosinor(xdata, fitParams[0], fitParams[1], fitParams[2]))**2)   #residual sum of squares                #v1, fitParams[3] removed
                #ss_res = sum((ydata - cosinor_v2(xdata, fitParams[0], fitParams[1], fitParams[2], fitParams[3], fitParams[4]))**2)   #residual sum of squares  #v2     
                ss_tot = sum((ydata - np.mean(ydata))**2)       #total sum of squares,  https://www.graphpad.com/guides/prism/7/curve-fitting/r2_ameasureofgoodness_of_fitoflinearregression.htm?toc=0&printWindow
                r_squared = 1 - (ss_res/ss_tot)            
                Mesor.append(float(f'{fitParams[0]}'))
                Amplitude.append(abs(float(f'{fitParams[1]}')))
                Phase.append(phase_correct(float(f'{fitParams[2]}')))                       #add 24 via function, only if phase is negative
                R2.append(float(f'{r_squared}'))
                Mesor_SD.append(np.sqrt(float(f'{fitCovariances[0, 0]}')))                  #np.sqrt(np.diag(fitCovariances)) calculates SD of parameter
                Amp_SD.append(np.sqrt(float(f'{fitCovariances[1, 1]}')))    
                Phase_SD.append(np.sqrt(float(f'{fitCovariances[2, 2]}')))                  #STDEV()/SQRT(COUNT())
                Phase_SEM.append((np.sqrt(float(f'{fitCovariances[2, 2]}')))/np.sqrt(34))    # not sure about N for SEM calc, is Std.Error SEM or SD in prism?
                ss_res_h = ss_tot                                               #horizontal line residuals is same as ss_tot
                Fvalue = ((ss_res_h - ss_res)/ss_res)/((34-32)/32)              #/(DF1-DF2)/DF2 - degreees of freedom for horizontal-cosinor model/cosinor, check Prism if in doubt
                Ftest_res.append(Fvalue)  #F value - the closer to 1, the more likely simple model is correct and not cosinor, https://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?REG_Comparing_Models_Tab.htm
                pVal = (1-stats.f.cdf(Fvalue, 2, 32))           #calculate two-tailed p values of cosinor vs horizontal, formula = (1 - scipy.stats.f.cdf(F,DFnumerator,DFdenominator))
                pValue.append(float(pVal))                      
                x_fit = np.linspace(np.min(xdata), np.max(xdata), 1000)
                y_fit = cosinor(x_fit, fitParams[0], fitParams[1], fitParams[2])    # , fitParams[3] was removed
                height = y_fit.min() + (y_fit.max() - y_fit.min())/3
                peak_time, peak_height = find_peaks(y_fit, height=height, distance=24)
                Peaks.append(array_to_number(x_fit[peak_time]))
            except ValueError:
                print(f'{i}_{food}_{norm_gen}_is not expressed')
                not_expressed.append(f'{i}_{food}_{norm_gen}')
                Mesor.append(np.nan)
                Amplitude.append(np.nan)
                Phase.append(np.nan)   
                R2.append(np.nan)
                Mesor_SD.append(np.nan) 
                Amp_SD.append(np.nan)    
                Phase_SD.append(np.nan) 
                Phase_SEM.append(np.nan)    
                Ftest_res.append(np.nan)  
                pValue.append(np.nan)
                Peaks.append(np.nan)
            except TypeError:
                print(f'{i}_{food}_{norm_gen}_is missing')
                #missing_samples.append(f'{i}_{food}')
                missing_samples.append(f'{i}_{food}_{norm_gen}')
                Mesor.append(np.nan)
                Amplitude.append(np.nan)
                Phase.append(np.nan)   
                R2.append(np.nan)
                Mesor_SD.append(np.nan)  
                Amp_SD.append(np.nan)    
                Phase_SD.append(np.nan) 
                Phase_SEM.append(np.nan)    
                Ftest_res.append(np.nan)  
                pValue.append(np.nan)
                Peaks.append(np.nan)
            except IndexError:
                Peaks.append(np.nan)

df_cos['profile'] = profile
df_cos['Mesor'] = Mesor
df_cos['Mesor_SD'] = Mesor_SD
df_cos['Amplitude'] = Amplitude
df_cos['Amp_SD'] = Amp_SD
df_cos['Rel_Amp_Error'] = df_cos['Amp_SD']/df_cos['Amplitude']
df_cos['Phase'] = Phase
df_cos['Phase_SD'] = Phase_SD
df_cos['R2'] = R2
df_cos['P_value'] = pValue
df_cos['Phase_SEM'] = Phase_SEM
df_cos['Phase_12'] = df_cos['Phase'] + 12                                       #alternative Phase for FFT Cosinor problem
df_cos['Peaks'] = Peaks

arrays = [tissue_id, food_id, norm_gen_id]                                      #for MultiIndex
index = pd.MultiIndex.from_arrays(arrays, names=('tissue', 'food', 'Gene'))     #for MultiIndex
df_cos2 = df_cos.set_index(index)                                               #this is df_cos with hierarchical multiindex

empty = pd.DataFrame()
df_max_values = empty.append(maxValue)                                          #creates df with CTs and Expr values of max mean of expression

df_cos_max = (pd.merge(df_max_values, df_cos2, on=['tissue', 'food', 'Gene']).set_index(['tissue', 'food', 'Gene'])).drop(columns=['Aggr']) #merge cos and max to single dataframe

df_cos_max['Phase_final'] = df_cos_max.apply(lambda x: phase_correct_2(x['Phase'], x['Phase_12'], x['CT']), axis=1)     #apply Phase correcting function to remediate FFT Cosinor problem
# Final Phase may still be wrong, needs manual checking, compare Peaks, Max CT, Phase

#####################################################################
######## 1W ANOVA of individual LD or RF profiles for each gene #####
#####################################################################
"""
#To check details, use 
print(model.summary())
"""

dfs = df0.copy().set_index(['tissue', 'food', 'CT']).sort_index()                   # to calculate with values normalized separately to LD max and to RF max

not_expressed_s = []
profile_ow = []
P_ow = []
F_ow = []

for i in tissues:    
    for food in food_regime:        
        for norm_gen in norm_gene:
            try:
                profile_ow.append(f'{i}_{food}_{norm_gen}')
                model = ols(f'{norm_gen} ~ C(CT)', dfs.loc[(i, food), norm_gen][dfs.loc[(i, food), norm_gen].astype(bool)].reset_index()).fit()    # 1-way ANOVA, exclude 0 by bool index
                table = anova_lm(model)                             # 1-way ANOVA result
                P_ow.append(table.loc['C(CT)', 'PR(>F)'])           # adds 1w ANOVA p value to list
                F_ow.append(table.loc['C(CT)', 'F'])                # adds 1w ANOVA F value to list
               
            except ValueError:
                print(f'{i}_{food}_{norm_gen}_is not expressed')
                not_expressed_s.append(f'{i}_{food}_{norm_gen}')
                P_ow.append(np.nan)
                F_ow.append(np.nan)
            except KeyError:
                print(f'{i}_{food}_{norm_gen}_is missing')
                not_expressed_s.append(f'{i}_{food}_{norm_gen}')
                P_ow.append(np.nan)
                F_ow.append(np.nan)
                

df_stat = pd.DataFrame()
df_stat['profile'] = profile_ow 
df_stat['P_1w_ANOVA'] = P_ow
df_stat['F_1w_ANOVA'] = F_ow

#arrays = [tissue_id, food_id, norm_gen_id]                                                 #for MultiIndex, use the one created for df_cos, it is the same
#index = pd.MultiIndex.from_arrays(arrays, names=('tissue', 'food', 'Gene'))                #for MultiIndex, use the one created for df_cos, it is the same
df_stat_i = df_stat.set_index(index)                                                         #this is df_stat with hierarchical multiindex
df_cos_max_stat = pd.merge(df_cos_max, df_stat_i, on=['tissue', 'food', 'Gene', 'profile'])  #merge to single dataframe


#####################################################################
######## 2W ANOVA of LD vs RF and interaction with CT ###############
#####################################################################
"""
#To check details, use 
print(model2.summary())
"""

dfs1 = df1.copy().set_index(['tissue', 'food', 'CT']).sort_index()               # to calculate with values normalized to max of LD and RF 

not_expressed_s2 = []
profile_tw = []
P_tw_food = []
F_tw_food = []
P_tw_int = []
F_tw_int = []

for i in tissues:    
    for food in food_regime:        
        for norm_gen in norm_gene:
            try:
                profile_tw.append(f'{i}_{food}_{norm_gen}') 
                model2 = ols(f'{norm_gen} ~ C(food)*C(CT)', dfs1.loc[i, norm_gen][dfs1.loc[i, norm_gen].astype(bool)].reset_index()).fit()      # 2-way ANOVA, exclude 0 by bool index
                table2 = anova_lm(model2)
                P_tw_food.append(table2.loc['C(food)', 'PR(>F)'])             # adds 2w ANOVA food p value  to list
                F_tw_food.append(table2.loc['C(food)', 'F'])                  # adds 2w ANOVA food F value to list
                P_tw_int.append(table2.loc['C(food):C(CT)', 'PR(>F)'])      # adds 2w ANOVA food p value  to list
                F_tw_int.append(table2.loc['C(food):C(CT)', 'F'])           # adds 2w ANOVA food F value to list
               
            except ValueError:
                print(f'{i}_{food}_{norm_gen}_is not expressed')
                not_expressed_s2.append(f'{i}_{food}_{norm_gen}')
                P_tw_food.append(np.nan)
                F_tw_food.append(np.nan)
                P_tw_int.append(np.nan)
                F_tw_int.append(np.nan)
            except KeyError:
                print(f'{i}_{food}_{norm_gen}_is missing')
                not_expressed_s2.append(f'{i}_{food}_{norm_gen}')
                P_tw_food.append(np.nan)
                F_tw_food.append(np.nan)
                P_tw_int.append(np.nan)
                F_tw_int.append(np.nan)


df_stat2 = pd.DataFrame()
df_stat2['profile'] = profile_tw 
df_stat2['P_2w_ANOVA_food'] = P_tw_food
df_stat2['F_2w_ANOVA_food'] = F_tw_food
df_stat2['P_2w_ANOVA_interact'] = P_tw_int
df_stat2['F_2w_ANOVA_interact'] = F_tw_int
                                                     
df_cos_max_stat2 = pd.merge(df_cos_max_stat, df_stat2.set_index(index), on=['tissue', 'food', 'Gene', 'profile']).sort_values(by=['tissue', 'food', 'function'])  #merge to single dataframe


####### Export final dataframes to csv ######################################################################################################
df_mean_cat.to_csv('df_mean_categories.csv', index=True)    #means/SD/SEM with categories for plotting
df_mean.to_csv('df_mean.csv', index=True)                   #Final File with Mean/SD of expression, with NaN
df_mean1.to_csv('df_mean_max.csv', index=True)              #mean/SD/SEM for each gene, in columns
#df_cos.to_csv('df_cos.csv', index=True)                    #this is cos parameters, but without Phase correction 
df_heat.to_csv('df_heat.csv', index=True)                   #mean/SD/SEM for each gene, in rows
#df_max_values.to_csv('df_max_values.csv')                  #max mean of expression
#df_cos_max.to_csv('df_cos_max.csv')                        #this is cos parameters + max CT and expr value of each profile + Final Corrected Phase, use Final dataframe with ANOVA
df_cos_max_stat2.to_csv('df_cos_max_stat.csv')              #Final dataframe for now, cosinor, 1w, 2w ANOVA
###LEGEND for df_cos_max_stat>>> Expression columns = max mean expression value, CT column = time when max expression
#############################################################################################################################################



##########################################################################
####### Final plots ######################################################
##########################################################################


####### Read previously prepared data to speed up testing ################

#df_cos = pd.read_csv('.\df_cos.csv', delimiter = ',')
#df_heat = pd.read_csv('.\df_heat.csv', delimiter = ',')
#df_mean = pd.read_csv('.\df_mean.csv', delimiter = ',')
#df_mean1 = pd.read_csv('.\df_mean_max.csv', delimiter = ',')
#df_mean_cat = pd.read_csv('.\df_mean_categories.csv', delimiter = ',')

#df_mean.set_index(['tissue', 'food', 'CT'])
#df_mean.columns = pd.MultiIndex.from_product([genes_list, means], names=['Genes', 'Means'])

####### Quality Control Plot ########
#####################################
"""

x_ax = df_cos['R2']
y_ax1 = df_cos['P_value']
y_ax2 = df_cos['Rel_Amp_Error']

fig0, ax1 = plt.subplots(sharey = True)
ax1.scatter(x_ax, y_ax1, color='b')
ax1.set_xlabel(x_ax.name)
ax1.set_ylabel(y_ax1.name, color='b')
ax1.tick_params(colors='b')
ax2 = ax1.twinx()
ax2.scatter(x_ax, y_ax2, color='r')
ax2.set_ylabel(y_ax2.name, color='r')
ax2.tick_params(colors='r')
plt.title("Quality Control")

fig0.legend(loc = 1)
#fig0.tight_layout()
plt.show()
#####################################
"""


####### Line plots ##################
#####################################

"""
#plot from mean values
sns.lineplot(x="CT", y=df_mean['Per2_rel', 'mean'].values, hue="food", style="tissue", data=df_mean.reset_index())
plt.show()

#plot from raw values, automatic error bars
#sns.lineplot(x="CT", y='Per2_rel', hue="food", style="tissue", data=df.reset_index())
#plt.show()
#####################################
"""


####### Heatmap #####################
#####################################

"""
###### Heatmap variables ############
heat_tissue_1 = 'liv'
heat_LD = 'LD'
heat_RF = 'RF'

###### Heatmap plot ############
#df_heat_spec = df_heat.loc[('liv', 'LD'), 'mean'].unstack('CT')   #specify what to plot but without categories
#to use gene categories in multiindex, do this:
dfrest = df_mean_cat.reset_index()      #not sure how to select using multiindex tuples or query, it is easier to convert to values and then back to index
# LD heat data
dfrest2 = dfrest.loc[(dfrest['tissue'] == heat_tissue_1) & (dfrest['food'] == heat_LD) & (dfrest['Aggr'] == 'mean') & (dfrest['Gene'].str.contains("_rel"))]   #select by columns
df_heat_spec2 = dfrest2.set_index(['tissue', 'food', 'function', 'Gene', 'Aggr', 'CT']).unstack('CT')   #now set index back again and then unstack CT for 2d heatmap plot
# RF heat data
dfrest3 = dfrest.loc[(dfrest['tissue'] == heat_tissue_1) & (dfrest['food'] == heat_RF) & (dfrest['Aggr'] == 'mean') & (dfrest['Gene'].str.contains("_rel"))]
df_heat_spec3 = dfrest3.set_index(['tissue', 'food', 'function', 'Gene', 'Aggr', 'CT']).unstack('CT')

fig, axs = plt.subplots(ncols=3, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [4, 4, 1]})    #alt>>> fig, (ax1, ax2)=..then ax=ax1. To make small cbar, set width ratios
heat_LD = sns.heatmap(df_heat_spec2, xticklabels=[0, 4, 8, 12, 16, 20, 24], yticklabels=True, annot=False, cbar=False, ax=axs[0], cmap="YlGnBu")  # need to tell sns which ax to use  #cmap='coolwarm'
heat_RF = sns.heatmap(df_heat_spec3, xticklabels=[0, 4, 8, 12, 16, 20, 24], yticklabels=False, annot=False, cbar=True, cbar_ax=axs[2], ax=axs[1], cmap="YlGnBu") 

plt.show()
#####################################
"""

"""
####### Polar Phase Plot #########################################
##################################################################


####### Single Polar Phase Plot #########################################
# Amplitude is length and width of bar is SD of Phase

data = df_cos_max.copy().reset_index().set_index(['tissue', 'food', 'function']).sort_values(by=['tissue', 'food', 'function'])

phaseseries = data.loc[('SCN', 'LD'), 'Phase_final'].values.flatten()   # unlike previously with cos, this lacks the nan values
#phaseseries = data.loc[('SCN', 'LD'), 'Peaks'].values.flatten()        # To plot scipy.find_peaks data instead Cosinor Phase
phase_sdseries = data.loc[('SCN', 'LD'), 'Phase_SD'].values.flatten()
phase = []
phase_sd = []
for phasevalue in phaseseries:
    phase.append(polarphase(phasevalue))
for phasesdvalue in phase_sdseries:
    phase_sd.append(polarphase(phasesdvalue))
amp = data.loc[('SCN', 'LD'), 'Amplitude'].values.flatten()
genes = data.loc[('SCN', 'LD'), 'Gene'].values.flatten()
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

### To save as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)

ax = plt.subplot(111, projection='polar')                                               #plot with polar projection
bars = ax.bar(phase, amp, width=phase_sd, color=colorcode, bottom=0.0, alpha=0.8)       # >prusvitnost-> alpha=0.5

#ax.set_yticklabels([])          # this deletes radial ticks
ax.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
ax.set_theta_direction(-1)      #reverse direction of theta increases
ax.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
ax.legend(bars, genes, fontsize=8, bbox_to_anchor=(1.1, 1.1))   # legend needs sequence of labels after object bars which contains sequence of bar plots 
ax.set_xlabel("Circadian phase (h)", fontsize=12)
plt.title("Rayleigh-style phase plot", fontsize=14, fontstyle='italic')

plt.savefig(f'Phase plot RNA.svg', format = 'svg', bbox_inches = 'tight')
#plt.show()             #to show plot, comment out mpl.use and rcparams lines and do not save as svg

"""



####### Multiple Polar Phase Plots, v2 ########
####### Data LD ###############################

"""
data = df_cos_max.copy().reset_index().set_index(['tissue', 'food', 'function']).sort_values(by=['tissue', 'food', 'function']).reset_index()

#Use CTRL+H to replace tissue name to create different plots

#phaseseries = data.loc[('liv', 'LD'), 'Phase_final'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()   # only include those with P<0.05 in LD or RF
#phase_sdseries = data.loc[('liv', 'LD'), 'Phase_SD'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()

phaseseries = data.loc[(data['tissue'] == 'liv') & (data['food'] == 'LD') & (data['function'] != '6.NATs'), 'Phase_final'].values   # But need filtering of p values sep for LD and RF, how?
phase_sdseries = data.loc[(data['tissue'] == 'liv') & (data['food'] == 'LD') & (data['function'] != '6.NATs'), 'Phase_SD'].values
phase = []
phase_sd = []
for phasevalue in phaseseries:
    phase.append(polarphase(phasevalue))
for phasesdvalue in phase_sdseries:
    phase_sd.append(polarphase(phasesdvalue))
#amp = data.loc[('liv', 'LD'), 'Amplitude'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()
#genes = data.loc[('liv', 'LD'), 'Gene'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()

amp = data.loc[(data['tissue'] == 'liv') & (data['food'] == 'LD') & (data['function'] != '6.NATs'), 'Amplitude'].values
genes = data.loc[(data['tissue'] == 'liv') & (data['food'] == 'LD') & (data['function'] != '6.NATs'), 'Gene'].values

colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

### To save as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)

####### Data RF ##########
#phaseseries1 = data.loc[('liv', 'RF'), 'Phase_final'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()
#phase_sdseries1 = data.loc[('liv', 'RF'), 'Phase_SD'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()

phaseseries1 = data.loc[(data['tissue'] == 'liv') & (data['food'] == 'RF') & (data['function'] != '6.NATs'), 'Phase_final'].values

phase1 = []
phase_sd1 = []
for phasevalue1 in phaseseries1:
    phase1.append(polarphase(phasevalue1))
for phasesdvalue1 in phase_sdseries1:
    phase_sd1.append(polarphase(phasesdvalue1))
#amp1 = data.loc[('liv', 'RF'), 'Amplitude'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()
#genes1 = data.loc[('liv', 'RF'), 'Gene'][((data.loc[('liv', 'LD'), 'P_value']) < 0.05) | ((data.loc[('liv', 'RF'), 'P_value']) < 0.05)].values.flatten()


colorcode1 = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes1)))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

####### Polar Plots ##########
fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1,  subplot_kw={'projection':'polar'})    #, sharex=False, sharey=False

bars0 = ax0.bar(phase, amp, width=phase_sd, color=colorcode, bottom=0.0, alpha=0.8)
ax0.set_ylim(0,0.4)                                                                     # sets limit for polar axis
ax0.set_yticks(np.arange(0.1,0.4,0.1))                                                    # sets min, max ticks for polar axis + steppings
ax0.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
ax0.set_theta_direction(-1)      #reverse direction of theta increases
ax0.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
#ax0.legend(bars0, genes, fontsize=7, bbox_to_anchor=(1.1, 1.1))   # legend needs sequence of labels after object bars which contains sequence of bar plots, use fig.legend instead 
ax0.set_xlabel("Circadian phase (h)", fontsize=12)
ax0.set_title("liv LD, EC genes", fontsize=12)

bars1 = ax1.bar(phase1, amp1, width=phase_sd1, color=colorcode1, bottom=0.0, alpha=0.8)
ax1.set_ylim(0,0.4)
ax1.set_yticks(np.arange(0.1,0.4,0.1))
ax1.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
ax1.set_theta_direction(-1)      #reverse direction of theta increases
ax1.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
#ax1.legend(bars1, genes1, fontsize=7, bbox_to_anchor=(1.1, 1.1))   # legend needs sequence of labels after object bars which contains sequence of bar plots, use fig.legend instead  
ax1.set_xlabel("Circadian phase (h)", fontsize=12)
ax1.set_title("liv RF, EC genes",fontsize=12)

fig.legend(bars1, genes1, fontsize=6, loc=7)                        # common Fig legend using ax1 data, use when both plots have the same genes
fig.tight_layout()
fig.subplots_adjust(right=0.9)                                      # adjust size to fit legend better
fig.suptitle("Rayleigh-style phase plot", fontsize=14, fontstyle='italic')
plt.savefig(f'Phase plot RNA.svg', format = 'svg', bbox_inches = 'tight')
#plt.show()             #to show plot, comment out mpl.use and rcparams lines and do not save as svg


"""
            


"""
####### Line Plots #########################################
############################################################


####### Cosin Plots #####################
df = df_mean_cat.copy()

height = 9      #this should be range(len(df_cos_mac['Function']))
width = 4       #this should be range(len(df_cos_mac['Gene']))

fig, ax = plt.subplots(height, width, sharex=True, sharey=False) 

i1 = 0
i2 = 0

byfunction = df_cos_max.groupby('function')
fun_list = []
for func_grp, frame in byfunction:
	fun_list.append(func_grp)



DODELAT - zatim iteruje jen prvni radek pak spadne kvuli index out of bounds
potreba doplnit categories do df0 nebo delat z jineho, napr dr_mean_categories.csv


norm_gene2 = need list of genes in each category

or try except solution



for func in fun_list:
    
    for norm_gen in norm_gene2:
        xdata0 = [0, 4, 8, 12, 16, 20, 24]
        ydata0 = df_mean_cat.loc[('liv', 'LD', func, norm_gen, 'mean'), 'Expression'].values.flatten()
        ydata1 = df_mean_cat.loc[('liv', 'RF', func, norm_gen, 'mean'), 'Expression'].values.flatten()
        xerr0 = df_mean_cat.loc[('liv', 'LD', func, norm_gen, 'sem'), 'Expression'].values.flatten()
        xerr1 = df_mean_cat.loc[('liv', 'RF', func, norm_gen, 'sem'), 'Expression'].values.flatten()
        newindex = ~((ydata0 == 0) & (np.isnan(xdata0)) & (np.isnan(ydata0)))
        newindex1 = ~((ydata1 == 0) & (np.isnan(xdata0)) & (np.isnan(ydata1)))
        xdata = xdata0[newindex]            
        ydata = ydata0[newindex]
        ydataRF = ydata1[newindex1]       
        
        ax[i1, i2].errorbar(xdata, ydata, xerr=xerr0, c="b", label='LD')
        ax[i1, i2].errorbar(xdata, ydataRF, xerr=xerr1, c="g", label='RF')
                
        x_fit = np.linspace(np.min(xdata), np.max(xdata), 1000)
        try:

            fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor, xdata, ydata, p0, full_output=1)        
            ax[i1, i2].plot(x_fit, cosinor(x_fit, fitParams[0], fitParams[1], fitParams[2]), c="b", label='Cosinor')
            fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor, xdata, ydataRF, p0, full_output=1)        
            ax[i1, i2].plot(x_fit, cosinor(x_fit, fitParams[0], fitParams[1], fitParams[2]), c="y", label='Cosinor')
            
        except ValueError:
            print(f'{i}_{food}_{norm_gen}_is not expressed')
        except TypeError:
            print(f'{i}_{food}_{norm_gen}_is missing')
        except KeyError:
            print(f'{i}_{food}_{norm_gen}_is missing')
            
        i2 += 1
    i1 += 1

"""



"""
for func in fun_list:
    
    for norm_gen in norm_gene:
        xdata0 = df.loc[(df['tissue'] == 'liv') & (df['food'] == 'LD'), ['CT']].values.flatten()
        ydata0 = df.loc[(df['tissue'] == 'liv') & (df['food'] == 'LD') , [norm_gen]].values.flatten()
        ydata1 = df.loc[(df['tissue'] == 'liv') & (df['food'] == 'RF') , [norm_gen]].values.flatten()
        newindex = ~((ydata0 == 0) & (np.isnan(xdata0)) & (np.isnan(ydata0)))
        newindex1 = ~((ydata1 == 0) & (np.isnan(xdata0)) & (np.isnan(ydata1)))
        xdata = xdata0[newindex]            
        ydata = ydata0[newindex]
        ydataRF = ydata1[newindex1]
        ax[i1, i2].scatter(xdata, ydata, c="r", label='LD')
        ax[i1, i2].scatter(xdata, ydataRF, c="g", label='RF')
        
        x_fit = np.linspace(np.min(xdata), np.max(xdata), 1000)
        try:

            fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor, xdata, ydata, p0, full_output=1)        
            ax[i1, i2].plot(x_fit, cosinor(x_fit, fitParams[0], fitParams[1], fitParams[2]), c="b", label='Cosinor')
            fitParams, fitCovariances, infodict, mesg, ier = curve_fit(cosinor, xdata, ydataRF, p0, full_output=1)        
            ax[i1, i2].plot(x_fit, cosinor(x_fit, fitParams[0], fitParams[1], fitParams[2]), c="y", label='Cosinor')
            
        except ValueError:
                print(f'{i}_{food}_{norm_gen}_is not expressed')
        except TypeError:
                print(f'{i}_{food}_{norm_gen}_is missing')               
        i2 += 1
    i1 += 1
"""

#### Need to iterate over data and also over fig dimensions
#fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
#for i in range(2):
#	for j in range(3):
#ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')

           

"""
xdata0 = df.loc[(df['tissue'] == 'SCN') & (df['food'] == 'LD'), ['CT']].values.flatten()
ydata0 = df.loc[(df['tissue'] == 'SCN') & (df['food'] == 'LD') , [norm_gene[21]]].values.flatten()  #norm_gene[21] is Per2
newindex = ~((ydata0 == 0) & (np.isnan(xdata0)) & (np.isnan(ydata0)))
xdata = xdata0[newindex]            
ydata = ydata0[newindex]
ax[0, 0].scatter(xdata, ydata, c="r", label='Data')             #plot original data
x_fit = np.linspace(np.min(xdata), np.max(xdata), 1000)         #create smooth x axis your hires curve can plot over
# this needs to be specific for each plot, call df_cos instead fitParams[i]
ax[0, 0].plot(x_fit, cosinor(x_fit, fitParams[0], fitParams[1], fitParams[2]), c="b", label='Cosinor')    #plot cosinor curve, use calc parameters to create y funct over x_fit
"""
"""
###### Legends and text using matplotlib.offsetbox.AnchoredText #############
#############################################################################
###### Set common ################
x_lab = 'time(h)'
y_lab = f' {str(gene_y)} expression'
suptitle_all = 'Tissue'
xlim = (-0.5, xdata.max() + 0.5)    
xticks = [0, 4, 8, 12, 16, 20, 24]
ylim = (0, ydata.max() + ydata.max()/10)
yticks = [0, 0.5, 1]
"""
"""
# this needs to be specific for each plot, call df_cos instead fitParams[i]
anchored_text = AnchoredText(f'''Mesor = {round(fitParams[0], 4)} \nAmplitude = {round(fitParams[1], 4)}
Phase = {round(fitParams[2], 4)} \nTrend = {round(fitParams[3], 4)} \nSum of sq = {round(fres, 4)}''', loc=2, prop=dict(fontsize=5), frameon=True) #loc=2 is left upper corner, 1 is right corner

###### Add to plot ###########
ax[0, 0].add_artist(anchored_text)

ax[0, 0].set(xlabel='time(h)', ylabel=y_lab, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)

fig.suptitle(suptitle_all, fontsize=14, fontweight='bold')

######## Display and save plots #######
plt.show()
"""

