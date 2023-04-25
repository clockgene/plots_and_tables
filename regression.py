# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:54:53 2022
Multivariette regression models, dimension. reduct. for Katyas weight data
v.20230425 v.3slopes
@author: martin.sladek
"""

import seaborn as sns
import pandas as pd
import numpy as np
import os, re
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
from statsmodels.multivariate.pca import PCA
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels import graphics
from matplotlib_venn import venn2
import pingouin as pg
import datetime
import winsound
from statsmodels.multivariate.pca import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from scipy.stats import linregress
from tkinter import filedialog
from tkinter import *

def LinearMixedModel(null_formula, formula, data, y, groups, re_formula=None):     # e.g. >> LinearMixedModel(data, "tes", "sex", "age", groups=data['MSFsasc_dec'])
    null = smf.mixedlm(null_formula, data=data, groups=groups, re_formula=re_formula, missing='drop')
    model = smf.mixedlm(formula, data=data, groups=groups, re_formula=re_formula, missing='drop')
    results = model.fit(reml=False)
    results_res = null.fit(reml=False)
    summary = results.summary()
    wald = results.wald_test_terms()
    lrdf = (results_res.df_resid - results.df_resid) # df_resid = degress of freedom
    lrstat = -2*(results_res.llf - results.llf) # llf = log_likelihood_ratio
    lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)
    with open (f'MIXED_LM_{y}.txt', 'a') as f: f.write (f'{summary} \n{wald} \n\nChi-square test of models p = {lr_pvalue}')
    print(summary)
    print(lr_pvalue)
    
def OLS(data, formula, y):
    model = smf.ols(formula, data=data, missing='drop')
    results = model.fit()
    summary = results.summary()
    anova = sm.stats.anova_lm(results, typ=2)
    anova['PercntExplained'] = anova.sum_sq/anova.sum_sq.sum() * 100 
    # test Heteroskedasticity (if residuals/variance get larger or smaller with larger X value, null hypothesis (H0): Homoscedasticity is present, if p>0.05, it is OK), use GLS instead of OLS if p<0.05
    bp_test = het_breuschpagan(results.resid,  results.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    hetero = dict(zip(labels, bp_test)) # if P value is significant, data is heterosk. and GLS should be used
    # Q-Q plot to show distribution, https://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot
    resid = results.resid
    graphics.gofplots.qqplot(resid, line='r')
    plt.savefig(f'OLS_residuals_qqplot_{y}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'OLS_residuals_qqplot_{y}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    # write results to text file
    print(summary)
    print(hetero)
    with open (f'OLS_{y}.txt', 'a') as f: f.write (f'{summary}\n\nANOVA{anova}\n\nBreusch-Pagan test for Heteroskedasticity \n{hetero}')

# Generalized Least Squares Multiple Linear Regression, for Heteroskedastic data ?
def GLS(data, formula, y, version='GLS'):
    if version == 'GLS':
        model = sm.GLS.from_formula(formula, data=data, missing='drop')
    else:        
        model = sm.GLSAR.from_formula(formula, data=data, missing='drop')
    results = model.fit()
    summary = results.summary()
    # test Heteroskedasticity (if residuals/variance get larger or smaller with larger X value), if it does, use GLS instead of OLS
    bp_test = het_breuschpagan(results.resid,  results.model.exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    hetero = dict(zip(labels, bp_test)) # if P value is significant, data is heterosk. and GLS should be used
    # Q-Q plot to show distribution, https://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot
    resid = results.resid
    graphics.gofplots.qqplot(resid, line='r')
    plt.savefig(f'GLS_residuals_qqplot_{y}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'GLS_residuals_qqplot_{y}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    # Histogram of residuals
    plt.hist(resid, bins='auto')
    k2, p = stats.normaltest(resid, nan_policy='omit')
    plt.annotate('normality test \nP = ' + str(round(p, 6)), xy=(1, 1), xycoords='axes fraction', fontsize=10, 
                 xytext=(-20, -20), textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    plt.savefig( f'GLS_residuals_histplot_{y}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'GLS_residuals_histplot_{y}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    # write results to text file
    print(summary)
    print(hetero)
    with open ('GLS_{y}.txt', 'a', encoding="utf-8") as f: f.write(f'{summary}\n\n\n\nBreusch-Pagan test for Heteroskedasticity \n{hetero}')    

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()        
    return df_std    
    # call the z_score function
    # df_cars_standardized = z_score(df_cars)

def KW_BoxenPlot1(data, col_var, y, ylim=(None, None), kind='boxstrip'):
    x = col_var
    x_lab = col_var
    y_lab = y
    by = data.groupby(col_var)  # for ANOVA and labels, auto-create col_order
    #categories = len(by)
    col_order = []
    for a, frame in by:
        col_order.append(a)
    suptitle_all = f'{x_lab} vs {y_lab}'

    f, ax = plt.subplots(figsize=(3, 6))  # figsize not working?
    
    if kind == 'boxstrip':    
        sns.boxplot(x=x, y=y, data=data, order=col_order)
        sns.stripplot(x=x, y=y, data=data, order=col_order)
    else:
        g = sns.catplot(x=x, y=y, order=col_order, kind=kind, data=data, aspect=0.5)
        g.set(ylim=ylim)

    ##### Kruskal-Wallis H-test with auto-assigned data ######
    alist = []
    for i in range(len(col_order)):
        alist.append(data[y][data[x] == col_order[i]].dropna(how='any'))
    tano, pano = stats.kruskal(*alist) # asterisk is for *args - common idiom to allow arbitrary number of arguments to functions    
    if pano < 0.0000000001:
        #plt.text(x_coord, y_coord, 'Kruskal-Wallis p < 1e-10' , fontsize=10)
        ax.annotate('Kruskal-Wallis p < 1e-10', xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                    textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    else:
        #plt.text(x_coord, y_coord, 'Kruskal-Wallis p = ' + str(round(pano, 8)), fontsize=10)
        ax.annotate('Kruskal-Wallis p = ' + str(round(pano, 8)), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                    textcoords='offset points', horizontalalignment='right', verticalalignment='top')        

    plt.savefig(f'KW_BoxenPlot1_{suptitle_all}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'KW_BoxenPlot1_{suptitle_all}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()

    ##### Post-hoc tests - Dunn's ###############
    xx = data[x][(data[x] == data[x]) & (data[y] == data[y])].dropna(how='any')
    yy = data[y][data[x] == data[x]].dropna(how='any')
    df_stat = pd.DataFrame(xx)          #create new dataframe to avoid NaN problems
    df_stat[y] = yy                     #add column with analysed data to new dataframe

    #posthoc = sp.posthoc_dunn(df_stat.reset_index(drop=True), val_col=y, group_col=x)
    pc = sp.posthoc_dunn(df_stat, val_col=y, group_col=x)
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.79, 0.35, 0.035, 0.3]}  #
    sp.sign_plot(pc, **heatmap_args)

    plt.savefig(f'KW_BoxenPlot1_{suptitle_all}_posthoc.png', format = 'png')
    plt.savefig(f'KW_BoxenPlot1_{suptitle_all}_posthoc.svg', format = 'svg')    
    plt.clf()
    plt.close()   

# 3-d scatter plot with rotation animation
def three_d_scatter(data, x, y, z, hue, mydir, animate=True, markevery=1):  # three_d_scatter(data, 'PC2_bio', 'LD1_bio', 'tSNE2_bio', 'SJL_quantile', mydir)
    #data = data[[x,y,z,hue]].dropna()
    by = data.groupby(data[hue])
    hue_order = []
    for a, frame in by:
        hue_order.append(a)
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for s in hue_order:
        # ax.scatter(data[x][data[hue]==s], data[y][data[hue]==s], data[z][data[hue]==s], label=s, s=3, alpha=0.7)
        ax.scatter(data[x][data[hue]==s][::markevery], data[y][data[hue]==s][::markevery], data[z][data[hue]==s][::markevery], label=s, s=3, alpha=0.7)
        
    ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()
    
    if animate == True:
        import matplotlib.animation as animation
        def rotate(angle):
            ax.view_init(azim=angle)
        degrees = 360   # 360  140
        print("Making animation")
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, degrees + 2, 2), interval=100)
        rot_animation.save(f'{mydir}\\3d_scatter_animated_{hue}.gif', dpi=80, writer='imagemagick')  # dpi=200
    
    else:
        plt.savefig(f'{mydir}\\3d_scatter_{hue}.png', format = 'png', bbox_inches = 'tight')
        plt.savefig(f'{mydir}\\3d_scatter_{hue}.svg', format = 'svg', bbox_inches = 'tight')
    
    plt.clf()
    plt.close()  
    
    
# T-distributed Stochastic Neighbor Embedding (t-SNE) – probabilistic (not mathematical as PCA and LDA), nonlinear, dimensionality reduction tech.
# alterantive to tSNE is UMAP (need import umap, not installed yet), better for higher number of dimensions
# hue - can be ordinal or contin. as well, seaborn plots heatmaps
def tSNE(data, columns, hue, name, mydir, init='pca', perplexity=30, three_d=False, z=True):      # init='random'
    x = data[columns]
    if z is True:
        x = z_score(data[columns])
    x = x.dropna().reset_index()
    # oldindex = x.pop('index')
    oldindex = data.index
    
    by_hue = data.groupby(data[hue])  # for ANOVA and labels, auto-create col_order
    hue_order = []
    for a, frame in by_hue:
        hue_order.append(a) 
    
    if three_d == True:
        tsne = TSNE(n_components=3, verbose=1, random_state=123, init=init, perplexity=perplexity)
        z = tsne.fit_transform(x) 
        dataset = pd.DataFrame(z[:,0], index=oldindex, columns=['tSNE1'])
        dataset2 = pd.DataFrame(z[:,1], index=oldindex, columns=['tSNE2'])
        dataset3 = pd.DataFrame(z[:,2], index=oldindex, columns=['tSNE3'])    
        dataset = dataset.join(dataset2)
        dataset = dataset.join(dataset3)
        data = data.join(dataset)
        three_d_scatter(data=data, x='tSNE1', y='tSNE2', z='tSNE3', hue=hue, mydir=mydir, markevery=5) # set markevery>1 if u see visual artifacts due to big n
    
    else:
        tsne = TSNE(n_components=2, verbose=1, random_state=123, init=init, perplexity=perplexity)
        z = tsne.fit_transform(x) 
        dataset = pd.DataFrame(z[:,0], index=oldindex, columns=['tSNE1'])
        dataset2 = pd.DataFrame(z[:,1], index=oldindex, columns=['tSNE2'])   
        dataset = dataset.join(dataset2)
        data = data.join(dataset)    
    
    sns.scatterplot(x="tSNE1", y="tSNE2", hue=hue, hue_order=hue_order, data=data, s=18, edgecolor="none").set(title=f't-SNE of {hue}') # edgecolor="none" to remove white outlines
    plt.savefig(f'{mydir}' + '\\' + f'tSNE_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'tSNE_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()  
    
    # description in txt file
    print(f'Number of variables without NaNs: {len(x)}')
    description = open(f'{mydir}\\tSNE_{hue}_{name}.txt', 'w')
    description.write(f'hue = {hue}\n\n{columns} \n\n Number of variables without NaNs: {len(x)}')
    description.close()   
    
    return data['tSNE1'], data['tSNE2']



def PCA_plot(data, columns, hue, name, mydir, pc='PC2'):
    # alternative PCA from sklearn
    #from sklearn.decomposition import PCA as SKL_PCA
    #from sklearn.preprocessing import scale  #alternative standardization
    #dta = pd.DataFrame(scale(data[columns]))
    #pca = SKL_PCA(n_components=2)
    #X_pca = SKL_PCA.fit_transform(X, y)    
    dta = z_score(data[columns])  # values for PCA should be standardized/normalized, 
    pca_model = PCA(dta.dropna().T, standardize=False, demean=True)
    #idx = pca_model.loadings.iloc[:, 0].argsort() # sorts subjects by pc1 score, returns indexes of sorted values, may be useful for some plots   
    
    # scree plot to show PC scores
    pca_model.plot_scree(log_scale=False)
    plt.savefig(f'{mydir}' + '\\' + f'PCA_scree_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'PCA_scree_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    
    # scatter plot of PC1 x PC2
    data['PC1'] = pca_model.loadings.iloc[:, 0] # pca_model.loadings['comp_0'] >> or 'comp_01'
    data['PC2'] = pca_model.loadings.iloc[:, 1] # pca_model.loadings['comp_1']
    sns.scatterplot(x="PC1", y="PC2", data=data, hue=hue, s=12)
    plt.savefig(f'{mydir}' + '\\' + f'PCA_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'PCA_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
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



# calculate the mean values and SD of variables grouped by groupvariable, e.g. printMeanAndSdByGroup(data[['Social_jetlag', 'slequal']], data['age_quantile'])
def printMeanAndSdByGroup(variables, groupvariable):
    data_groupby = variables.groupby(groupvariable)
    print("## Means:")
    display(data_groupby.apply(np.mean))
    print("\n## Standard deviations:")
    display(data_groupby.apply(np.std))
    print("\n## Sample sizes:")
    display(pd.DataFrame(data_groupby.apply(len)))
#printMeanAndSdByGroup(X, y)


# from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.grid(False)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# LDA - linear discriminant analysis, eigendecomposition of  within- and between-class covariance matrices 
# calculates n of LD scores, n(LD) = n(group, dependent variable) - 1
# columns = list of one X and all y variables [X, y1, y2,...], hue = class, group
def LDA_plot(data, columns, hue, name, mydir, scatter=True, plot_PCAxLDA=True, pc='PC2', stat="density", normal_dis=False):
    #dta = data[columns].dropna().reset_index().drop('index', axis=1)
    dta = data[columns].dropna()   # .reset_index()
    # oldindex = dta.pop('index')
    oldindex = data.index
    X = dta.iloc[:, 1:]  # independent variables data
    y = dta.iloc[:, 0]  # dependednt variable data
    y = pd.Categorical(y)
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, y) 
    #lda.explained_variance_ratio_
    dta['LD1'] = X_lda[:,0]    
    dataset = pd.DataFrame(X_lda[:,0], index=oldindex, columns=['LD1'])
    data = data.join(dataset) # need to return data and then run> data = LDA(...)
    
    # Scatter plot LD1 x LD2
    if scatter == True:  
        dta['LD2'] = X_lda[:,1]
        #sns.lmplot("LD1", "LD2", dta, hue=hue, fit_reg=False)
        sns.scatterplot(x="LD1", y="LD2", data=dta, hue=hue, s=12)
        plt.savefig(f'{mydir}' + '\\' + f'LDA_scatter_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
        plt.savefig(f'{mydir}' + '\\' + f'LDA_scatter_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
        plt.clf()
        plt.close()
    
    # Scatter plot PC2 x LD1
    if plot_PCAxLDA == True:
        sns.scatterplot(x=pc, y="LD1", data=data, hue=hue, s=12)
        plt.savefig(f'{mydir}\\LDAxPCA_scatter_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
        plt.savefig(f'{mydir}\\LDAxPCA_scatter_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
        plt.clf()
        plt.close()   
    
    # Confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X_lda, y, random_state=1)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    confusion_matrix(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, lda.classes_, title='Normalized confusion matrix')
    plt.savefig(f'{mydir}' + '\\' + f'LDA_CM_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'LDA_CM_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()   

    ###### Calculate p values between hue for separate categories in LD1 ######
    by_hue = dta.groupby(dta[hue])  # for ANOVA and labels, auto-create col_order
    hue_order = []
    for a, frame in by_hue:
        hue_order.append(a) 
        
    # Histogram
    xmin = np.trunc(np.min(X_lda[:,0])) - 1
    xmax = np.trunc(np.max(X_lda[:,0])) + 1
    #ncol = len(set(y))
    binwidth = 0.5
    bins=np.arange(xmin, xmax + binwidth, binwidth)  
    #from sklearn.preprocessing import LabelEncoder
    #hue = LabelEncoder().fit_transform(dta[f'{hue}'])  # encodes even strings as numerical categorical values
    fig = plt.figure()
    ax = fig.add_subplot()
    ax = sns.histplot(data=dta, x='LD1', hue=hue, hue_order=hue_order, bins=bins, binwidth=binwidth, kde=True,  stat="density")  
    ax.set_box_aspect(1)
    plt.xlabel('Linear discriminant function')     

    if normal_dis == True:
        ##### ANOVA ######
        alist = []
        for i in range(len(hue_order)):
            alist.append(dta['LD1'][(dta[hue] == hue_order[i])].dropna(how='any'))
        F, p = stats.f_oneway(*alist) # asterisk is for *args - common idiom to allow arbitrary number of arguments to functions    
        if p < 0.0000000001:
            plt.annotate('ANOVA p < 1e-10', xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                        textcoords='offset points', horizontalalignment='right', verticalalignment='top')
        else:
            plt.annotate('ANOVA p = ' + str(round(p, 8)), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                        textcoords='offset points', horizontalalignment='right', verticalalignment='top')  
    else:    
        ##### Mann-Whitney ######
        alist = []
        for i in range(len(hue_order)):
            alist.append(dta['LD1'][(dta[hue] == hue_order[i])].dropna(how='any'))    
        U, p = stats.kruskal(*alist)       
        plt.annotate('Mann-Whit. p = ' + str(round(p, 3)), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                    textcoords='offset points', horizontalalignment='right', verticalalignment='top')  

    plt.savefig(f'{mydir}' + '\\' + f'LDA_histo_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'LDA_histo_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()  

    # report model performance
    n_scores = cross_val_score(lda, X, y, scoring='accuracy', n_jobs=-1, error_score='raise')    
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    
    # Describe variables in txt file
    description = open(f'{mydir}\\LDA_{hue}_{name}.txt', 'w')
    if normal_dis == True:
        description.write(f'{columns} \nANOVA p = {p} \nAccuracy: {np.mean(n_scores)} +- {np.std(n_scores)}')
    else:
        description.write(f'{columns} \nMann-Whitney p = {p} \nAccuracy: {np.mean(n_scores)} +- {np.std(n_scores)}')
    description.close()     
    
    if scatter == True:  
        return data['LD1']          # , data['LD2']
    else:
        return data['LD1']



# Divide (discretize) column into 2 discrete groups similarly sized, ADVANCED BINNING with sklearn, choose number of bins n_bins=2, # strategy='kmeans' 'quantile' 'uniform'
# example - divide age to 2 bins, name new column age_quantile, only pick subjects that have also papi=1 in data4 (those were given MCTQ)>
# data = k2bins(data, 'age', 'age', first_bin_name='1.Young', second_bin_name='2.Old', loc_filter='Yes', loc_data=data4, loc_col='papi', loc_value=1)
def k2bins(data, source_column, quantile_column, first_bin=0, second_bin=1, first_bin_name='Low', second_bin_name='High', loc_filter='No', loc_data='data', loc_col='column', loc_value=1, n_bins=2, encode='ordinal', strategy='quantile'):
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    if loc_filter == 'No':
        data_t = data[[source_column]].dropna().reset_index()           
    else:
        data_t = data[[source_column]].loc[loc_data[loc_col] == loc_value].dropna().reset_index()    
    # oldindex = data_t.pop('index')
    oldindex = data_t.pop('Litter')
    data = data.join(pd.DataFrame(enc.fit_transform(data_t.values[:, :]), index=oldindex, columns=[f'{quantile_column}_{strategy}']))
    data.loc[data[f'{quantile_column}_{strategy}'] == first_bin, f'{quantile_column}_{strategy}'] = first_bin_name
    data.loc[data[f'{quantile_column}_{strategy}'] == second_bin, f'{quantile_column}_{strategy}'] = second_bin_name
    return data      

# Loads raw data tables without calculated chronotypes and analyzes all of them at once
# os.chdir('C:\\Users\\martin.Sladek\\Disk Google\\vysledky\\\publikace 2021 Choroid plexus follow up\\data mod for review')

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

mydir = os.getcwd()

data_raw = pd.read_csv('katya.csv', delimiter = ',', encoding = "utf-8", low_memory=False)

df = data_raw[['Group', 'Litter_size_P14', 'Litter', 'Maternal_age_at_birth', 'Sex', 'percent_of_M','P14',
'P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91', 'Problem']]

# no filter, problematic pups as variable 'Problem'
# data = df.groupby(by=["Litter"]).mean()

# filter out dead pups
data = df.loc[df['Problem'] < 9].groupby(by=["Litter"]).mean()

# filter out dead pups but seprate weights males and females
data = df.loc[df['Problem'] < 9].groupby(by=["Litter", "Sex"]).mean()

# filter out all problematic pups
# data = df.loc[df['Problem'] == 0].groupby(by=["Litter"]).mean()


data['wgain'] = data['P91'] - data['P14'] 
# data['wgain42'] = data['P42'] - data['P28']
# data['wgain63'] = data['P63'] - data['P28']
# data['wgain91'] = data['P91'] - data['P77']
# data['wgain77'] = data['P77'] - data['P63']
# data['wgain56'] = data['P56'] - data['P42']
# data['wgain70'] = data['P70'] - data['P56']
# data['wgain84'] = data['P84'] - data['P70']
# data_std = z_score(data[['P14', 'Maternal_age_at_birth', 'Litter_size_P14']])


# this is wrong, need to calculate slopes for different litters, not groups!
# slopes91 = []
# for i in data.Group.unique():
#     dfn = data.loc[data['Group'] == i, 'P42':'P91']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     # slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     # slopes91.append(slope)
#     # data.loc[data.Group == i, 'slopes91'] = slope
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     plt.savefig(f'age42-91_vs_weight_Group_{i}.png', format = 'png')
#     plt.clf()
#     plt.close() 

   
# this is wrong, need to calculate slopes for different litters, not groups!
# slopes42 = []
# for i in data.Group.unique():
#     dfn = data.loc[data['Group'] == i, 'P14':'P42']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     slopes42.append(slope)
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     data.loc[data.Group == i, 'slopes42'] = slope
#     plt.scatter(data_melted['Time'], data_melted['Value'])
#     plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
#     plt.clf()
#     plt.close() 

# PLOT Time vs. Weight XY scatterplot by Group
dfnn = data[['Group', 'P14','P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91']]
# dfn = data.loc[:, 'P14':'P91']
# data_melted = dfnn.melt(var_name="Timepoint", value_name="Value")
data_melted = dfnn.melt(id_vars='Group', var_name="Timepoint", value_name="Value")
# Split the Timepoint column into two separate columns for time and replicate   
data_melted['Time'] = data_melted['Timepoint'].str[1:]
data_melted['Time'] = data_melted['Time'].astype(int) 
sns.scatterplot(data=data_melted, x="Time", y="Value", hue="Group")
plt.savefig('_slopes.png', format = 'png')
plt.clf()
plt.close() 

# data = k2bins(data, 'wgain', 'wgain_quant', first_bin_name='low', second_bin_name='high')

data = data.reset_index()

## BOTH SEXES TOGETHER
# for i in data.Litter.unique():    
#     dfn = data.loc[data['Litter'] == i, 'P14':'P42']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     data.loc[data['Litter'] == i, 'slopes42'] = slope
#     plt.scatter(data_melted['Time'], data_melted['Value'])
#     #plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
#     plt.clf()
#     plt.close() 

# for i in data.Litter.unique():    
#     dfn = data.loc[data['Litter'] == i, 'P42':'P91']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     data.loc[data['Litter'] == i, 'slopes91'] = slope
#     plt.scatter(data_melted['Time'], data_melted['Value'])
#     #plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
#     plt.clf()
#     plt.close() 
    
# SEXES separate    
# for i in data.Litter.unique():    
#     dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'P14':'P42']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'slopes42m'] = slope
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
#     # plt.clf()
#     # plt.close() 


# for i in data.Litter.unique():    
#     dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'P14':'P42']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     try:
#         slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     except ValueError:
#         slope = np.nan
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'slopes42f'] = slope
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
#     # plt.clf()
#     # plt.close() 

    
# for i in data.Litter.unique():    
#     dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'P42':'P91']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     try:
#         slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     except ValueError:
#         slope = np.nan
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'slopes91m'] = slope
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
#     # plt.clf()
#     # plt.close() 


# for i in data.Litter.unique():    
#     dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'P42':'P91']
#     # Reshape the data into long format using melt
#     data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
#     # Split the Timepoint column into two separate columns for time and replicate   
#     data_melted['Time'] = data_melted['Timepoint'].str[1:]
#     data_melted['Time'] = data_melted['Time'].astype(int) 
#     try:
#         slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
#     except ValueError:
#         slope = np.nan
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'slopes91f'] = slope
#     # plt.scatter(data_melted['Time'], data_melted['Value'])
#     # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
#     # plt.clf()
#     # plt.close() 

    
# SEXES sep and 3 ages    
for i in data.Litter.unique():    
    dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'P14':'P28']
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
    # Split the Timepoint column into two separate columns for time and replicate   
    data_melted['Time'] = data_melted['Timepoint'].str[1:]
    data_melted['Time'] = data_melted['Time'].astype(int) 
    slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'slopes28m'] = slope
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
    # plt.clf()
    # plt.close() 


for i in data.Litter.unique():    
    dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'P14':'P28']
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
    # Split the Timepoint column into two separate columns for time and replicate   
    data_melted['Time'] = data_melted['Timepoint'].str[1:]
    data_melted['Time'] = data_melted['Time'].astype(int) 
    try:
        slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
    except ValueError:
        slope = np.nan
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'slopes28f'] = slope
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
    # plt.clf()
    # plt.close() 

    
for i in data.Litter.unique():    
    dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'P28':'P49']
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
    # Split the Timepoint column into two separate columns for time and replicate   
    data_melted['Time'] = data_melted['Timepoint'].str[1:]
    data_melted['Time'] = data_melted['Time'].astype(int) 
    try:
        slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
    except ValueError:
        slope = np.nan
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'slopes49m'] = slope
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
    # plt.clf()
    # plt.close() 


for i in data.Litter.unique():    
    dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'P28':'P49']
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
    # Split the Timepoint column into two separate columns for time and replicate   
    data_melted['Time'] = data_melted['Timepoint'].str[1:]
    data_melted['Time'] = data_melted['Time'].astype(int) 
    try:
        slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
    except ValueError:
        slope = np.nan
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'slopes49f'] = slope
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
    # plt.clf()
    # plt.close() 

for i in data.Litter.unique():    
    dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'P49':'P91']
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
    # Split the Timepoint column into two separate columns for time and replicate   
    data_melted['Time'] = data_melted['Timepoint'].str[1:]
    data_melted['Time'] = data_melted['Time'].astype(int) 
    try:
        slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
    except ValueError:
        slope = np.nan
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    data.loc[(data['Litter'] == i) & (data['Sex'] == 'm'), 'slopes91m'] = slope
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
    # plt.clf()
    # plt.close() 


for i in data.Litter.unique():    
    dfn = data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'P49':'P91']
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
    # Split the Timepoint column into two separate columns for time and replicate   
    data_melted['Time'] = data_melted['Timepoint'].str[1:]
    data_melted['Time'] = data_melted['Time'].astype(int) 
    try:
        slope, intercept, rvalue, pvalue, stderr = linregress(data_melted['Time'], data_melted['Value'])
    except ValueError:
        slope = np.nan
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    data.loc[(data['Litter'] == i) & (data['Sex'] == 'f'), 'slopes91f'] = slope
    # plt.scatter(data_melted['Time'], data_melted['Value'])
    # plt.savefig(f'age14-42_vs_weight_Group_{i}.png', format = 'png')
    # plt.clf()
    # plt.close() 


data['slopes91'] = data['slopes91m']
data['slopes91'] = data['slopes91'].fillna(data['slopes91f'].loc[data['slopes91f'] == data['slopes91f']])
# data['slopes42'] = data['slopes42m']
# data['slopes42'] = data['slopes42'].fillna(data['slopes42f'].loc[data['slopes42f'] == data['slopes42f']])
data['slopes49'] = data['slopes49m']
data['slopes49'] = data['slopes49'].fillna(data['slopes49f'].loc[data['slopes49f'] == data['slopes49f']])
data['slopes28'] = data['slopes28m']
data['slopes28'] = data['slopes28'].fillna(data['slopes28f'].loc[data['slopes28f'] == data['slopes28f']])

data.loc[data.Sex == 'm', 'Sex_num'] = 1
data.loc[data.Sex == 'f', 'Sex_num'] = 2



# oldindex problem, need solving
# data = k2bins(data, 'slopes42', 'slopes42_quant', first_bin_name='low', second_bin_name='high')
# data = k2bins(data, 'slopes91', 'slopes91', first_bin_name='low', second_bin_name='high')

KW_BoxenPlot1(data, "Group", "wgain", kind='violin') 
KW_BoxenPlot1(data, "Group", "slopes91", kind='violin') 
# KW_BoxenPlot1(data, "Group", "slopes42", kind='violin') 
KW_BoxenPlot1(data, "Group", "slopes49", kind='violin') 
KW_BoxenPlot1(data, "Group", "slopes28", kind='violin') 
KW_BoxenPlot1(data, "Group", "P14", kind='violin') 
KW_BoxenPlot1(data, "Group", "P91", kind='violin') 
# def KW_BoxenPlot1(data, col_var, y, ylim=(None, None), kind='boxstrip'):



# enc = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy='quantile') 
# data_t = data[['wgain']].dropna().reset_index()
# oldindex = data_t.pop('Litter')
# data = data.join(pd.DataFrame(enc.fit_transform(data_t.values[:, :]), index=oldindex, columns=['wgain_2bin']))       
# data.loc[data.wgain_2bin == 0, 'wgain_quant'] = 'low'
# data.loc[data.wgain_2bin == 1 , 'wgain_quant'] = 'high'    


# sns.catplot(data=data, x="Group", y="P14",  kind="violin", split=False)
# sns.catplot(data=data, x="Group", y="wgain",  hue="wgain_quant", kind="violin", split=True)
# sns.catplot(data=data, x="Group", y="wgain", hue="percent_of_M", kind="violin", split=True)
# sns.catplot(data=data, x="Group", y="wgain28", hue="percent_of_M", kind="violin", split=True)


# ADD INTERACTIONS (sex:P14 - add just this interaction, sex*P14 - adds interaction + sex + P14 = shorthand)

# # categorical group
# OLS(data, "slopes91 ~ Sex + P14 + Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes91_no_interactions')
# OLS(data, "slopes42 ~ Sex + P14 + Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes42_no_interactions')

# # no additional slopes
# OLS(data, "slopes91 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes91')
# OLS(data, "slopes42 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes42')

# # all available interactions with slopes
# OLS(data, "slopes91 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group) + Sex * P14 * slopes42", 'slopes91_complex')
# OLS(data, "slopes42 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes42_complex')

# # weights
# OLS(data, "P91 ~ P14 + Sex + C(Group) + Litter_size_P14", 'weight_P91')
# OLS(data, "P91 ~ Sex + C(Group) + Litter_size_P14 * P14 + Maternal_age_at_birth", 'weight_P91_complex')
# OLS(data, "P14 ~ Sex + C(Group)", 'weight_P14')
# OLS(data, "P14 ~ Sex + C(Group) + Litter_size_P14 + Maternal_age_at_birth", 'weight_P14_complex')
# OLS(data, "P42 ~ P14 + Sex + C(Group) + Litter_size_P14", 'weight_P42')
# OLS(data, "P42 ~ Sex + C(Group) + Litter_size_P14 * P14 + Maternal_age_at_birth", 'weight_P42_complex')


# 3 SLOPES
# categorical group
OLS(data, "slopes91 ~ Sex + P14 + Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes91_no_interactions')
OLS(data, "slopes49 ~ Sex + P14 + Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes49_no_interactions')
OLS(data, "slopes28 ~ Sex + P14 + Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes28_no_interactions')

# no additional slopes
OLS(data, "slopes91 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes91')
OLS(data, "slopes49 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes49')
OLS(data, "slopes28 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes28')

# all available interactions with slopes
OLS(data, "slopes91 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group) + Sex * P14 * slopes49", 'slopes91_complex')
OLS(data, "slopes49 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group) + Sex * P14 * slopes28", 'slopes49_complex')
OLS(data, "slopes28 ~ Sex + P14 * Litter_size_P14 + Maternal_age_at_birth + C(Group)", 'slopes28_complex')

# weights
OLS(data, "P91 ~ P14 + Sex + C(Group) + Litter_size_P14", 'weight_P91')
OLS(data, "P91 ~ Sex + C(Group) + Litter_size_P14 * P14 + Maternal_age_at_birth", 'weight_P91_complex')
OLS(data, "P14 ~ Sex + C(Group)", 'weight_P14')
OLS(data, "P14 ~ Sex + C(Group) + Litter_size_P14 + Maternal_age_at_birth", 'weight_P14_complex')
OLS(data, "P49 ~ P14 + Sex + C(Group) + Litter_size_P14", 'weight_P49')
OLS(data, "P49 ~ Sex + C(Group) + Litter_size_P14 * P14 + Maternal_age_at_birth", 'weight_P49_complex')
OLS(data, "P28 ~ P14 + Sex + C(Group) + Litter_size_P14", 'weight_P28')
OLS(data, "P28 ~ Sex + C(Group) + Litter_size_P14 * P14 + Maternal_age_at_birth", 'weight_P28_complex')

# OLS(data, "slopes91 ~ P14 + Maternal_age_at_birth + Litter_size_P14 + Group + percent_of_M", 'slopes91')  # SIGNIFICANT if slopes calculated for both sexes together
# OLS(data, "slopes42 ~ P14 + Maternal_age_at_birth + Litter_size_P14 + Group + percent_of_M", 'slopes42')
# OLS(data, "P14 ~ Maternal_age_at_birth + Litter_size_P14 + Group + percent_of_M", 'P14')
# OLS(data, "P91 ~ P14 + slopes91 + slopes42 + Maternal_age_at_birth + Litter_size_P14 + Group + percent_of_M", 'P91')

# def tSNE(data, columns, hue, mydir):
tSNE(data, ['Litter_size_P14', 'Maternal_age_at_birth', 'Sex_num', 'P14','P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91', 'Problem', 'slopes91', 'slopes49', 'slopes28'], 'Group', 'Group', mydir, perplexity=19, three_d=False, z=True)

# tSNE(data, ['Litter_size_P14', 'Maternal_age_at_birth', 'percent_of_M', 'P14','P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91', 'Problem', 'slopes91', 'slopes42'], 'Group', 'Group', mydir, perplexity=20, three_d=False, z=True)


data['PC1'], data['PC2'] = PCA_plot(data, ['Litter_size_P14', 'Maternal_age_at_birth', 'Sex_num', 'P14','P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91', 'Problem', 'slopes91', 'slopes49', 'slopes28'], 'Group', 'Group', mydir, pc='PC1')
data['PC1'], data['PC2'] = PCA_plot(data, ['Litter_size_P14', 'Maternal_age_at_birth', 'Sex_num', 'P14','P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91', 'Problem', 'slopes91', 'slopes49', 'slopes28'], 'Sex', 'Group', mydir, pc='PC1')

data['LD1_group'] = LDA_plot(data, ['Group', 'Litter_size_P14', 'Maternal_age_at_birth', 'Sex_num', 'P14','P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91', 'Problem', 'slopes91', 'slopes49', 'slopes28'], 'Group', 'Group', mydir, scatter=False, plot_PCAxLDA=False, pc='PC1')

data['LD1_sex'] = LDA_plot(data, ['Sex_num', 'Litter_size_P14', 'Maternal_age_at_birth', 'Group', 'P14','P21', 'P28', 'P35', 'P42', 'P49', 'P56', 'P63', 'P70', 'P77', 'P84', 'P91', 'Problem', 'slopes91', 'slopes49', 'slopes28'], 'Sex_num', 'Sex', mydir, scatter=False, plot_PCAxLDA=False, pc='PC1')


# filtered data wo Problem pups
# OLS(data, "wgain ~ P14 + Maternal_age_at_birth + Litter_size_P14 + Group + percent_of_M", 'wgain')
# OLS(data, "P91 ~ P14 + Maternal_age_at_birth + Litter_size_P14 + Group + percent_of_M", 'P91')

# LME makes no sense here with means only
# null = "wgain ~ Group + P14 + Maternal_age_at_birth + Litter_size_P14 + percent_of_M"
# model = "wgain ~  P14 + Maternal_age_at_birth + Litter_size_P14 + percent_of_M"
# LinearMixedModel(null, model, data, 'wgain', 'Group')  

# data.to_csv('data.csv', index=False)
# data.Group.nunique()
