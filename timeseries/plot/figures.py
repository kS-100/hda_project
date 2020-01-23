# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_timeseries(dataframe, title_addition = "", savePath = "", units = "value"):
    """
    Generates one plot for each column in dataframe
    If savePath is not modified, plots will not be saved; else saved as pdf
    """
    if units == "value":
        units = ["value"]*dataframe.shape[1] 
    for index, col in enumerate(dataframe.columns):
        plt.figure(figsize=(25, 5))
        g=sns.set(style='white',font_scale=1.5)
        g = sns.lineplot(data=dataframe[col])
        sns.despine()
        g.set(xlabel='Date', ylabel=units[index] ,title=col + " " + title_addition)
        if savePath != "":
            g.figure.savefig(savePath + "ts_" + col + ".pdf")
            
def plot_distribution(dataframe, title_addition = "", savePath = "", units = "value"):
    """
    Generates one plot for each column in dataframe
    If savePath is not modified, plots will not be saved; else saved as pdf
    """
    dataframe = pd.DataFrame(dataframe)
    if units == "value":
        units = ["value"]*dataframe.shape[1] 
    for index, col in enumerate(dataframe.columns):
        g=sns.set(style='white',font_scale=1.5)
        g = sns.distplot(dataframe[col].dropna(axis = 0), hist = True, norm_hist = True)
        sns.despine()
        g.set(xlabel=units[index], ylabel= "rate" ,title="Distribution of " + col + " " + title_addition)
        plt.show()
        if savePath != "":
            g.figure.savefig(savePath + "ts_" + col + ".pdf")