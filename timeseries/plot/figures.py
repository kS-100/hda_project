# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_timeseries_single(dataframe, title_addition="", savePath="", units="value"):
    """
    Generates one plot for each column in dataframe
    If savePath is not modified, plots will not be saved; else saved as pdf
    """
    dataframe = pd.DataFrame(dataframe)
    if units == "value":
        units = ["value"]*dataframe.shape[1]
    for index, col in enumerate(dataframe.columns):
        plt.figure(figsize=(20, 5))
        g = sns.set(style='white', font_scale=1.5)
        g = sns.lineplot(data=dataframe[col])
        sns.despine()
        g.set(xlabel='Date', ylabel=units[index], title=col + " " + title_addition)
        if savePath != "":
            g.figure.savefig(savePath + "ts_" + col + ".pdf")


def plot_timeseries_multiple(dataframe, title_addition="", savePath="", units="value"):
    """
    Generates one plot for all columns in dataframe
    If savePath is not modified, plots will not be saved; else saved as pdf
    """
    dataframe = pd.DataFrame(dataframe)
    if units == "value":
        units = ["value"] * dataframe.shape[1]

    plt.figure(figsize=(20, 5))
    g = sns.set(style='white', font_scale=1.5)
    for index, col in enumerate(dataframe.columns):
        g = sns.lineplot(data=dataframe[col], label=col)
    sns.despine()
    g.legend()
    g.set(xlabel='Date', ylabel=units[index], title="Comparison" + " " + title_addition)
    if savePath != "":
        g.figure.savefig(savePath + "ts_Comparison" + col + title_addition + ".pdf")


def plot_distribution(dataframe, title_addition="", savePath="", units="value", xlim=""):
    """
    Generates one plot for each column in dataframe
    If savePath is not modified, plots will not be saved; else saved as pdf
    """
    dataframe = pd.DataFrame(dataframe)
    if units == "value":
        units = ["value"]*dataframe.shape[1] 
    for index, col in enumerate(dataframe.columns):
        g = plt.gcf().subplots_adjust(bottom=0.15)
        g = sns.set(style='white',font_scale=1.5)
        g = sns.distplot(dataframe[col].dropna(axis = 0), hist = True, norm_hist = True)
        sns.despine()
        g.set(xlabel=units[index], ylabel="rate", title="Distribution of " + col + " " + title_addition)
        if xlim != "":
            plt.xlim(xlim[0], xlim[1])
        plt.show()
        if savePath != "":
            g.figure.savefig(savePath + "ts_" + col + "_" + title_addition + ".pdf")