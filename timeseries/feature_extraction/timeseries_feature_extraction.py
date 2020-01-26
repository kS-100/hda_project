import pandas as pd
from tqdm.notebook import tqdm as tqdm
import multiprocessing
from functools import reduce
import numpy as np
from tsfresh.feature_extraction.settings import MinimalFCParameters

def get_sub_timeseries(df_x, index, window_start, window_end, identifier):
    """
    Helper method which extracts a sub dataframe of a pandas dataframe. The sub dataframe is defined by a
    index aswell as the relative start end end index of the window. An identifier column is added with a
    constant value which is given as parameter. This ensures multiple sub dataframes can be distinguished
    from each other.

    Example:
    -----[|-----------|-----------|----------|]------------|-------------|---------------->
      index-window_end                 index-window_start              index

    Because iloc is not supported in dask dataframes it is assumed the index equals the level of the row
    (like reset_index does).
    :param df_x: the pandas dataframe the sub dataframe should be extracted
    :param index: absolute index of the dataframe the window is extracted
    :param window_start: relative start of the subwindow
    :param window_end: relative end of the subwindow
    :param identifier: a unique constant identifier to distinguish later the sub dataframe
    :return: the extracted sub dataframe
    """
    sub_df_x = df_x.iloc[index - window_end:index - window_start]
    sub_df_x['window_id'] = identifier
    return sub_df_x


def get_rolling_timeseries(df_x, start_index, lag, window_start, window_end):
    """
    Extracts all possible sub windows of the given dataframe. It is assumed that the index of the dataframe is
    the default one (reset_index()).
    Example:
    -----[|[[...-------|-----------|----------|]]]...--------|-------------|-------------|------->
         |<---    window_end-window_start  --->|           index
         |<---                 window_end               --->|
                                             |<---window--->|
                                                 _start
                                                            |<---        lag        -->|
    :param df_x: pandas dataframe where the sub windows are extracted.
    :param start_index: the first index the sub windows should be extracted.
                        This is necessary because the method can be applied multiple times with different windows.
                        To merge the extracted features later on the window id must match.
    :param lag: the distance between the current row and the target row (y). necessary to limit the number of windows
                at the end of the dataframe where an extraction of sub windows would be possible but no target row
                is available.
    :param window_start: relative distance between the current row and the start of the sub windows
    :param window_end: relative distance between the current row and the start of the sub windows
    :param npartitions: (dask parameter) the number of partitions used for the dataframe (used for parallelization).
                        According to stackoverflow the number should be a multiple of the number of processors
                        (default = 1xcpu_count)
    :return: a pandas dataframe containing all sub windows each with a unique window id (ascending numbers from 0 to #windows)
    """
    print("Extracting sub windows", window_start, "-", window_end, ":")
    # extract every possible sub window
    sub_dfs = [get_sub_timeseries(
                    df_x=df_x,
                    index=i,
                    window_start=window_start,
                    window_end=window_end,
                    identifier=i - start_index -1) for i in
               tqdm(range(start_index, len(df_x) - lag))]
    sub_df_x_comp = pd.concat([df for df in tqdm(sub_dfs)], ignore_index=True)
    return sub_df_x_comp

def extract_sub_window(df_x, y, window, start_index, lag, fc_parameters=MinimalFCParameters(), n_jobs=-1):
    from tsfresh import extract_relevant_features
    window_start, window_end = window
    sub_df_x = get_rolling_timeseries(df_x, start_index, lag, window_start, window_end)
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    y = y[y.index.isin(sub_df_x.window_id)]
    features = extract_relevant_features(sub_df_x, y, column_id="window_id", column_sort="timestamp", column_value=None,
                                         default_fc_parameters=fc_parameters, n_jobs=n_jobs)
    features = features.add_suffix(f"_{window_start}_{window_end}")
    return (features, y)


def extract_sub_windows(df_x, df_y, window_array, lag, fc_parameters, n_jobs=-1):
    df_x = df_x.reset_index('timestamp')

    split_func = lambda x: list(map(int, x.split("-")))
    windows = np.array(list(map(split_func, window_array)))
    max_end = max(windows[:, 1])

    y = df_y.iloc[max_end + lag:len(df_y)]
    y = y.reset_index(drop=True)
    y.index.name = 'window_id'
    features = [extract_sub_window(df_x, y, window, max_end, lag, fc_parameters, n_jobs) for window in windows]
    features = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'),
                      features)
    return features, y
