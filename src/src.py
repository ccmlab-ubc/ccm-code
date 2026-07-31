import pandas as pd
import numpy as np
import json

def decode_json(val):
    """
    Decode a JSON-encoded string (if possible).

    Parameters
    ----------
    val : object
        Value to decode.
    
    Returns
    ----------
    object
        Decoded Python object if ``val`` is valid JSON string. Otherwise, returns unchanged val.
    """
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val

def preprocess(file_paths, subjects):
    """
    Load and preprocess experimental data from multiple subjects.

    For each subject, active and passive condition files are loaded,
    JSON-encoded values are decoded, trial numbers and condition labels are
    added, and the two conditions are concatenated into a single DataFrame.
    Localization coordinates are converted to centimetres, reach endpoints
    are extracted (when using the KINARM), and subject ids are added.

    Parameters
    ----------
    file_paths : list of list of str
        Nested list of file paths, where each element contains the active and
        passive data files for one subject.
    subjects : list
        Subject ids corresponding to ``file_paths``.

    Returns
    ----------
    pandas.DataFrame
        Preprocessed dataset containing all subjects and conditions, with
        subject numbers (``SN``), trial numbers (``TN``), condition labels,
        and any derived variables added during preprocessing.
    """

    conditions = ["Active", "Passive"]
    data = [] # Master list of all subject data

    # Loop through subjects
    for k, s in enumerate(subjects):
        dfs = []    # list of dataframes for each participant which holds df for each condition

        # Loop through each participant's active and passive data 
        # and concat into single df
        for p, cond in zip(file_paths[k], conditions):
            df_json = pd.read_csv(p)                        # read in participant's data
            df_temp = df_json.map(decode_json)              # decode JSON strings
            df_temp["Condition"] = cond                        # create condition column
            df_temp["TN"] = np.arange(1, len(df_temp)+1)    # create trial number column
            dfs.append(df_temp)                             # append to dfs
        df_subj = pd.concat(dfs)                            # concat list of dfs into subject df
        
        df_subj[["Endpoint X", "Endpoint Y"]] = df_subj[["X", "Y"]].map(lambda x: x[-1])

        # Add subject info and append to data list
        df_subj["SN"] = k + 1
        df_subj["Subject ID"] = s
        data.append(df_subj)

    # Final concatenation of all subject data
    df = pd.concat(data).reset_index(drop=True)

    # Rearrange columns to have subject and trial numbers in first two columns
    df = df[["TN"] + [cond for cond in df.columns if cond != "TN"]] 
    df = df[["SN"] + [cond for cond in df.columns if cond != "SN"]] 

    # Master list of all subject data
    return df


def remove_outliers(z_thresh, df, reach_col, subj_col, task, abs_cutoff, switch_outlier_with_mean=False):
    """
    Remove outliers from a kinematic variable using within-subject z-scoring.

    Parameters:
    ----------
    z_thresh : float
        The z-score threshold above which data points are considered outliers.
    df : pd.DataFrame
        The input DataFrame containing the kinematic data.
    reach_col : str
        The column name of the kinematic variable to clean (e.g., "theta_maxradv").

    Returns:
    -------
    pd.DataFrame
        The DataFrame with additional columns:
        - <reach_col>_z: z-scored values within each subject (subj_col).
        - <reach_col>_outlier: boolean flag for outliers.
        - <reach_col>_mean: mean of non-outlier trials within each subject.
        - <reach_col>_clean: original values with outliers replaced by NaN.
    """
    
    # hard limit on reach angles above abs_cutoff degrees
    idx_below_cutoff = np.abs(df[f"{reach_col}"]) <= abs_cutoff 
    
    # Calculate z-score hand angle data using non-outlier trials only
    df[f"{reach_col}_z"] = df.loc[idx_below_cutoff].groupby(subj_col)[reach_col].transform(stats.zscore)

    # Create outlier column
    idx_outlier = (~idx_below_cutoff) | (np.abs(df[f"{reach_col}_z"]) > z_thresh)
    df[f"{reach_col}_outlier"] = idx_outlier
    
    # Create col to indicate which trials to exclude from MLE (trials after outlier).
    # Tasks treated differently: adaptation fitting involves predicting next trial; sdt fitting involves predicting
    # current.
    if task == "adapt":
        desired_idx = [idx_outlier[i] or (idx_outlier[i - 1] if i > 0 else False) for i in range(len(idx_outlier))]
    elif task == "sdt":
        desired_idx = idx_outlier
    df["no_fit"] = desired_idx
                                                
    # Calculate within-subject mean using non-outlier trials only
    df[f"{reach_col}_mean"] = df[~idx_outlier].groupby(subj_col)[f"{reach_col}"].transform("mean")

    # Replace outliers with within-subject mean values
    df[f"{reach_col}_mean"] = df.groupby(subj_col)[f"{reach_col}_mean"].transform(lambda x: x.fillna(np.nanmean(x)))

    # Create final column with "cleaned" hand angles OR replace outliers with NaNs
    if switch_outlier_with_mean == True:
        df[f"{reach_col}_clean"] = np.where(df[f"{reach_col}_outlier"] == True,
                                             df[f"{reach_col}_mean"],
                                             df[f"{reach_col}"])
    else:
        df[f"{reach_col}_clean"] = np.where(df[f"{reach_col}_outlier"] == True,
                                            np.nan,
                                            df[f"{reach_col}"])

    return df