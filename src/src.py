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

def pre_process(file_paths, subjects, joystick=False):
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
    joystick : bool, default=False
        If ``False``, reach endpoints are computed from the recorded
        trajectories. If ``True``, endpoint computation is skipped.

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

        # Loop through each subjects active and passive data 
        # and concat into single df
        for p, c in zip(file_paths[k], conditions):
            df_json = pd.read_csv(p)                        # read in participant's data
            df_temp = df_json.map(json_decode)              # decode JSON strings
            df_temp["Condition"] = c                        # create condition column
            df_temp["TN"] = np.arange(1, len(df_temp)+1)    # create trial number column
            dfs.append(df_temp)                             # append to dfs
        df_subj = pd.concat(dfs)                            # concat list of dfs into subject df

        # Convert units to CM
        if all(col in df_subj.columns for col in ["Localize X", "Localize Y"]):
            df_subj[["Localize X", "Localize Y"]] = df_subj[["Localize X", "Localize Y"]].map(lambda x: x*100)

        # Calculate endpoints
        if not joystick:
            df_subj[["Endpoint X", "Endpoint Y"]] = df_subj[["X", "Y"]].map(lambda x: x[-1])

        # Add subject info and append to data list
        df_subj["SN"] = k + 1
        df_subj["Subject ID"] = s
        data.append(df_subj)

    # Final concatenation of all subject data
    df = pd.concat(data).reset_index(drop=True)

    # Rearrange columns to have subject and trial numbers in first two columns
    df = df[["TN"] + [c for c in df.columns if c != "TN"]] 
    df = df[["SN"] + [c for c in df.columns if c != "SN"]] 

    # Master list of all subject data
    return df