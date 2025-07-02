import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys

TASKS = {
    1: "Predict the severity of mental health symptoms (Worry/Anxiety)", 
    2: "Predict the number of chronic conditions",
    3: "Predict the frequency of joint pain in the past 30 days",
    4: "Predict an individual's health status",
    5: "Predict the age of onset of first regular smoking",
    6: "Predict the impact of mental health on perceived physical health",
    7: "Predict the number of work loss days based on various factors",
    8: "Predict fruit consumption levels",
    9: "Predict the likelihood of delaying medical care due to cost",
    10: "Predict the age when someone was first diagnosed with a chronic condition (Diabetes)",
}


def calculate_num_chronic(df):
    df_copy = df.copy()
    chronic_vars = ['ADDEV', 'ANGIPECEV', 'ARTHGLUPEV', 'ASTHMAEV', 'HYP2TIME']
    temp_binary_cols = []
    for var in chronic_vars:
        bin_col_name = f'{var}_bin'
        temp_binary_cols.append(bin_col_name)
        conditions_to_nan = df_copy[var].isin([7, 8, 9])
        df_copy.loc[conditions_to_nan, var] = np.nan
        df_copy[bin_col_name] = np.where(df_copy[var] == 2, 1, 0)
    diabetes_bin_col = 'DIABETES_bin'
    temp_binary_cols.append(diabetes_bin_col)
    diabetes_age_to_nan = df_copy['DIABETICAGE'].between(93, 99, inclusive='both')
    df_copy.loc[diabetes_age_to_nan, 'DIABETICAGE'] = np.nan
    df_copy[diabetes_bin_col] = np.where(df_copy['DIABETICAGE'] <= 85, 1, 0)
    df_copy['NUM_CHRONIC'] = df_copy[temp_binary_cols].sum(axis=1)
    return df_copy['NUM_CHRONIC']


def calculate_mental_health_score(df):
    df_copy = df.copy()
    freq_map = {
        5: 0,  # Never
        4: 1,  # Few times/year
        3: 2,  # Monthly
        2: 3,  # Weekly
        1: 4,  # Daily
        # Special codes to be mapped to 0
        0: 0,
        7: 0,
        8: 0,
        9: 0
    }
    level_map = {
        4: 0,  # Not at all
        3: 2,  # Between little and lot
        1: 1,  # A little
        2: 3,  # A lot
        # Special codes to be mapped to 0
        0: 0,
        7: 0,
        8: 0,
        9: 0
    }
    temp_cols = ['WORFREQ_NUM', 'DEPFREQ_NUM', 'WORFEELEVL_NUM', 'DEPFEELEVL_NUM']
    df_copy['WORFREQ_NUM'] = df_copy['WORFREQ'].map(freq_map).fillna(0)
    df_copy['DEPFREQ_NUM'] = df_copy['DEPFREQ'].map(freq_map).fillna(0)
    df_copy['WORFEELEVL_NUM'] = df_copy['WORFEELEVL'].map(level_map).fillna(0)
    df_copy['DEPFEELEVL_NUM'] = df_copy['DEPFEELEVL'].map(level_map).fillna(0)
    df_copy['MENTAL_HEALTH_SCORE'] = df_copy[temp_cols].sum(axis=1)
    df_copy.drop(columns=temp_cols, inplace=True)
    df_copy['MENTAL_HEALTH_SCORE'] = df_copy['MENTAL_HEALTH_SCORE'].astype(int)
    return df_copy['MENTAL_HEALTH_SCORE']


def getNHIS(path: str, variables: list=None, task: int=None):
    # How to attach data to be decided
    return datasetLoad(path=path, variables=variables, task=task)


def datasetLoad(path, variables, task):
    try:
        # load user's dataset
        if os.path.isfile(path):
            print_sys(f"Loading data from {path}")
            dataset = loadUserFiles(path, variables=variables)
            return dataset
        else:
            datasetPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../defaultData/nhis/nhis_00007.txt")
            if os.path.exists(datasetPath):
                print_sys("Found local default dataset...")
                if task is None:
                    if variables is None:
                        print_sys(f"Loading whole default dataset")
                    else:
                        print_sys(f"Loading whole default dataset with columns {variables}")
                    return loadLocalFiles(datasetPath, variables=variables, task=task)
                else:
                    if task not in TASKS:
                        raise AttributeError(f"Please select a id for predefined task in {TASKS} or set task as none for the whole default dataset")
                    else:
                        print_sys(f"Loading data from default dataset for task {TASKS[task]}")
                        return loadLocalFiles(datasetPath, variables=variables, task=task)
            else:
                raise FileExistsError("Local default data not exits...")
    except Exception as e:
        print_sys(f"error: {e}")


def loadUserFiles(path, variables):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise AttributeError(f"Current path {path} is not valid cav file, please check again")
    
    if variables is None:
        return df
    else:
        loaded_columns = set(df.columns)
        required_set = set(variables)
        if not required_set.issubset(loaded_columns):
            missing_columns = required_set - loaded_columns
            raise AttributeError(f"Your selected variables {variables} contained columns that are not included in your file: {missing_columns}")
        df = df[variables]
        return df


def loadLocalFiles(path, variables=None, task=None):
    df = pd.read_csv(path)
    if task is None:
        if variables is None:
            df = df
        else:
            loaded_columns = set(df.columns)
            required_set = set(variables)
            if not required_set.issubset(loaded_columns):
                missing_columns = required_set - loaded_columns
                raise AttributeError(f"Your selected variables {variables} contained columns {missing_columns} that are not included in the default dataset, please refer to docs for detailed information.")
            df = df[variables]
    else:
        if task == 1:
            selected_cols = ['WORFEELEVL', 'AGE', 'SEX', 'EDUC', 'NUM_CHRONIC', 'WORFREQ', 'WORRX']
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)
            
        elif task == 2:
            selected_cols = ['NUM_CHRONIC', 'AGE', 'SEX', 'RACEA', 'EDUC', 'FAMTOTINC', 'SMOKEV', 'BMICAT', 'HEALTH']
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['AGE', 'SEX', 'RACEA', 'EDUC', 'SMOKEV', 'BMICAT', 'HEALTH'])
            df = df.reset_index(drop=True)

        elif task == 3:
            selected_cols = ['JNTMO', 'AGE', 'SEX', 'BMICAT', 'NUM_CHRONIC', 'ARTHGLUPEV']
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 4:
            selected_cols = ['HEALTH', 'AGE', 'SEX', 'BMICAT', 'SMOKEV', 'ALCSTAT1', 'NUM_CHRONIC', 'MENTAL_HEALTH_SCORE']
            df['MENTAL_HEALTH_SCORE'] = calculate_mental_health_score(df)
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 5:
            selected_cols = ['SMOKAGEREG', 'AGE', 'EDUC', 'POORYN', 'ALCSTAT1']
            df = df[selected_cols]
            
            df = df[df["SMOKAGEREG"]<96]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 6:
            selected_cols = ['HEALTH', 'MENTAL_HEALTH_SCORE', 'AGE', 'SEX', 'NUM_CHRONIC', 'HINOTCOVE']
            df['MENTAL_HEALTH_SCORE'] = calculate_mental_health_score(df)
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 7:
            selected_cols = ['WLDAYR', 'AGE', 'HEALTH', 'NUM_CHRONIC', 'MENTAL_HEALTH_SCORE']
            df['MENTAL_HEALTH_SCORE'] = calculate_mental_health_score(df)
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 8:
            selected_cols = ['FRUTNO', 'AGE', 'SEX', 'EDUC', 'FAMTOTINC']
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['FRUTNO', 'AGE', 'SEX', 'EDUC'])
            df = df.reset_index(drop=True)

        elif task == 9:
            selected_cols = ['DELAYCOST', 'HINOTCOVE', 'FAMTOTINC']
            df = df[df["AGE"]>=18]
            df = df[selected_cols]

            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 10:
            selected_cols = ['DIABETICAGE', 'AGE', 'BMICAT', 'FAMTOTINC']
            df = df[selected_cols]

            df = df[df['DIABETICAGE']<93]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

    dataset = df.to_dict(orient="records")
    return dataset