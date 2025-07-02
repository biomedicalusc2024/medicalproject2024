import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys

TASKS = {
    1: "Assess the causal effect of the number of chronic conditions on mental health severity", 
    2: "Investigate the causal impact of reporting frequent mental health issues on work loss days",
    3: "Examine the causal effect of having public health insurance on the likelihood of having hypertension diagnosed",
    4: "Assess the causal effect of smoking on the frequency of feeling depressed",
    5: "Investigate the causal effect of ever being told they have arthritis on work limitation",
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
    

def calculate_has_mental_health_issue(df):
    df_copy = df.copy()
    worried_frequent = df_copy['WORFREQ'].isin([1, 2])
    depressed_frequent = df_copy['DEPFREQ'].isin([1, 2])
    df_copy['HAS_MENTAL_HEALTH_ISSUE'] = (worried_frequent | depressed_frequent).astype(int)
    return df_copy['HAS_MENTAL_HEALTH_ISSUE']


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
            selected_cols = ['NUM_CHRONIC', 'WORFEELEVL', 'AGE', 'SEX', 'EDUC', 'SMOKEV', 'BMICAT', 'HEALTH']
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)
            
        elif task == 2:
            selected_cols = ['HAS_MENTAL_HEALTH_ISSUE', 'WLDAYR', 'AGE', 'SEX', 'NUM_CHRONIC', 'HEALTH']
            df['HAS_MENTAL_HEALTH_ISSUE'] = calculate_has_mental_health_issue(df)
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 3:
            selected_cols = ['HIPUBCOVE', 'HYP2TIME', 'AGE', 'SEX', 'EDUC', 'FAMTOTINC', 'DELAYCOST']
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['HIPUBCOVE', 'HYP2TIME'])
            df = df.reset_index(drop=True)

        elif task == 4:
            selected_cols = ['SMOKEV', 'DEPFREQ', 'AGE', 'SEX', 'ALCSTAT1', 'NUM_CHRONIC']
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 5:
            selected_cols = ['ARTHGLUPEV', 'ARTHLIMIT', 'AGE', 'SEX', 'BMICAT', 'HEALTH']
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

    dataset = df.to_dict(orient="records")
    return dataset