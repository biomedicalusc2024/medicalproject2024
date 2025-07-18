import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys

TASKS = {
    1: "Identify individuals with multiple chronic conditions", 
    2: "Predict if an individual has any mental health issue",
    3: "Classify individuals with specific chronic conditions (Diabetes)",
    4: "Predict high healthcare utilization",
    5: "Classify individuals based on their combined health and mental well-being",
    6: "Predict limitation in activities due to any health issue",
    7: "Identify individuals with a potential anxiety and depression comorbidity",
    8: "Classify individuals at risk of severe mental distress",
    9: "Predict flu vaccination uptake",
    10: "Classify employment status based on health and demographics",
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


def calculate_has_multiple_chronic(df):
    df_copy = df.copy()
    chronic_vars = ['ADDEV', 'ANGIPECEV', 'ARTHGLUPEV', 'ASTHMAEV', 'HYP2TIME']
    temp_cols = []
    for var in chronic_vars:
        bin_col_name = f'{var}_bin'
        temp_cols.append(bin_col_name)
        df_copy[bin_col_name] = np.where((df_copy[var] == 2) & (~df_copy[var].isin([7, 8, 9])), 1, 0)
    diabetes_bin_col = 'DIABETES_bin'
    temp_cols.append(diabetes_bin_col)
    df_copy[diabetes_bin_col] = np.where((df_copy['DIABETICAGE'] <= 85) & (~df_copy['DIABETICAGE'].between(93, 99)), 1, 0)
    num_chronic_col = 'NUM_CHRONIC'
    df_copy[num_chronic_col] = df_copy[temp_cols].sum(axis=1)
    temp_cols.append(num_chronic_col)
    df_copy['HAS_MULTIPLE_CHRONIC'] = np.where(df_copy[num_chronic_col] > 1, 1, 0)
    df_copy.drop(columns=temp_cols, inplace=True)
    return df_copy['HAS_MULTIPLE_CHRONIC']


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
            selected_cols = ['HAS_MULTIPLE_CHRONIC', 'AGE', 'SEX', 'RACEA', 'EDUC', 'POORYN', 'SMOKEV', 'ALCSTAT1', 'BMICAT', 'HINOTCOVE']
            df['HAS_MULTIPLE_CHRONIC'] = calculate_has_multiple_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)
            
        elif task == 2:
            selected_cols = ['HAS_MENTAL_HEALTH_ISSUE', 'AGE', 'SEX', 'EDUC', 'MARSTCUR', 'HEALTH', 'NUM_CHRONIC', 'ANXIETYEV', 'DEPRX']
            df['HAS_MENTAL_HEALTH_ISSUE'] = calculate_has_mental_health_issue(df)
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 3:
            selected_cols = ['DIABETES', 'DIABETICAGE', 'AGE', 'SEX', 'BMICAT', 'HYP2TIME', 'ARTHGLUPEV', 'SMOKEV']
            df['DIABETES'] = df['DIABETICAGE'].apply(lambda x: int(x<93))
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 4:
            selected_cols = ['HIGH_UTILIZATION', 'BEDAYR2', 'AGE', 'SEX', 'NUM_CHRONIC', 'HEALTH', 'HINOTCOVE']
            df['HIGH_UTILIZATION'] = df['BEDAYR2'].isin([3, 4, 5])
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 5:
            selected_cols = ['HEALTH', 'WORFREQ', 'DEPFREQ', 'AGE', 'SEX', 'EDUC', 'BMICAT', 'SMOKEV']
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 6:
            selected_cols = ['LANY_LIMITED', 'LANY', 'AGE', 'NUM_CHRONIC', 'MENTAL_HEALTH_SCORE', 'ARTHLIMIT', 'HEALTH']
            df['MENTAL_HEALTH_SCORE'] = calculate_mental_health_score(df)
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df['LANY_LIMITED'] = df['LANY'].apply(lambda x: int(x==10))
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 7:
            selected_cols = ['COMORBIDITY', 'ANXIETYEV', 'DEPFREQ', 'AGE', 'SEX', 'WORRX', 'DEPRX']
            df['COMORBIDITY'] = df.apply(lambda row: int((row['ANXIETYEV']==2) and (row['DEPFREQ'] in [1, 2])), axis=1)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 8:
            selected_cols = ['HIGH_DISTRESS', 'WORFEELEVL', 'DEPFEELEVL', 'AGE', 'SEX', 'EDUC', 'FAMTOTINC']
            df['HIGH_DISTRESS'] = df.apply(lambda row: int((row['WORFEELEVL']==1) or (row['DEPFEELEVL']==1)), axis=1)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 9:
            selected_cols = ['VACCINATED', 'VACFLU12M', 'AGE', 'SEX', 'NUM_CHRONIC', 'HINOTCOVE', 'EDUC', 'HEALTH']
            df['VACCINATED'] = df['VACFLU12M'].apply(lambda x: int(x in [2,3]))
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

        elif task == 10:
            selected_cols = ['EMPLOYED', 'EMPSTAT', 'AGE', 'SEX', 'EDUC', 'HEALTH', 'NUM_CHRONIC', 'MENTAL_HEALTH_SCORE']
            df['MENTAL_HEALTH_SCORE'] = calculate_mental_health_score(df)
            df['NUM_CHRONIC'] = calculate_num_chronic(df)
            df['EMPLOYED'] = df['EMPSTAT'].apply(lambda x: int(x>=100 and x<=122))
            df = df[selected_cols]

            df = df[(df['AGE']>=18)&(df['AGE']<=64)]
            df = df.dropna(subset=selected_cols)
            df = df.reset_index(drop=True)

    dataset = df.to_dict(orient="records")
    return dataset