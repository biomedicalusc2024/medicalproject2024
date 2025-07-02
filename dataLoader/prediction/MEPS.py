import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys

TASKS = {
    1: "Predict Annual Total Medical Expenditure", 
    2: "Predict Prescription Drug Expenditure",
    3: "Predict Hospital Days",
    4: "Predict Psychological Distress Score",
    5: "Predict Number of Chronic Conditions",
    6: "Predict Office Visits",
    7: "Predict Pain Interference Level",
    8: "Predict Number of Prescription Medications",
    9: "Predict Family Total Income",
    10: "Predict Hypertension Diagnosis Age",
}


# Major chronic conditions count
chronic_conditions = ['DIABETICEV', 'HYPERTENEV', 'CHOLHIGHEV', 'ARTHGLUPEV', 
                     'ASTHMAEV', 'HEARTCONEV', 'STROKEV', 'CANCEREV']
def create_chronic_count(df):
    """Create chronic condition count variable"""
    count = 0
    for condition in chronic_conditions:
        if condition in df.columns:
            # Count conditions coded as 2 (Yes)
            count += (df[condition] == 2).astype(int)
    
    # Set to NA if all conditions are missing
    all_missing = df[chronic_conditions].isna().all(axis=1)
    count.loc[all_missing] = np.nan
    return count


# K6 Scale Construction
k6_items = ['ANERVOUS', 'AHOPELESS', 'ARESTLESS', 'ASAD', 'AEFFORT', 'AWORTHLESS']
def create_k6sum(df):
    """Create K6 psychological distress scale"""
    # Recode special missing values
    k6_data = df[k6_items].copy()
    k6_data = k6_data.replace([96, 98], np.nan)  # Don't know, Refused
    # Reverse score: 1→4, 2→3, 3→2, 4→1, 5→0
    reverse_map = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    k6_reversed = k6_data.replace(reverse_map)
    # Sum scores (range: 0-24)
    k6sum = k6_reversed.sum(axis=1)
    # Set to NA if more than 2 items missing
    missing_count = k6_data.isna().sum(axis=1)
    k6sum.loc[missing_count > 2] = np.nan
    return k6sum


def create_phq2(df):
    """Create PHQ-2 depression screening score"""
    phq2_items = ['ADPRST', 'ADDPRS']  # Depression screening items
    # Recode missing values
    phq2_data = df[phq2_items].copy()
    phq2_data = phq2_data.replace([96, 98], np.nan)
    # Sum scores: 0-6 scale
    phq2_score = phq2_data.sum(axis=1)
    # Set to NA if any item missing
    any_missing = phq2_data.isna().any(axis=1)
    phq2_score.loc[any_missing] = np.nan
    return phq2_score


def getMEPS(path: str, variables: list=None, task: int=None):
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
            datasetPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../defaultData/meps/meps_00093.txt")
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
            selected_cols = ["EXPTOT", "AGE", "SEX", "RACEA", "HISPYN", "EDUCYR", "POVCAT", "HEALTH"]
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['EXPTOT'])
            df['EXPTOT_log'] = np.log1p(df['EXPTOT'])
            df['EDUCYR'] = df['EDUCYR'].fillna(df['EDUCYR'].median())
            df["POVCAT"] = df["POVCAT"].fillna(df["POVCAT"].median())
            df = df.reset_index(drop=True)
            
        elif task == 2:
            selected_cols = ["RXEXPTOT", "AGE", "SEX", "HINOTCOV", "CHRONIC_COUNT"]
            df["CHRONIC_COUNT"] = create_chronic_count(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['RXEXPTOT'])
            df['RXEXPTOT_log'] = np.log1p(df['RXEXPTOT'])
            df['HINOTCOV'] = df['HINOTCOV'].fillna(df['HINOTCOV'].median())
            df['CHRONIC_COUNT'] = df['CHRONIC_COUNT'].fillna(df['CHRONIC_COUNT'].median())
            df = df.reset_index(drop=True)

        elif task == 3:
            selected_cols = ['HPTOTNIGHT', 'AGE', 'SEX', 'HEALTH', 'CHRONIC_COUNT']
            df["CHRONIC_COUNT"] = create_chronic_count(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df.loc[df['HPTOTNIGHT'] == 998, 'HPTOTNIGHT'] = np.nan
            df = df.dropna(subset=['HPTOTNIGHT'])
            df['HEALTH'] = df['HEALTH'].fillna(df['HEALTH'].median())
            df['CHRONIC_COUNT'] = df['CHRONIC_COUNT'].fillna(df['CHRONIC_COUNT'].median())
            df = df.reset_index(drop=True)

        elif task == 4:
            selected_cols = ['K6SUM', 'AGE', 'SEX', 'POVCAT', 'MARSTAT', 'WORKEV', 'HEALTH']
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df.loc[df['K6SUM'] == 96, 'K6SUM'] = np.nan
            df.loc[df['K6SUM'] == 98, 'K6SUM'] = np.nan
            df = df.dropna(subset=['K6SUM'])
            df['POVCAT'] = df['POVCAT'].fillna(df['POVCAT'].median())
            df['MARSTAT'] = df['MARSTAT'].fillna(df['MARSTAT'].median())
            df['WORKEV'] = df['WORKEV'].fillna(df['WORKEV'].median())
            df['HEALTH'] = df['HEALTH'].fillna(df['HEALTH'].median())
            df = df.reset_index(drop=True)

        elif task == 5:
            selected_cols = ['CHRONIC_COUNT', 'AGE', 'SEX', 'RACEA', 'EDUCYR', 'POVCAT']
            df["CHRONIC_COUNT"] = create_chronic_count(df)
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['CHRONIC_COUNT'])
            df['EDUCYR'] = df['EDUCYR'].fillna(df['EDUCYR'].median())
            df['POVCAT'] = df['POVCAT'].fillna(df['POVCAT'].median())
            df = df.reset_index(drop=True)

        elif task == 6:
            selected_cols = ['OBTOTVIS', 'AGE', 'HEALTH', 'CHRONIC_COUNT', 'K6SUM']
            df["CHRONIC_COUNT"] = create_chronic_count(df)
            df = df[selected_cols]
            
            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['OBTOTVIS'])
            df['HEALTH'] = df['HEALTH'].fillna(df['HEALTH'].median())
            df['CHRONIC_COUNT'] = df['CHRONIC_COUNT'].fillna(df['CHRONIC_COUNT'].median())
            df = df.reset_index(drop=True)

        elif task == 7:
            selected_cols = ['ADPAIN', 'AGE', 'SEX', 'HEALTH', 'CHRONIC_COUNT']
            df["CHRONIC_COUNT"] = create_chronic_count(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df.loc[(df['ADPAIN']>=6)&(df['ADPAIN']<=9), 'ADPAIN'] = np.nan
            df = df.dropna(subset=['ADPAIN'])
            df['HEALTH'] = df['HEALTH'].fillna(df['HEALTH'].median())
            df['CHRONIC_COUNT'] = df['CHRONIC_COUNT'].fillna(df['CHRONIC_COUNT'].median())
            df = df.reset_index(drop=True)

        elif task == 8:
            selected_cols = ['RXPRMEDSNO', 'AGE', 'SEX', 'HINOTCOV', 'CHRONIC_COUNT']
            df["CHRONIC_COUNT"] = create_chronic_count(df)
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df = df.dropna(subset=['RXPRMEDSNO'])
            df['HINOTCOV'] = df['HINOTCOV'].fillna(df['HINOTCOV'].median())
            df['CHRONIC_COUNT'] = df['CHRONIC_COUNT'].fillna(df['CHRONIC_COUNT'].median())
            df = df.reset_index(drop=True)

        elif task == 9:
            selected_cols = ['FTOTVAL', 'AGE', 'SEX', 'EDUCYR', 'WORKEV', 'RACEA', 'HISPYN']
            df = df[selected_cols]

            df['FTOTVAL_log'] = np.log1p(df['FTOTVAL'])
            df['EDUCYR'] = df['EDUCYR'].fillna(df['EDUCYR'].median())
            df['WORKEV'] = df['WORKEV'].fillna(df['WORKEV'].median())
            df['HISPYN'] = df['HISPYN'].fillna(df['HISPYN'].median())
            df = df.reset_index(drop=True)

        elif task == 10:
            selected_cols = ['HYPERTENAGE', 'AGE', 'SEX', 'RACEA', 'POVCAT']
            df = df[df["HYPERTENEV"]==2]
            df = df[selected_cols]

            df.loc[(df['HYPERTENAGE']>=96)&(df['HYPERTENAGE']<=99), 'HYPERTENAGE'] = np.nan
            df = df.dropna(subset=['HYPERTENAGE'])
            df = df.reset_index(drop=True)

    dataset = df.to_dict(orient="records")
    return dataset