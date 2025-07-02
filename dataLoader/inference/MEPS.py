import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys

TASKS = {
    1: "Causal Effect of Insurance on Medical Expenditure", 
    2: "Causal Effect of Poverty on Health",
    3: "Causal Effect of Psychological Distress on Emergency Utilization",
    4: "Causal Effect of Delayed Care on Health",
    5: "Causal Effect of Diabetes on Non-Diabetic Expenditure",
}


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
            selected_cols = ['HINOTCOV', 'EXPTOT', 'AGE', 'SEX', 'RACEA', 'POVCAT', 'HEALTH']
            df = df[selected_cols]

            df = df.dropna(subset=['EXPTOT', 'HINOTCOV'])
            df['EXPTOT_log'] = np.log1p(df['EXPTOT'])
            df['HEALTH'] = df['HEALTH'].fillna(df['HEALTH'].median())
            df["POVCAT"] = df["POVCAT"].fillna(df["POVCAT"].median())
            df = df.reset_index(drop=True)
            
        elif task == 2:
            selected_cols = ['POVCAT', 'HEALTH', 'AGE', 'SEX', 'RACEA', 'EDUCYR', 'WORKEV']
            df = df[selected_cols]

            df.loc[(df['HEALTH']>=7)&(df['HEALTH']<=9), 'HEALTH'] = np.nan
            df = df.dropna(subset=['HEALTH'])
            df['EDUCYR'] = df['EDUCYR'].fillna(df['EDUCYR'].median())
            df['WORKEV'] = df['WORKEV'].fillna(df['WORKEV'].median())
            df = df.reset_index(drop=True)

        elif task == 3:
            selected_cols = ['K6SUM', 'ERTOTVIS', 'AGE', 'SEX', 'POVCAT', 'HINOTCOV', 'HEALTH']
            df = df[selected_cols]

            df = df[df["AGE"]>=18]
            df.loc[df['K6SUM'] == 96, 'K6SUM'] = np.nan
            df.loc[df['K6SUM'] == 98, 'K6SUM'] = np.nan
            df = df.dropna(subset=['K6SUM', 'ERTOTVIS'])
            df['HEALTH'] = df['HEALTH'].fillna(df['HEALTH'].median())
            df['POVCAT'] = df['POVCAT'].fillna(df['POVCAT'].median())
            df['HINOTCOV'] = df['HINOTCOV'].fillna(df['HINOTCOV'].median())
            df = df.reset_index(drop=True)

        elif task == 4:
            selected_cols = ['DELAYCOST', 'YEAR', 'HEALTH', 'AGE', 'HEALTH', 'POVCAT', 'HINOTCOV', 'HEALTH_NEXT_YEAR']
            df = df.sort_values(by=['MEPSID', 'YEAR'])
            df['HEALTH_NEXT_YEAR'] = df.groupby('MEPSID')['HEALTH'].shift(-1)
            df = df[selected_cols]
            
            df.loc[(df['DELAYCOST']>=7)&(df['DELAYCOST']<=9), 'DELAYCOST'] = np.nan
            df = df.dropna(subset=['DELAYCOST', 'HEALTH_NEXT_YEAR'])
            df['POVCAT'] = df['POVCAT'].fillna(df['POVCAT'].median())
            df['HINOTCOV'] = df['HINOTCOV'].fillna(df['HINOTCOV'].median())
            df = df.reset_index(drop=True)

        elif task == 5:
            selected_cols = ['DIABETICEV', 'EXPTOT', 'RXEXPTOT', 'AGE', 'SEX', 'POVCAT', 'HINOTCOV', 'HEALTH']
            df = df[selected_cols]
            
            df.loc[(df['DIABETICEV']>=7)&(df['DIABETICEV']<=9), 'DIABETICEV'] = np.nan
            df = df.dropna(subset=['DIABETICEV', 'EXPTOT', 'RXEXPTOT'])
            df['EXPTOT-RXEXPTOT'] = df.apply(lambda row: row['EXPTOT']-row['RXEXPTOT'], axis=1)
            df['HINOTCOV'] = df['HINOTCOV'].fillna(df['HINOTCOV'].median())
            df['POVCAT'] = df['POVCAT'].fillna(df['POVCAT'].median())
            df['HEALTH'] = df['HEALTH'].fillna(df['HEALTH'].median())
            df = df.reset_index(drop=True)

    dataset = df.to_dict(orient="records")
    return dataset