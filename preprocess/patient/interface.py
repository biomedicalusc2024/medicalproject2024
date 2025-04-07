import os
from .tabular_utils import MinMaxScaler
from .tabular_utils import read_csv_to_df
from .patient_data import TabularPatientBase
from .trial_simulation.data import TabularPatient
from .demo_data import load_trial_patient_tabular

def get_tabular_patient(file_path=None, metadata=None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "demo_data", "trial_patient_data", "data_processed.csv")
    else:
        # check user's file is valid
        pass
    if metadata is None:
        metadata = {
            'transformers': {
                'tumor size': MinMaxScaler()
            },
        }
    else:
        # check user's metadata is valid
        pass

    df = read_csv_to_df(file_path)
    patient_data_custom = TabularPatientBase(df, metadata)
    return patient_data_custom


def get_trial_patient_records(file_path=None, metadata=None):
    if file_path is None:
        data = load_trial_patient_tabular(os.path.join(os.path.dirname(__file__), "demo_data/trial_patient_data"))
        df = data['data']
    else:
        # check user's file is valid
        df = read_csv_to_df(file_path)
    if metadata is None:
        metadata = {
            'sdtypes':{
                'target_label': 'boolean',
            },
            'transformers':{
                'target_label': None,
            }
        }
    else:
        # check user's metadata is valid
        pass

    transformed_data = TabularPatient(df, metadata)
    return transformed_data