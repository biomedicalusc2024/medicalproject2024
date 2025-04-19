import os, rdt
from .tabular_utils import read_csv_to_df
from .patient_data import TabularPatientBase
from .trial_simulation.data import TabularPatient
from .trial_patient_match.data import TrialData, PatientData
from .demo_data import load_trial_patient_tabular, load_mimic_ehr_sequence, load_trial_outcome_data

def patient_prediction():
    file_path = os.path.join(os.path.dirname(__file__), "demo_data/patient/tabular/patient_tabular.csv")
    metadata = {
        'sdtypes': {
            'gender': 'boolean',
            'mortality': 'boolean',
            'ethnicity': 'categorical',
        },
        'transformers': {
            'gender': rdt.transformers.FrequencyEncoder(),
            'mortality': rdt.transformers.FrequencyEncoder(),
            'ethnicity': rdt.transformers.FrequencyEncoder(),
        },
    }

    # Note that read_csv_to-df will automatically convert column names to lowercase
    df = read_csv_to_df(file_path)

    # Custom Metadata Conversion
    patient_data_custom = TabularPatientBase(df, metadata)
    print(patient_data_custom.df.head())

    # Auto Metadata Conversion
    patient_data_auto = TabularPatientBase(df)
    print(patient_data_auto.df.head())

    # Restore the Original Metadata
    df_reversed = patient_data_custom.reverse_transform()
    print(df_reversed.head())

    print("This is only a demo for patient prediction, for custom use, please refer to the code")
    return patient_data_custom


def trial_patient_simulation():
    data = load_trial_patient_tabular(os.path.join(os.path.dirname(__file__), "demo_data/trial_patient_data"))
    df = data['data']
    metadata = {
        'sdtypes':{
            'target_label': 'boolean',
        },
        'transformers':{
            'target_label': None,
        }
    }
    transformed_data = TabularPatient(df, metadata)
    print(transformed_data.df.head())

    print("This is only a demo for trial patient simulation, for custom use, please refer to the code")
    return transformed_data


def trial_patient_matching():
    # load demo ehr sequence and patient data
    data = load_trial_outcome_data(os.path.join(os.path.dirname(__file__), "demo_data/trial_outcome_data"))
    ehr = load_mimic_ehr_sequence(os.path.join(os.path.dirname(__file__), "demo_data/patient/sequence"), 
                                  n_sample=1000)

    # make subsampling
    df = data['data'].iloc[:10]
    trial_data = TrialData(df,encode_ec=True)

    # we first simulate the eligibility criteria matching labels for each patient
    ec_label_list = []
    for i in range(len(ehr['feature'])):
        # randomly choose a trial that this patient satisfies
        trial = df.sample(1)
        ec_label_list.append([trial['inclusion_criteria_index'].values[0], trial['exclusion_criteria_index'].values[0]])

    # build patient seq data
    # ec_label_list, first is matched inclusion criteria, second is matched exclusion criteria
    ehr_data = PatientData(
        data={'v':ehr['visit'], 'y': ec_label_list,  'x':ehr['feature'],},
        metadata={
            'visit': {'mode':'dense', 'order':ehr['order']},
            'label': {'mode':'dense'},
            'voc': ehr['voc'],
            },
    )
    print("visit: \n", ehr_data.visit[:5])
    print("feature: \n", ehr_data.feature[:5])
    print("label: \n", ehr_data.label[:5])

    print("This is only a demo for trial patient matching, for custom use, please refer to the code")
    return trial_data, ehr_data

    