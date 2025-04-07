import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

# NHANES year and data sources
NHANES_YEAR = "08/2021-08/2023"

NHANES_INDEX = {
    "Demographics": [
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt" #Demographic Variables and Sample Weights
    ],
    "Dietary": [
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1IFF_L.xpt", #Dietary Interview - Individual Foods, First Day
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR2IFF_L.xpt", #Dietary Interview - Individual Foods, Second Day
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1TOT_L.xpt", #Dietary Interview - Total Nutrient Intakes, First Day
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR2TOT_L.xpt", #Dietary Interview - Total Nutrient Intakes, Second Day
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DRXFCD_L.xpt", #Dietary Interview Technical Support File - Food Codes
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DSBI.xpt", #Dietary Supplement Database - Blend Information
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DSII.xpt", #Dietary Supplement Database - Ingredient Information
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/DSPI.xpt", #Dietary Supplement Database - Product Information
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DR1IFF_L.xpt", #Dietary Supplement Use 30-Day - Individual Dietary Supplements
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DSQIDS_L.xpt", #Dietary Supplement Use 30-Day - Individual Dietary Supplements
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DSQTOT_L.xpt", #Dietary Supplement Use 30-Day - Total Dietary Supplements
    ],
    "Examination": [
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BAX_L.xpt", #Balance
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPXO_L.xpt", #Blood Pressure - Oscillometric Measurements
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt", #Body Measures
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/LUX_L.xptt", #Liver Ultrasound Transient Elastography
    ],
    "Lab": [
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/AGP_L.xpt", #alpha-1-Acid Glycoprotein
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HDL_L.xpt", #Cholesterol – High-Density Lipoprotein
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.xpt", #Cholesterol - Total
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt", #Complete Blood Count with 5-Part Differential in Whole Blood
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FASTQX_L.xpt", #Fasting Questionnaire
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FERTIN_L.xpt", #Ferritin
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FOLATE_L.xpt", #Folate - RBC
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt", #Glycohemoglobin
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HEPA_L.xpt", #Hepatitis A
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HEPB_S_L.xpt", #Hepatitis B Surface Antibody
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HSCRP_L.xpt", #High-Sensitivity C-Reactive Protein
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/INS_L.xpt", #Insulin
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PBCD_L.xpt", #Lead, Cadmium, Total Mercury, Selenium, & Manganese – Blood
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/IHGEM_L.xpt", #Mercury: Inorganic, Ethyl, and Methyl - Blood
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.xpt", #Plasma Fasting Glucose
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FOLFMS_L.xpt", #Serum Folate Forms - Total & Individual - Serum
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TST_L.xpt", #Sex Steroid Hormone Panel - Serum
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TFR_L.xpt", #Transferrin Receptor
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/UCPREG_L.xpt", #Urine Pregnancy Test
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/VID_L.xpt", #Vitamin D
    ],
    "Questionnaire": [
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/AUQ_L.xpt", #Audiometry
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/ACQ_L.xpt", #Acculturation
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/ALQ_L.xpt", #Alcohol Use
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BAQ_L.xpt", #Balance
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.xpt", #Blood Pressure & Cholesterol
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HSQ_L.xpt", #Current Health Status
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEQ_L.xpt", #Dermatology
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.xpt", #Diabetes
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DBQ_L.xpt", #Diet Behavior & Nutrition
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/ECQ_L.xpt", #Early Childhood
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/FNQ_L.xpt", #Functioning
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HIQ_L.xpt", #Health Insurance
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HEQ_L.xpt", #Hepatitis
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HUQ_L.xpt", #Hospital Utilization & Access to Care
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HOQ_L.xpt", #Housing Characteristics
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/IMQ_L.xpt", #Immunization
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/INQ_L.xpt", #Income
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/KIQ_U_L.xpt", #Kidney Conditions - Urology
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.xpt", #Medical Conditions
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DPQ_L.xpt", #Mental Health - Depression Screener
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/OCQ_L.xpt", #Occupation
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/OHQ_L.xpt", #Oral Health
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PUQMEC_L.xpt", #Pesticide Use
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQ_L.xpt", #Physical Activity
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PAQY_L.xpt", #Physical Activity - Youth
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/RXQ_RX_L.xpt", #Prescription Medications
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/RXQASA_L.xpt", #Preventive Aspirin Use
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/RHQ_L.xpt", #Reproductive Health
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SLQ_L.xpt", #Sleep Disorders
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQ_L.xpt", #Smoking - Cigarette Use
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQFAM_L.xpt", #Smoking - Household Smokers
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SMQRTU_L.xpt", #Smoking - Recent Tobacco Use
        "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/WHQ_L.xpt", #Weight History
    ],
}

def getNHANES(path):
    all_data = {}
    year_path = os.path.join(path, "NHANES", NHANES_YEAR.replace("-", "_"))

    for category, urls in NHANES_INDEX.items():
        category_data = []

        for dataset_url in urls:
            datasetPath = os.path.join(year_path, category)
            os.makedirs(datasetPath, exist_ok=True)

            file_name = dataset_url.split("/")[-1]
            file_path = os.path.join(datasetPath, file_name)

            if not os.path.exists(file_path):
                print_sys(f"Downloading NHANES file: {file_name}")
                download_file(dataset_url, file_path, datasetPath)
            else:
                print_sys(f"Found local file: {file_name}")

            train, test = loadLocalFile(file_path)
            if train:
                category_data.extend(train)

        all_data[category] = category_data

    return all_data

def loadLocalFile(file_path):
    try:
        if file_path.lower().endswith(".xpt"):
            df = pd.read_sas(file_path, format="xport", encoding="utf-8")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format")

        df['__source_file__'] = os.path.basename(file_path)

        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        return train_df.to_dict(orient="records"), test_df.to_dict(orient="records")
    except Exception as e:
        print_sys(f"Failed to load file: {e}")
        return None, None
