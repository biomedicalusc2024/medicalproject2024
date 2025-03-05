from download_trial.client import ClinicalTrials
import pandas as pd

ct = ClinicalTrials()

# Get 50 full studies related to Coronavirus and COVID in csv format.
ct.get_full_studies(search_expr="Coronavirus+COVID", max_studies=50)

# Get the NCTId, Condition, and Brief title fields from 1000 studies related to Coronavirus and Covid.
corona_fields = ct.get_study_fields(
    search_expr="Coronavirus+COVID",
    fields=["NCT Number", "Conditions", "Study Title"],
    max_studies=1000,
    fmt="csv",
)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame.from_records(corona_fields[1:], columns=corona_fields[0])

# Save the DataFrame to a CSV file
df.to_csv("covid_clinical_trials.csv", index=False)

print("CSV file saved as covid_clinical_trials.csv")
