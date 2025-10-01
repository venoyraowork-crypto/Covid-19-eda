📊 COVID-19 Clinical Trials EDA (Python Project)

This project performs Exploratory Data Analysis (EDA) on the COVID-19 Clinical Trials dataset from ClinicalTrials.gov.
It uses Python, Pandas, Matplotlib, and Seaborn to clean the data, analyze trial statuses, phases, demographics, and visualize trends over time.

🚀 Project Features

Load and explore the raw dataset (CSV from ClinicalTrials.gov).
Handle missing values and clean the dataset.
Extract country information from trial locations.
Perform univariate analysis (status, phases, gender, age, country).
Perform bivariate analysis (phases vs. status, top conditions and outcomes).
Time-series analysis of trials started over time.
Generate plots and CSV summaries in the output/ folder.
Save the cleaned dataset for future analysis.

▶️ Running the Project

Run the script from terminal:
python eda_covid_trials.py covid_clinical_trials.csv

📊 Example Outputs

The script generates:
Cleaned CSV → cleaned_covid_clinical_trials.csv
Missing data report → missing_percentages.csv
Univariate plots → Status distribution, Phases distribution, Gender distribution, Top countries, Enrollment histogram
Bivariate plots & summaries → Phases vs. Status, Monthly trials time series, Top conditions & outcomes
