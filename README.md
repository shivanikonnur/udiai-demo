# UIDAI Aadhaar Analytics
<b>📌 PROJECT OVERVIEW</b>
UIDAI Aadhaar Data Analytics & Machine Learning project that analyzes enrolment and update data to identify temporal, geographic, and demographic trends. The project focuses on trend detection, anomaly identification, and predictive insights to support data-driven governance.<br>
Author-Shivani Konnur<br>
<b>🎯 OBJECTIVES</b>
- Analyze Aadhaar enrolment vs update trends
- Identify geographic and demographic patterns
- Detect anomalies in update behavior
- Generate predictive insights using machine learning
<br>
<b>🛠️ TECH STACK</b>
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
<br>
<b>📂 PROJECT STRUCTURE</b>
AADHAR ANALYSIS/<br>
├── DATA/# Input datasets     <br>   
├── FIGURES/# Generated visualizations  <br>   
├── SRC/
│   └── analysis.py<br>
├── README.md<br>
└── .gitignore<br>
<br>
<b>📊 HOW TO RUN IN THE TERMINAL</b>
bash<br>
cd AADHAR\ ANALYSIS<br>
python SRC/analysis.py<br>
<b>📄 DATASET INFORMATION</b>
The dataset used in this project was officially provided by the Government of India for a hackathon and is not publicly distributable.<br>

Due to data usage and licensing constraints, the raw dataset is not included in this repository.<br>

However, the complete data processing, analysis, visualization, and machine learning pipeline is fully implemented and reproducible with authorized access to the dataset.<br>
<b>🗂️ DATASET SCHEMA</b>

| Column Name | Description |<br>
| State | Indian state name |<br>
| Year | Year of record |<br>
| Month | Month of activity |<br>
| Enrolments | New Aadhaar enrolments|<br>
| Updates | Total Aadhaar updates |<br>
| Demographic_Updates | Demographic changes<br>(Name, DOB, Address, etc.) |<br>
| Biometric_Updates | Biometric updates<br>(Fingerprint, Iris, Photo) |<br>
| Male | Male beneficiaries |<br>
| Female | Female beneficiaries |<br>
| Age_0_18 | Age group 0–18 |<br>
| Age_18_60 | Age group 18–60 |
| Age_60_plus | Age group 60+ |


