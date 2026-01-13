# UIDAI Demographic Update Analysis

live- **https://irunonironic-uidai-data-hackathon-app-mplmt9.streamlit.app/**

This project analyses Aadhaar demographic update and enrolment data to detect abnormal activity, spatial hotspots, and potential internal mobility patterns across India.

The pipeline performs:
- State-level aggregation and anomaly detection  
- District and PIN-code hotspot analysis  
- Correlation between updates and enrolments  
- Heatmap visualisation generation  

---

## Requirements

- Python 3.9+
- pandas  
- numpy  
- matplotlib  

Install dependencies:

```bash
pip3 install pandas numpy matplotlib
````

---

## How to Run

1. Place input CSV files inside:

```
data/demographic/
data/enrollment/
```

2. Run the analysis:

```bash
python3 main.py
```

---

## Outputs

Generated automatically in:

```
outputs/
```

Includes:

* Anomaly detection tables
* District and PIN hotspot CSVs
* Correlation reports
* Heatmap images (PNG)

---

## Notes

* Input data must follow the UIDAI dataset schema.
* Dates should be in DD-MM-YYYY format.
* Analysis is performed on anonymised aggregated data.




