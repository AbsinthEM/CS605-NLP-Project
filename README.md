# CS605-NLP-PROJECT

## Project Structure

```
CS605-NLP-PROJECT/                          
├── data/                                # Data storage directory
│   ├── gold/                              # Feature extraction and engineering data
│   ├── processed/                         # Cleaned and processed data ready for analysis
│   │   ├── USS_EDA_Summary.json             # Exploratory data analysis summary results
│   │   ├── USS_Reviews_Silver.csv           # Clean structured review data in CSV format
│   │   └── USS_Reviews_Silver.parquet       # Clean structured review data in Parquet format
│   └── raw/                               # Original unprocessed data
│       ├── USS_Reviews_Raw_10k.csv          # Raw review dataset with 10k records
│       └── USS_Reviews_Raw_100k.csv         # Raw review dataset with 100k records
├── notebooks/                           # Jupyter notebooks for analysis and exploration
│   ├── data_cleaning.ipynb                # Data preprocessing and cleaning notebook
│   └── uss_eda_notebook.ipynb             # Exploratory data analysis notebook
├── output/                              # Generated outputs and results
│   ├── figures/                           # Generated plots and visualizations
│   │   ├── dashboard/                       # Dashboard-related charts and graphs
│   │   └── eda/                             # Exploratory data analysis visualizations
│   ├── models/                            # Trained machine learning models
│   └── results/                           # Analysis results
│       └── json/                            # Analysis results in JSON format
└── src/                                 # Source code modules
    ├── data/                              # Data processing modules                
    │   ├── cleaner.py                       # Data cleaning functions and utilities
    │   ├── loader.py                        # Data loading and import utilities
    │   └── transformer.py                   # Basic data transformation
    ├── features/                          # Feature extraction and engineering modules
    └── models/                            # Machine learning model implementations
```

