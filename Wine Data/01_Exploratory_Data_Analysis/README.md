# Project 1: Data Cleaning & Exploratory Data Analysis (EDA)

##  Objective
The goal of this project is to take a raw, messy dataset containing chemical properties of red and white wines and transform it into a clean, statistically sound format ready for machine learning applications. The script systematically handles invalid entries, imputes missing data based on categorical grouping, and manages outliers using standard statistical thresholds[cite: 4].

##  Dataset Overview
* **Source:** `Data_Analysis_2026.csv`[cite: 4].
* **Sampling:** 80% of the raw data was randomly sampled using a custom seed for reproducibility[cite: 4].
* **Features:** Includes continuous chemical properties (e.g., pH, alcohol, residual sugar) and categorical variables (wine type)[cite: 4].

##  Methodology & Technical Approach

### 1. Data Validation & Cleaning
Real-world data is rarely perfect. The initial phase focused on identifying and neutralizing "garbage" data:
* **Deduplication:** Identified and removed duplicate rows to prevent skewed analysis[cite: 4].
* **String Standardization:** Applied fuzzy string matching (`get_close_matches`) to standardize inconsistent text entries in the `wine_type` column (e.g., correcting misspellings to strictly "red" or "white")[cite: 4].
* **Logical Bounds Checking:** Filtered out physically impossible or invalid numeric entries, such as negative pH values, pH > 10, alcohol percentages > 100, and quality scores outside the 0-10 range[cite: 4]. 
* **Garbage Value Removal:** Scanned for and nullified placeholder strings used for missing data (e.g., `?`, `-999`, `9999 N/A`)[cite: 4].

### 2. Missing Value Imputation
* Dropped highly corrupted rows missing data in more than 3 columns[cite: 4].
* Instead of using a global mean to fill remaining missing numerical values, the script intelligently imputes `NaN` values using the specific mean of the corresponding `wine_type` group (red or white), preserving the distinct chemical profiles of each wine variant[cite: 4].

### 3. Outlier Detection & Transformation
Because wine chemistry varies significantly by type, outlier detection was performed *separately* for red and white wines:
* **Detection:** Evaluated outliers using both the Interquartile Range (IQR) method and the Z-score method (flagging data points > 3.5 standard deviations from the mean)[cite: 4].
* **Treatment (Clamp Transformation):** Applied a clipping function to cap extreme outliers at the lower (`Q1 - 1.5 * IQR`) and upper (`Q3 + 1.5 * IQR`) bounds, retaining the data points while minimizing their distortion on future models[cite: 4].

### 4. Statistical Profiling
Generated comprehensive summary statistics for the finalized dataset:
* **Continuous Variables:** Count, mean, median, min, max, standard deviation, missing values, and cardinality[cite: 4].
* **Categorical Variables:** Count, most frequent value (mode), min, max, missing values, and cardinality[cite: 4].

##  How to Run

1. Ensure you have the `Data_Analysis_2026.csv` file in the same directory as the script.
2. Install required dependencies:
```bash
   pip install pandas numpy matplotlib
