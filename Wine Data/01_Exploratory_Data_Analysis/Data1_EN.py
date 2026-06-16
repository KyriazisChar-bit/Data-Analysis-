import numpy as np
import pandas as pd

AEM = 7137

arxeio = pd.read_csv("Data_Analysis_2026.csv")
# Create 2 files to store the data

# sample_arxeio (sarxeio) is the data to work on
sarxeio = arxeio.sample(frac=0.8, random_state=AEM)

# test_arxeio (tarxeio) is the test data to check the code
tarxeio = arxeio.drop(sarxeio.index)
# First, let's get some information about the DataSet

sarxeio.info()
sarxeio.describe()

print(sarxeio.head())

# Find the duplicate rows and remove them
double = sarxeio.duplicated().sum()

print("Rows before cleaning:", sarxeio.shape[0])
print("Doublicates are:", double)

sarxeio = sarxeio.drop_duplicates()
double = sarxeio.duplicated().sum()

print("Rows after cleaning:", sarxeio.shape[0])
print('Doublicates are:', double)

# Thorough check for each column separately
print("--- UNIQUE VALUES ANALYSIS PER COLUMN ---")

for col in sarxeio.columns:
    print(f"\n{'='*40}")
    print(f"COLUMN: {col.upper()}")
    print(f"{'='*40}")

    # Get the unique values and sort them for easier reading
    # We use dropna() to temporarily view the data without the empty cells
    unique_vals = sarxeio[col].unique()

    # Print all unique values
    print(f"Found {len(unique_vals)} unique values:")
    print(unique_vals)

    # Check whether the column is recognized as numeric or as text (object)
    print(f"\nColumn data type: {sarxeio[col].dtype}")

    # If the type is object, there are certainly strings (e.g. '?', 'N/A')
    # that prevent statistical analyses.
    if sarxeio[col].dtype == 'object':
        print(f" Column '{col}' contains text. Possible 'garbage' detected.")

print("\n--- END OF CHECK ---")

# We will separate the white from the red wines to get a better
# picture of the mean values of the characteristics
from difflib import get_close_matches

valid_types = ["red", "white"]
# Convert to string and basic cleaning
sarxeio["wine_type"] = (
    sarxeio["wine_type"]
    .astype(str)
    .str.lower()
    .str.strip()
)
# Convert the "nan" string back to NaN
sarxeio.loc[sarxeio["wine_type"] == "nan", "wine_type"] = pd.NA

# Get unique values without NaN
unique_values = sarxeio["wine_type"].dropna().unique()
mapping = {}

for val in unique_values:
    match = get_close_matches(val, valid_types, n=1, cutoff=0.6)
    if match:
        mapping[val] = match[0]

# Replace values
sarxeio["wine_type"] = sarxeio["wine_type"].replace(mapping)
valid_values = sarxeio.loc[sarxeio["wine_type"].isin(valid_types), "wine_type"]

# Check the result

print("Clean values:")
print(valid_values.value_counts(dropna=False))

# Check for missing data after renaming the wines
invalid_values = sarxeio.loc[~sarxeio["wine_type"].isin(valid_types), "wine_type"]

print("Invalid values found:")
print(invalid_values.value_counts(dropna=False))

print("\n--- VALIDITY CHECK pH & quality ---")

# ===== pH =====
# 1. FIRST convert to numbers. Anything that is not a number becomes NaN
sarxeio["pH"] = pd.to_numeric(sarxeio["pH"], errors="coerce")

# 2. NOW that the column is numeric, the comparison will work
invalid_quality_mask = (sarxeio["pH"] < 0) | (sarxeio["pH"] > 10)

print(f"Invalid pH values found: {invalid_quality_mask.sum()}")

# 3. Convert the extreme values to NaN
sarxeio.loc[invalid_quality_mask, "pH"] = np.nan

# ===== ALCOHOL ====
# 1. FIRST convert to numbers. Anything that is not a number becomes NaN
sarxeio["alcohol"] = pd.to_numeric(sarxeio["alcohol"], errors="coerce")

# 2. NOW that the column is numeric, the comparison will work
invalid_quality_mask = (sarxeio["alcohol"] < 0) | (sarxeio["alcohol"] > 100)

print(f"Invalid quality values found: {invalid_quality_mask.sum()}")

# 3. Convert the extreme values to NaN
sarxeio.loc[invalid_quality_mask, "alcohol"] = np.nan

# ===== QUALITY =====
# 1. FIRST convert to numbers. Anything that is not a number becomes NaN
sarxeio["quality"] = pd.to_numeric(sarxeio["quality"], errors="coerce")

# 2. NOW that the column is numeric, the comparison will work
invalid_quality_mask = (sarxeio["quality"] < 0) | (sarxeio["quality"] > 10)

print(f"Invalid quality values found: {invalid_quality_mask.sum()}")

# 3. Convert the extreme values to NaN
sarxeio.loc[invalid_quality_mask, "quality"] = np.nan


# Convert to NaN
sarxeio.loc[(sarxeio["quality"] < 0) | (sarxeio["quality"] > 10), "quality"] = np.nan


# Separate the white from the red wines
red_wines = sarxeio[sarxeio["wine_type"] == "red"]
white_wines = sarxeio[sarxeio["wine_type"] == "white"]

print("Red wines:", red_wines.shape[0])
print("White wines:", white_wines.shape[0])

# List of the "suspicious" values we detected in the csv and in the Unique values
garbage_values = ['?', '-999', '999', '9999', '9999 N/A', 'N/A', '-999?', 'nan']


print("--- START OF CLEANING PER COLUMN ---")

# Take all columns except 'wine_type' (which we already cleaned)
cols_to_clean = sarxeio.columns
rejected_vals = []


for col in cols_to_clean:

    if col == "wine_type":

        invalid_count = len(invalid_values)
        rejected_vals.append(int(invalid_count))

        print(f"Column '{col}': Found {invalid_count} invalid values.")

        continue

    # Convert the garbage values to NaN ONLY for this specific column
    sarxeio[col] = sarxeio[col].replace(garbage_values, np.nan)

    # Convert the column to numeric (float)
    # errors='coerce' converts anything else non-numeric to NaN
    sarxeio[col] = pd.to_numeric(sarxeio[col], errors='coerce')

    # Count the empty cells (NaN) in the column
    missing_count = sarxeio[col].isna().sum()
    rejected_vals.append(int(missing_count))

    if missing_count > 0:
        print(f"Column '{col}': Found {missing_count} missing values.")

    else:
        print(f"Column '{col}': Clean (0 missing values).")

invalid_pH = []


# Final check whether any NaN remained

print("\n--- CLEANING COMPLETED ---")

print(f"Final file shape: {sarxeio.shape}")
print(f'The values changed in each column are: {rejected_vals}')

print("\n--- MEAN VALUES PER WINE TYPE ---")

# Cleaning loop
for col in sarxeio.columns:
    if col == "wine_type":
        continue

    # Now the commands are INSIDE the loop
    sarxeio[col] = sarxeio[col].replace(garbage_values, np.nan)
    sarxeio[col] = pd.to_numeric(sarxeio[col], errors='coerce')

# Remove Unnamed columns if they still exist
sarxeio = sarxeio.loc[:, ~sarxeio.columns.str.contains('^Unnamed')]
# First drop the Unnamed columns
sarxeio = sarxeio.loc[:, ~sarxeio.columns.str.contains('^Unnamed')]

# Then count how many ACTUAL columns remained (e.g. 12 or 13)
columns_count = len(sarxeio.columns)

# Define the threshold: must have at least (Total - 3) filled cells
# If you have 12 columns, the limit will be 9.
limit = columns_count - 3

# Apply dropna with the thresh
sarxeio = sarxeio.dropna(thresh=limit)

print(f"Total columns examined: {columns_count}")
print(f"Minimum values per row: {limit}")
print(f"Remaining rows: {len(sarxeio)}")


print(f"Remaining rows after cleaning: {len(sarxeio)}")

if not sarxeio.empty:
    print("\n--- MEAN AND MEDIAN VALUES PER WINE TYPE ---")
    # Compute Mean and Median together
    stats = sarxeio.groupby('wine_type').agg(['mean', 'median']).T
    print(stats)

# Identify the numeric columns (except wine_type)
numeric_cols = sarxeio.select_dtypes(include=[np.number]).columns


# pH cannot be above 14 and below 0, so we treat them as bad entries
# They will be replaced with the mean pH value of the corresponding wine group


# Fill the NaN with the mean value depending on wine_type
# transform computes the mean for each group and places it exactly at the NaN positions
sarxeio[numeric_cols] = sarxeio.groupby('wine_type')[numeric_cols].transform(lambda x: x.fillna(x.mean()))

# Check whether any NaN remained (e.g. if a column was all NaN for one type)
print("\n--- CHECK FOR EMPTY CELLS AFTER FILLING ---")
print(sarxeio[numeric_cols].isna().sum())

# Final print of the statistics with the transposed table you requested
if not sarxeio.empty:
    print("\n--- FINAL MEAN AND MEDIAN VALUES ---")
    final_stats = sarxeio.groupby('wine_type').agg(['mean', 'median']).T
    print(final_stats.round(3))


red_wines = sarxeio[sarxeio["wine_type"] == "red"]
white_wines = sarxeio[sarxeio["wine_type"] == "white"]

# QUESTION 2
import matplotlib.pyplot as plt

numeric_cols = sarxeio.select_dtypes(include=[np.number]).columns
numeric_cols_red = red_wines.select_dtypes(include=[np.number]).columns
numeric_cols_white = white_wines.select_dtypes(include=[np.number]).columns

print("\n--- CREATING BOXPLOTS ---")

# pH cannot be above 14 and below 0, so we treat them as bad entries
# They will be replaced with the mean pH value of the corresponding wine group


# Question 3
# To find the outliers I will work separately for the white and the red wines
# since the wine type affects the values of the variables

print("\n--- CHECK FOR oUTLIERS ---")
# -----------------------------
# Boxplots for ALL wines
# -----------------------------


# RED wines
print("\n--- RED WINES ---")

# Check for Outliers based on which belong in the 25-75% range of the values
outliers_red = []
for col in numeric_cols_red:

    Q1 = red_wines[col].quantile(0.25)
    Q3 = red_wines[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = red_wines[(red_wines[col] < lower) | (red_wines[col] > upper)]

    print(f"{col}: {len(outliers)} outliers (IQR)")
    outliers_red.append(len(outliers))

print("\n--- OUTLIERS RED ---")
print(f'Outliers: {outliers_red}')

outliers_red = []

# Counts how many Outliers are more than 3.5 standard deviations from the mean for the RED WINES
print("\n--- Z-score RED ---")
for col in numeric_cols_red:

    mean = red_wines[col].mean()
    std = red_wines[col].std()

    z_scores = (red_wines[col] - mean) / std

    outliers = red_wines[abs(z_scores) > 3.5]

    print(f"{col}: {len(outliers)} outliers (Z-score)")
    outliers_red.append(len(outliers))

print(f'Outliers Z-score: {outliers_red}')

# Clamp transformation

for col in numeric_cols_red:

    Q1 = red_wines[col].quantile(0.25)
    Q3 = red_wines[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    red_wines[col] = red_wines[col].clip(lower, upper)



# --- WHITE wines ---
print("\n--- WHITE WINES ---")


# Check for Outliers based on which belong in the 25-75% range of the values
outliers_white = []
for col in numeric_cols_white:

    Q1 = white_wines[col].quantile(0.25)
    Q3 = white_wines[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = white_wines[(white_wines[col] < lower) | (white_wines[col] > upper)]

    print(f"{col}: {len(outliers)} outliers (IQR)")
    outliers_white.append(len(outliers))

print("\n--- OUTLIERS WHITE ---")
print(f'Outliers: {outliers_white}')

outliers_white = []

# Counts how many Outliers are more than 3.5 standard deviations from the mean for the WHITE WINES
print("\n--- Z-score WHITE ---")
for col in numeric_cols_white:

    mean = white_wines[col].mean()
    std = white_wines[col].std()

    z_scores = (white_wines[col] - mean) / std

    outliers = white_wines[abs(z_scores) > 3.5]

    print(f"{col}: {len(outliers)} outliers (Z-score)")
    outliers_white.append(len(outliers))

print("\n--- Z-score WHITE as a vector ---")
print(f'Outliers Z-score: {outliers_white}')

# Clamp transformation
for col in numeric_cols_white:

    Q1 = white_wines[col].quantile(0.25)
    Q3 = white_wines[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    white_wines[col] = white_wines[col].clip(lower, upper)


# =================================
# Question 4
# ===================================


numeric_cols = sarxeio.select_dtypes(include=[np.number]).columns

continuous_stats = pd.DataFrame({

    "count": sarxeio[numeric_cols].count(),
    "mean": sarxeio[numeric_cols].mean(),
    "median": sarxeio[numeric_cols].median(),
    "min": sarxeio[numeric_cols].min(),
    "max": sarxeio[numeric_cols].max(),
    "std": sarxeio[numeric_cols].std(),
    "missing_values": sarxeio[numeric_cols].isna().sum(),
    "cardinality": sarxeio[numeric_cols].nunique()

})

print("\n--- CONTINUOUS VARIABLES ---")
print(continuous_stats)

categorical_cols = sarxeio.select_dtypes(include=['object']).columns

categorical_stats = pd.DataFrame({

    "count": sarxeio[categorical_cols].count(),
    "most_frequent": sarxeio[categorical_cols].mode().iloc[0],
    "min": sarxeio[categorical_cols].min(),
    "max": sarxeio[categorical_cols].max(),
    "missing_values": sarxeio[categorical_cols].isna().sum(),
    "cardinality": sarxeio[categorical_cols].nunique()

})

print("\n--- CATEGORICAL VARIABLES ---")
print(categorical_stats)

numeric_cols = sarxeio.select_dtypes(include=[np.number]).columns
