
import pandas as pd
import os
import warnings

# Suppress warnings to avoid cluttering output, especially from pandas or SPSS reader
warnings.filterwarnings("ignore")

def check_datasets_availability() -> bool:
    """
    Load PISA 2022 datasets from local files.

    Users must manually download the files from Zenodo:
    https://zenodo.org/records/13382904

    Expected files in "data" directory:
    - CY08MSP_STU_QQQ.sav (1.97 GB)
    - CY08MSP_SCH_QQQ.sav (18.53 MB)
    """
    student_file = os.path.join("data", "CY08MSP_STU_QQQ.sav")
    school_file = os.path.join("data", "CY08MSP_SCH_QQQ.sav")

    # Check if files exist
    missing = []
    if not os.path.exists(student_file):
        missing.append("CY08MSP_STU_QQQ.sav (1.97 GB)")
    if not os.path.exists(school_file):
        missing.append("CY08MSP_SCH_QQQ.sav (18.53 MB)")
    
    if missing:
        print("\n" + "="*60)
        print("XXX ERROR: Data files not found! XXX")
        print("="*60)
        print("\nPlease download these files from Zenodo:")
        print("   https://zenodo.org/records/13382904")
        print("\nMissing files:")
        for f in missing:
            print(f"   • {f}")
        print(f"\nPlace them in: {os.path.abspath('data')}/")
        print("\nAnd relaunch the program afterwards.")
        print("="*60 + "\n")
        return False
    
    # Load datasets
    print("All the requested datasets are in the required directory. ✔")
    return True

def load_pisa_datasets() -> tuple:
    """
    Load PISA 2022 datasets from local files.
    It first checks for the availability of the .sav files.
    If .csv versions of the datasets exist, they are loaded for faster processing.
    Otherwise, the .sav files are loaded using pandas.read_spss and then saved
    as .csv files for future faster loading.

    Returns:
    Automatically converts to CSV for faster future loading.
    """
    if not check_datasets_availability():
        return None, None
    
    student_sav = os.path.join("data", "CY08MSP_STU_QQQ.sav")
    school_sav = os.path.join("data", "CY08MSP_SCH_QQQ.sav")
    student_csv = os.path.join("data", "pisa_2022_student.csv")
    school_csv = os.path.join("data", "pisa_2022_school.csv")
    
    # Check if CSV files exist
    if os.path.exists(student_csv) and os.path.exists(school_csv):
        # If CSV files are found, load them
        print("Loading from cache (faster)...")
        df_student = pd.read_csv(student_csv)
        df_school = pd.read_csv(school_csv)
    
    else:
        
        print("Loading from .sav datasets and converting to CSV...")
        print("This may take a few minutes for the student dataset (1.97 GB)...")
        df_student = pd.read_spss(student_sav)
        df_school = pd.read_spss(school_sav)
        print("Saving as CSV for faster future loading...")
        df_student.to_csv(student_csv, index=False)
        df_school.to_csv(school_csv, index=False)
        print("CSV datasets saved!")
    
    print(f"Student dataset: {len(df_student):,} rows, {len(df_student.columns)} columns")
    print(f"School dataset: {len(df_school):,} rows, {len(df_school.columns)} columns")
    
    return df_student, df_school

def filter_dataset_by_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Filters a DataFrame to include only records from a specified country.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - country (str): The country code to filter by (e.g., "Switzerland").

    Returns:
    - pd.DataFrame: A new DataFrame containing only the records for the specified country.
    """
    df_filtered_by_country = df.query("CNT == @country") #SOURCE: PYTHON DATASCIENCE HANDBOOK (p. 213)
    return df_filtered_by_country

def merge_datasets_by_school(df_student: pd.DataFrame, df_school: pd.DataFrame) -> pd.DataFrame:
    """
    Merge student and school datasets (csv format) by school. 
    It uses the common key 'CNTSCHID' (school ID).
    """
    # left merged is required to conserve all student records that have a school ID
    # that does not appear in the school dataset.
    df_merged = pd.merge(df_student, df_school, on="CNTSCHID", how="left")
    return df_merged

def build_swiss_merged_dataset():
    """
    Builds the merged dataset for Switzerland by loading PISA student and school data,
    filtering for Switzerland, and then merging them based on school ID.
    The resulting merged dataset is saved as 'swiss_merged_dataset.csv' in the 'data' directory.

    Returns: None
    """
    df_student, df_school = load_pisa_datasets()
    
    print(f"Filtering student dataset for Switzerland...")
    df_student_swiss = filter_dataset_by_country(df_student, "Switzerland")
    
    print(f"Filtering school dataset for Switzerland...")
    df_school_swiss = filter_dataset_by_country(df_school, "Switzerland")
    
    print(f"Merging student and school datasets for Switzerland...")
    df_merged_swiss = merge_datasets_by_school(df_student_swiss, df_school_swiss)

    #Save the swiss merged dataset in csv format
    print(f"Saving swiss merged dataset in the data directory...")
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", "swiss_merged_dataset.csv")
    df_merged_swiss.to_csv(data_path, index=False)
    print(f"Swiss merged dataset saved to: {data_path} ✔")
   
def reduced_swiss_dataset()-> pd.DataFrame:
    """
    Loads the full Swiss merged dataset and reduces it to a predefined set of important features.
    If the 'swiss_merged_dataset.csv' file does not exist, it calls `build_swiss_merged_dataset`
    to create it first. The reduced dataset is then saved as 'swiss_reduced_dataset.csv'
    in the 'src' directory.

    Returns:
    - pd.DataFrame: The reduced DataFrame containing only the important columns.
    """
    file_path = os.path.join("data", "swiss_merged_dataset.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        reduced_swiss_dataset()
    
    # Load the merged dataset
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    
    # Important and useful selected features (see Project report)
    important_columns = [
        'PV1MATH',         # Mathematics performance score (first plausible value. Target)
        'CNTSCHID',        # School ID (required to build MEAN_ESCS)
        'CNTSTUID',        # Student ID (kept for monitoring)
        'PAREDINT',        # Parents' highest level of education
        'HOMEPOS',         # Home possession index
        'HISEI',           # Highest parental occupational status
        'GRADE',           # Relative grade index
        'ST004D01T',       # Student gender
        'IMMIG',           # Immigrant status
        'MATHEFF',         # Mathematics self-efficacy
        'ANXMAT',          # Mathematics anxiety
        'REPEAT',          # Grade repetition
        'ST062Q01TA',      # Student abstenteeism
        'DISCLIM',         # Disciplinary climate in mathematics
        'STUBEHA',         # Student-related factors affecting school climate
        'STAFFSHORT',      # Shortage of educational staff
        'EDUSHORT',        # Shortage of educational material
        'PROATCE',         # Proportion of fully certified teachers
        'TEACHSUP',        # Matehmatics teacher support
        'SCHLTYPE',        # School ownership type
        'ST059Q01TA',      # Number of math periods per week (required to build MMINS)
        'SC175Q01JA',      # Average duration of a single math period (required to build MMINS)
        'ESCS',            # Index of economic, social and cultural status (required to build MEAN_ESCS)
    ]
    
    # Filter the DataFrame to keep only the important columns that actually exist
    existing_columns = [col for col in important_columns if col in df.columns]
    
    if existing_columns:
        df_reduced = df[existing_columns].copy()
        print(f"Reduced shape: {df_reduced.shape} ({len(existing_columns)} columns)")
        print(f"Columns kept: {existing_columns}")
    else:
        print("No important columns found, returning full dataset")
        df_reduced = df
    
    print(df_reduced.head())

    #Save the swiss merged dataset in csv format
    print(f"Saving reduced swiss dataset in the src directory...")
    os.makedirs("src", exist_ok=True)
    src_path = os.path.join("src", "swiss_reduced_dataset.csv")
    df_reduced.to_csv(src_path, index=False)
    print(f"Swiss reduced dataset saved to: {src_path} ✔")

    return df_reduced
