import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

def scale_features(df):
    """
    Scales specified numerical features using RobustScaler,
    making them robust to outliers.
    """
    # Specify the columns to be encoded robustly
    columns_to_scale = [
        'Age', 'BMI', 'HbA1c Baseline', 'EGFR (mL/min/1.73m²)',
        'ABI Right', 'ABI Left', 'NAPT', 'MAPT', 'DAPT',
        'Reaction Time (R) in min', 'Lysis at 30 min (LY30) in %',
        'CRT Max amplitude (MA) in mm', 'CFF Max Amplitude (MA) in mm',
        'HKH MA (mm)', 'ActF MA (mm)', 'ADP MA (mm)', 'AA MA (mm)',
        'ADP % Aggregation', 'ADP % Inhibition', 'AA % Aggregation',
        'AA % Inhibition', 'CK R (min)', 'CK K (min)', 'CK angle (deg)',
        'CK MA (mm)', 'CRT MA (mm)', 'CKH R (min)', 'CFF MA (mm)',
        'CFF FLEV (mg/dL)'
    ]

    # Initialize the RobustScaler
    scaler = RobustScaler()

    # Fit and transform the selected columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df

def encode_categorical_features(df):
    """
    Performs dummy encoding on specified categorical columns 
    to create binary indicator variables.
    """
    # Dummy encoding of categorical values
    df = pd.get_dummies(df, columns=['Artery affected', 'Extremity', 'Anticoagulation', 'Intervention Classification'],
                        prefix=['Artery affected', 'Extremity', 'Anticoagulation', 'Intervention Classification'])
    
    return df

def encode_ordinal_features(df):
    """ 
    Applies ordinal encoding to specified columns 
    with ordinal categorical values.
    """
    # Ordinal encoding
    category_orders = {
        'Hyperlipidemia': ['None', 'Moderate', 'High'],
        'Functional Status': ['Limited', 'Fair', 'Good', 'Excellent']
    }

    # Replace missing values with a specific category
    df['Hyperlipidemia'].fillna('None', inplace=True)

    # Initialize the OrdinalEncoder with specified category orders
    encoder = OrdinalEncoder(categories=[category_orders[column] for column in ['Hyperlipidemia', 'Functional Status']])

    # Fit and transform the selected columns to encode ordinal values
    df[['Hyperlipidemia', 'Functional Status']] = encoder.fit_transform(df[['Hyperlipidemia', 'Functional Status']])

    return df

def encode_timepoint(df):
    """
    Converts the "Visit Timepoint" column to integers, 
    mapping specific timepoints to numeric values
    """
    # Change timepoint to ints
    timepoint_mapping = {'Baseline': 0, '3 Months': 3, '6 Months': 6, '9 Months': 9, '12 Months': 12}

    # Replace values in the "Visit Timepoint" column with integers using the mapping
    df['Visit Timepoint'] = df['Visit Timepoint'].map(timepoint_mapping)

    return df

def calculate_difference_and_rate_since_last_timepoint(df, columns_to_process):
    """
    Calculate and add difference and rate columns for specified columns in a DataFrame.

    This function is designed for longitudinal data analysis, where it calculates the difference
    and rate of change for selected columns between consecutive time points for each record.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the data.
    - columns_to_process (list of str): A list of column names for which the difference and rate
      should be calculated.

    Returns:
    - pandas.DataFrame: The DataFrame with additional columns containing the difference and rate
      information for the specified columns.

    Example:
    >>> columns_to_process = ['Reaction Time (R) in min', 'Lysis at 30 min (LY30) in %']
    >>> df = calculate_difference_and_rate_since_last_timepoint(df, columns_to_process)
    >>> print(df)
    """
 
    # Sort the DataFrame by "Record ID" and "Visit Timepoint"
    df = df.sort_values(by=["Record ID", "Visit Timepoint"])

    # Initialize a dictionary to store the previous values and timepoints for each "Record ID"
    previous_values = {}
    previous_timepoints = {}

    # Create new columns for difference and rate calculations
    for column in columns_to_process:
        diff_column_name = f"{column}_difference_since_last_timepoint"
        rate_column_name = f"{column}_rate_since_last_timepoint"
        df[diff_column_name] = None
        df[rate_column_name] = None

    # Iterate through the DataFrame
    for index, row in df.iterrows():
        record_id = row["Record ID"]
        timepoint = row["Visit Timepoint"]

        # Skip the first time point for each "Record ID"
        if record_id in previous_values:
            for column in columns_to_process:
                value = row[column]
                previous_value = previous_values[record_id].get(column, None)
                previous_time = previous_timepoints[record_id].get(column, None)

                if previous_value is not None and previous_time is not None:
                    # Calculate the difference and rate since the last time point
                    time_difference = timepoint - previous_time
                    if time_difference > 0:
                        diff = value - previous_value
                        rate = diff / time_difference
                        df.at[index, f"{column}_difference_since_last_timepoint"] = diff
                        df.at[index, f"{column}_rate_since_last_timepoint"] = rate

        # Update the previous values and timepoints for the current "Record ID"
        if record_id not in previous_values:
            previous_values[record_id] = {}
            previous_timepoints[record_id] = {}
        for column in columns_to_process:
            previous_values[record_id][column] = row[column]
            previous_timepoints[record_id][column] = timepoint

    return df

# Assuming df is your DataFrame and columns_to_process contains the specified columns
columns_to_process = [
    'Reaction Time (R) in min', 'Lysis at 30 min (LY30) in %', 'CRT Max amplitude (MA) in mm',
    'CFF Max Amplitude (MA) in mm', 'HKH MA (mm)', 'ActF MA (mm)', 'ADP MA (mm)', 'AA MA (mm)',
    'ADP % Aggregation', 'ADP % Inhibition', 'AA % Aggregation', 'AA % Inhibition', 'CK R (min)',
    'CK K (min)', 'CK angle (deg)', 'CK MA (mm)', 'CRT MA (mm)', 'CKH R (min)', 'CFF MA (mm)'
]


def preprocess(df):
    """
    Orchestrates the entire data preprocessing pipeline 
    by sequentially applying the above functions. 
    It returns the preprocessed DataFrame.
    """
    
    df = scale_features(df)
    df = encode_categorical_features(df)
    df = encode_ordinal_features(df)
    df = encode_timepoint(df)

    # Process the DataFrame to calculate rates since the last time point
    df = calculate_difference_and_rate_since_last_timepoint(df, columns_to_process)

    return df


# # Main script
# data_path = "./data/DummyData_Extended.xlsx"
# df = pd.read_excel(data_path)
# df = preprocess(df)

# output_file = './data/Preprocessed_Data.xlsx'
# df.to_excel(output_file, index=False)
# print("Data saved")
