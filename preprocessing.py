import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder



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

def calculate_time_to_event(df):
    """
    Calculate the 'Time to Event' for each patient with a different 'Record ID'
    based on 'Date of Blood Draw' and the next non-empty 'Date of Thrombosis'.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the dataset.

    Returns:
    - df (pandas.DataFrame): The DataFrame with the 'Time to Event' column added.

    Description:
    This function calculates the 'Time to Event' for each patient with a different 'Record ID'
    based on the difference between the 'Date of Blood Draw' and the next non-empty 'Date of Thrombosis'.
    The DataFrame is first sorted by 'Record ID' and 'Date of Thrombosis' to ensure proper calculations.

    Example Usage:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Record ID': [1, 1, 2, 2],
    ...                    'Date of Blood Draw': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'],
    ...                    'Date of Thrombosis': [None, '2022-03-15', None, '2022-04-20']})
    >>> df = calculate_time_to_event(df)
    >>> print(df)
       Record ID Date of Blood Draw Date of Thrombosis  Time to Event
    0         1         2022-01-01               None            NaN
    1         1         2022-02-01         2022-03-15           43.0
    2         2         2022-03-01               None            NaN
    3         2         2022-04-01         2022-04-20           20.0
    """
    # Sort the DataFrame by "Record ID" and "Date of Thrombosis"
    df.sort_values(by=["Record ID", "Visit Timepoint"], inplace=True)


    # Group the DataFrame by "Record ID"
    grouped = df.groupby("Record ID")

    # Create an empty list to store time to event values
    time_to_event = []

    # Iterate through each group (each patient)
    for _, group in grouped:
        group["Time to Event"] = (group["Date of Thrombosis"].bfill() - group["Date of Blood Draw"]).dt.days
        time_to_event.extend(group["Time to Event"])

    # Assign the calculated values back to the DataFrame
    df["Time to Event"] = time_to_event

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

def binarize_thrombotic_event(df):
    """
    Create a new 'Label' column in a DataFrame based on the 'Date of Thrombosis' column.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the dataset.

    Returns:
    - df (pandas.DataFrame): The DataFrame with the 'Label' column added and 'Date of Thrombosis' column optionally dropped.

    Description:
    This function checks the 'Date of Thrombosis' column in the provided DataFrame for non-null values.
    It creates a new 'Label' column where 'True' represents the presence of non-null values in 'Date of Thrombosis'
    and 'False' represents null values or empty entries. Optionally, the original 'Date of Thrombosis' column
    can be dropped from the DataFrame.

    Example Usage:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Date of Thrombosis': ['2022-01-01', None, '2022-02-01']})
    >>> df = make_label(df)
    >>> print(df)
       Label
    0   True
    1  False
    2   True
    """

    # Check if the "Date of Thrombosis" column contains any non-null values
    df['Thrombosis_event'] = df['Date of Thrombosis'].notnull()
    df['Upcoming_event'] = df['Time to Event'].notnull()

    # Convert the boolean values to True/False (if needed)
    df['Thrombosis_event'] = df['Thrombosis_event'].astype(bool)
    df['Upcoming_event'] = df['Upcoming_event'].astype(bool)

    # Optionally, drop the original "Date of Thrombosis" column
    df = df.drop('Date of Thrombosis', axis=1)

    return df

def to_boolean(df):
    # Map 'yes' to True and 'no' to False for the specified columns
    df['Statin'] = df['Statin'].map({'yes': True, 'no': False})
    df['Cilostazol'] = df['Cilostazol'].map({'yes': True, 'no': False})

    # Optional: Convert the columns to boolean data type
    df['Statin'] = df['Statin'].astype(bool)
    df['Cilostazol'] = df['Cilostazol'].astype(bool)
    return df 

def preprocess(df):
    """
    Encodes ordinal categorical values and creates columns representing the 
    rate of change of TEG results.
    It returns the preprocessed DataFrame.
    """
    # Encode dategorical
    df = encode_categorical_features(df)

    # Encode ordinal features
    df = encode_ordinal_features(df)
    df = encode_timepoint(df)

    # "Statin" and "Cilostazol" columns into Boolean 
    df = to_boolean(df)
    
    # Assuming df is your DataFrame and columns_to_process contains the specified columns
    TEG_values = [
        'Reaction Time (R) in min', 'Lysis at 30 min (LY30) in %', 'CRT Max amplitude (MA) in mm',
        'CFF Max Amplitude (MA) in mm', 'HKH MA (mm)', 'ActF MA (mm)', 'ADP MA (mm)', 'AA MA (mm)',
        'ADP % Aggregation', 'ADP % Inhibition', 'AA % Aggregation', 'AA % Inhibition', 'CK R (min)',
        'CK K (min)', 'CK angle (deg)', 'CK MA (mm)', 'CRT MA (mm)', 'CKH R (min)', 'CFF MA (mm)'
    ]

    # Process the DataFrame to calculate rates since the last time point
    df = calculate_difference_and_rate_since_last_timepoint(df, TEG_values)

    # Calculate time to event
    df = calculate_time_to_event(df)

    # Convert trombotic event to label column
    df = binarize_thrombotic_event(df)

    return df


# # Main script
# data_path = "./data/DummyData_Extended.xlsx"
# df = pd.read_excel(data_path)
# df = preprocess(df)

# output_file = './data/Preprocessed_Data.xlsx'
# df.to_excel(output_file, index=False)
# print("Data saved")
