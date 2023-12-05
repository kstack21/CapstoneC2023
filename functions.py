# Import dependencies
import pandas as pd
import shap
import re
from IPython.display import Image, display
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

def merge_events_count(baseline_df, tegValues_df, events_df):
    # Count the number of events for each 'Record_ID' in events_df
    event_counts = events_df['Record ID'].value_counts().reset_index()
    event_counts.columns = ['Record ID', 'Events']

    # Merge the event counts with the baseline and teg values
    tegValues_df = tegValues_df.merge(event_counts, on='Record ID', how='left')
    baseline_df = baseline_df.merge(event_counts, on='Record ID', how='left')

    # Fill NaN values in the 'event_count' column with 0
    tegValues_df['Events'].fillna(0, inplace=True)
    baseline_df['Events'].fillna(0, inplace=True)

    return baseline_df, tegValues_df

def transform_data(baseline_df, tegValues_df, boundaries, training = True):
    """
    Transform and clean the given baseline and TEG values DataFrames.

    Parameters:
    - baseline_df (pd.DataFrame): DataFrame containing baseline data.
    - tegValues_df (pd.DataFrame): DataFrame containing TEG values data.
    - boundaries (dictionary): Boundary parameters for TEG values.
    - timepoints(dictionary): Meaning of timestamps in numerical values.

    Returns:
    - clean_baseline_df (pd.DataFrame): Transformed and cleaned baseline DataFrame.
    - clean_TEG_df (pd.DataFrame): Transformed and cleaned TEG values DataFrame.
    - tegValues (list): List of column names of tegValues_df excluding some values

    The function performs the following transformations and cleaning steps:

    1. Create copies of the input DataFrames to avoid modifying the original data.
    2. Handle numerical columns:
        - Extract relevant numerical columns for baseline and TEG values.
        - Split 'BP prior to blood draw' column into 'BP_Systolic' and 'BP_Diastolic' columns.
        - Convert 'BP_Systolic' and 'BP_Diastolic' columns to integers.
        - Remove unnecessary columns.
        - Clean and replace boundary values for EGFR and TEG columns.
        - Convert 'Rutherford Score' and TEG values to floats.
        - Convert timepoints from strings to integers representing days after the operation.

    3. Handle Boolean columns:
        - Create 'Is Male' column based on the 'Sex' column.
        - Convert specified columns to boolean values.

    4. Handle Categorical Ordinal columns:
        - Encode ordinal values for specified columns.

    5. Handle Categorical Nominal columns:
        - Dummy encode specified columns for both baseline and TEG values.

    6. Handle Artery Affected, Antiplatelet Therapy, Intervention Types, and Anticoagulation Medications:
        - Dummy encode specific values and create new columns for each unique value.

    Note:
    - The function utilizes external JSON files ('data_boundaries.json' and 'timepoints.json') for boundary values and timepoint mappings.
    - Columns and values are cleaned, replaced, and encoded according to predefined rules and mappings.

    Example Usage:
    ```python
    # Assuming baseline_df and tegValues_df are loaded DataFrames
    clean_baseline_df, clean_TEG_df = transform_data(baseline_df, tegValues_df)
    ```
    """

    # Clean df in new copy
    clean_TEG_df = tegValues_df.copy()
    clean_baseline_df = baseline_df.copy()


    # NUMBER #
    # Find teg values column
    columns_to_exclude = ['Record ID', 'Visit Timepoint', 'Antiplatelet Therapy within 7 Days',
                        'Anticoagulation within 24 Hours', 'Statin within 24 Hours', 'Cilostazol within 7 days',
                        'BP prior to blood draw', 'Events', 'Date of TEG Collection']

    tegValues = [col for col in tegValues_df.columns.values if col not in columns_to_exclude]

    number_columns_baseline = ["Age","BMI", "Clotting Disorder", "EGFR (mL/min/1.73m2)", "ABI Right", "ABI left", "Rutherford Score"]
    number_columns_teg = ["BP prior to blood draw"]+tegValues


    # Split the column into 'Systolic' and 'Diastolic' columns
    clean_TEG_df[['BP_Systolic', 'BP_Diastolic']] = clean_TEG_df['BP prior to blood draw'].str.split('/', expand=True)

    # Convert 'Systolic' and 'Diastolic' columns to integers
    clean_TEG_df['BP_Systolic'] = pd.to_numeric(clean_TEG_df['BP_Systolic'], errors='coerce').astype('Int64')
    clean_TEG_df['BP_Diastolic'] = pd.to_numeric(clean_TEG_df['BP_Diastolic'], errors='coerce').astype('Int64')

    # Drop the first column 'BP prior to blood draw'
    clean_TEG_df.drop(columns=['BP prior to blood draw'], inplace = True)
    number_columns_teg.remove('BP prior to blood draw')
    number_columns_teg.append('BP_Systolic')
    number_columns_teg.append('BP_Diastolic')

    clean_TEG_df[['BP_Systolic', 'BP_Diastolic']].dtypes


    # Clean EGFR and TEG data with boundary values and convert all to floats

    # Replace all boundary values with their correcponding right values
    # EGFR
    egfr_column = 'EGFR (mL/min/1.73m2)'
    efgr_replacement = boundaries.pop(egfr_column, None)
    # Remove spaces in the column
    clean_baseline_df[egfr_column] = clean_baseline_df[egfr_column].replace(regex={r'\s': ''})

    # Use a regular expression to match and replace values
    for name, replacement in efgr_replacement.items():
        clean_baseline_df[egfr_column] = clean_baseline_df[egfr_column].replace({f'^{name}': replacement}, regex=True)

    # Iterate over TEG DataFrame and apply boundaries
    for column, replacement_dict in boundaries.items():
        
        # Remove spaces in the column
        clean_TEG_df[column] = clean_TEG_df[column].replace(regex={r'\s': ''})
        
        # Use a regular expression to match and replace values
        for name, replacement in replacement_dict.items():
            clean_TEG_df[column] = clean_TEG_df[column].replace({f'^{name}': replacement}, regex=True)


    # Convert  Rutherford Score and TEG values to float
    clean_baseline_df["Rutherford Score"] = pd.to_numeric(clean_baseline_df["Rutherford Score"], errors='coerce')
    clean_baseline_df["Rutherford Score"].dtypes

    # Loop through the columns and convert to numeric
    for column in tegValues:
        clean_TEG_df[column] = pd.to_numeric(clean_TEG_df[column], errors='coerce')

    
    # Convert ABI values to floats
    clean_baseline_df['ABI Right'] = pd.to_numeric(clean_baseline_df['ABI Right'], errors='coerce')
    clean_baseline_df['ABI left'] = pd.to_numeric(clean_baseline_df['ABI left'], errors='coerce')

    #### Booleans
    # Create the 'Is Male' column based on the 'sex' column
    clean_baseline_df['Is Male'] = (clean_baseline_df['Sex'] == 'Male').astype(bool)

    # Drop the old 'sex' column
    clean_baseline_df.drop('Sex', axis=1, inplace=True)
    clean_baseline_df['Is Male']

    # Change following columns to booleans
    columns_to_convert_baseline = ['White', 'Diabetes', 'Hypertension', 'Hyperlipidemia (choice=None)', 'Coronary Artery Disease', 'History of MI',
                        'Functional impairment', 'Does Subject Currently have cancer?', 'Past hx of cancer', 'Hx of  DVT', 'Hx of stroke',
                        'Hx of pulmonary embolism', 'Does the patient have a history of solid organ transplant?', 
                        'Has subject had previous intervention of the index limb?', 'Previous occluded stents',]
    columns_to_convert_TEG =['Cilostazol within 7 days']

    # Dictionary for replacement
    replacement_dict = {'yes': True, 'no': False, '1': True, '0': False, 'cilostazol': True, 'NaN':False}

    # Fill NaN values with False
    clean_baseline_df[columns_to_convert_baseline] = clean_baseline_df[columns_to_convert_baseline].fillna('0')
    clean_TEG_df[columns_to_convert_TEG] = clean_TEG_df[columns_to_convert_TEG].fillna('0')

    # Put all columns in lowercase
    clean_baseline_df[columns_to_convert_baseline] = clean_baseline_df[columns_to_convert_baseline].astype(str)
    clean_baseline_df[columns_to_convert_baseline] = clean_baseline_df[columns_to_convert_baseline].apply(lambda x: x.str.lower())
    clean_TEG_df[columns_to_convert_TEG] = clean_TEG_df[columns_to_convert_TEG].astype(str)
    clean_TEG_df[columns_to_convert_TEG] = clean_TEG_df[columns_to_convert_TEG].apply(lambda x: x.str.lower())

    # Use the replace method to replace values in multiple columns
    clean_baseline_df[columns_to_convert_baseline] = clean_baseline_df[columns_to_convert_baseline].replace(replacement_dict).astype(bool)
    clean_TEG_df[columns_to_convert_TEG] = clean_TEG_df[columns_to_convert_TEG].replace(replacement_dict).astype(bool)


    ## Categorical ordinal

    # Ordinal encoding map
    category_orders = {
        'Tobacco Use (1 current 2 former, 3 none)': 
        ['None',
        'Past, quit >10 year ago',
        'quit 1 to 10 years ago', 
        'current within the last year ( < 1 pack a day)',
        'current within the last year (  > or = 1 pack a day)'],

        'Renal Status': 
        ['Normal', 
        'GFR 30 to 59', 
        'GFR 15 to 29', 
        'GFR<15 or patient is on dialysis',
        '1']
    }

    # Replace renal status values. Some of the values in the data set mean the same with different words
    # Define a dictionary to map old values to new values
    replace_dict = {'GFR 60 to 89': 'Normal', 'Evidence of renal dysfunction ( GFR >90)': 'Normal', '0': 'Normal', 0: 'Normal', 1: "1"}

    clean_baseline_df['Renal Status'] = clean_baseline_df['Renal Status'].replace(replace_dict)

    # Replace NaN values with a new category
    # Replace NaN values with a new category
    new_category = 'Unknown'
    clean_baseline_df[['Tobacco Use (1 current 2 former, 3 none)', 'Renal Status']] = clean_baseline_df[['Tobacco Use (1 current 2 former, 3 none)', 'Renal Status']].fillna(new_category)

    # Initialize the OrdinalEncoder with specified category orders
    encoder = OrdinalEncoder(categories=[category_orders[column] + [new_category] for column in ['Tobacco Use (1 current 2 former, 3 none)', 'Renal Status']])

    # Fit and transform the selected columns to encode ordinal values
    clean_baseline_df[['Tobacco Use (1 current 2 former, 3 none)', 'Renal Status']] = encoder.fit_transform(clean_baseline_df[['Tobacco Use (1 current 2 former, 3 none)', 'Renal Status']])

    # Rename column
    clean_baseline_df = clean_baseline_df.rename(columns={'Tobacco Use (1 current 2 former, 3 none)': 'Tobacco Use'})

    #### Categorical nominal
    columns_to_dummy_baseline = ['Extremity',
                        'Intervention Classification']
    columns_to_dummy_TEG = ['Statin within 24 Hours']

    # Dummy encoding of categorical values
    clean_baseline_df = pd.get_dummies(clean_baseline_df, columns=columns_to_dummy_baseline,
                        prefix=columns_to_dummy_baseline)
    clean_TEG_df = pd.get_dummies(clean_TEG_df, columns=columns_to_dummy_TEG,
                        prefix=columns_to_dummy_TEG)

    if training:
        # Drop unecessary columns
        clean_baseline_df = clean_baseline_df.drop(columns=['Extremity_left']) # Because it is either right, left or bilateral
        clean_baseline_df = clean_baseline_df.drop(columns=['Intervention Classification_Endo']) # Either endo, open or combined

    # Artery affected

    # Get all unique valuses
    unique_arteries = set()
    unique_antiplatelet = set()
    unique_intervention = set()
    unique_anticoagulation = set()

    for index, row in clean_baseline_df.iterrows():
        # Check if the value is a string before splitting
        if isinstance(row['Artery affected'], str):
            arteries = row['Artery affected'].split(', ')
            unique_arteries.update(arteries)

        # Check if the value is a string before splitting
        if isinstance(row['Intervention Type'], str):
            intervention = row['Intervention Type'].split(', ')
            unique_intervention.update(intervention)
        

    for index, row in clean_TEG_df.iterrows():
        # Check if the value is a string and not NaN before splitting
        if isinstance(row['Antiplatelet Therapy within 7 Days'], str):
            antiplatelet = row['Antiplatelet Therapy within 7 Days'].split(', ')
            unique_antiplatelet.update(antiplatelet)

        # Check if the value is a string and not NaN before splitting
        if isinstance(row['Anticoagulation within 24 Hours'], str):
            anticoagulation = row['Anticoagulation within 24 Hours'].split(', ')
            # Delete items in parenthesis ex: heparin (Calciparine) to be just heparin
            anticoagulation = {re.sub(r'\s*\([^)]*\)\s*', '', item) for item in anticoagulation}
            unique_anticoagulation.update(anticoagulation)

    # Fill NaN values with 0 in the 'Artery affected' column
    clean_baseline_df['Artery affected'].fillna(0, inplace=True)

    # Dummy encode arteries affected
    selected_arteries = []
    for artery in unique_arteries:
        column_name = "Artery affected_" + artery
        # Convert to int after filling NaN with 0
        clean_baseline_df[column_name] = clean_baseline_df['Artery affected'].str.contains(artery).fillna(0).astype(int)
        selected_arteries.append(column_name)

    selected_arteries.append('Artery affected')

    # Fill NaN values with 0 in the 'Antiplatelet Therapy within 7 Days' column
    clean_TEG_df['Antiplatelet Therapy within 7 Days'].fillna(0, inplace=True)

    # Dummy encode antiplatelet therapy
    selected_antiplatelet = []
    for antiplatelet in unique_antiplatelet:
        column_name = "Antiplatelet therapy_" + antiplatelet

        # Handle NaN values and convert to int
        clean_TEG_df[column_name] = clean_TEG_df['Antiplatelet Therapy within 7 Days'].apply(lambda x: 1 if antiplatelet in str(x) else 0).astype(int)

        selected_antiplatelet.append(column_name)

    selected_antiplatelet.append('Antiplatelet Therapy within 7 Days')

    # Fill NaN values with an empty string in the 'Intervention Type' column
    clean_baseline_df['Intervention Type'].fillna('', inplace=True)

    # Dummy encode intervention types
    selected_intervention = []
    for intervention in unique_intervention:
        column_name = 'Intervention type_' + intervention

        # Handle NaN values and convert to int
        clean_baseline_df[column_name] = clean_baseline_df['Intervention Type'].apply(lambda x: 1 if intervention in str(x) else 0).astype(int)

        selected_intervention.append(column_name)


    selected_intervention.append('Intervention Type')

    # Fill NaN values with an empty string in the 'Anticoagulation within 24 Hours' column
    clean_TEG_df['Anticoagulation within 24 Hours'].fillna('', inplace=True)

    # Dummy encode anticoagulation meds
    selected_anticoagulation = []
    for anticoagulation in unique_anticoagulation:
        column_name = "Anticoagulation_" + anticoagulation

        # Handle NaN values and convert to int
        clean_TEG_df[column_name] = clean_TEG_df['Anticoagulation within 24 Hours'].apply(lambda x: 1 if anticoagulation in str(x) else 0).astype(int)

        selected_anticoagulation.append(column_name)

    selected_anticoagulation.append('Anticoagulation within 24 Hours')

    # Drop old columns
    clean_baseline_df.drop(columns=['Artery affected','Intervention Type'], inplace=True)
    clean_TEG_df.drop(columns=['Antiplatelet Therapy within 7 Days', 'Anticoagulation within 24 Hours'], inplace=True)

    return clean_baseline_df, clean_TEG_df, tegValues


def extend_df (clean_TEG_df, tegValues):
    extended_df = clean_TEG_df.copy()
   
   # Make sure column is in date time format
    extended_df['Date of TEG Collection'] = pd.to_datetime(extended_df['Date of TEG Collection']).dt.date #BIG CHANGE

    # Sort the DataFrame by "Record ID" and "Date of TEG Collection"
    extended_df= extended_df.sort_values(by=["Record ID", "Date of TEG Collection"])
    extended_df[["Record ID", "Date of TEG Collection"]]


    # Group by 'Record ID'
    grouped = extended_df.groupby('Record ID')

    #Calculate the difference in 'Date of TEG Collection'
    extended_df['Days Diff'] = grouped['Date of TEG Collection'].diff()

    # Replace 0s to avoid infinity
    extended_df['Days Diff'] = pd.to_timedelta(extended_df['Days Diff']).dt.total_seconds() / (24 * 60 * 60)
    extended_df["Days Diff"] = extended_df["Days Diff"].replace(0, 1)


    new_columns = []
    # Iterate TEG values
    for value in tegValues:

        # Get column names
        diff_column_name = f"{value}_difference_since_last_timepoint"
        rate_column_name = f"{value}_rate_since_last_timepoint"
        new_columns.append(diff_column_name)
        new_columns.append(rate_column_name)


        # Calculate the difference in TEG values
        extended_df[diff_column_name] = grouped[value].diff()

        # Divide  by the differences in "Date of TEG Collection"
        extended_df[rate_column_name] = extended_df[diff_column_name] / extended_df['Days Diff']

    # Fill the first value with the next one to avoid NaN
    extended_df.bfill(inplace=True)


    # Drop column with diff in dates
    extended_df.drop(columns=["Days Diff"], inplace = True)

    return extended_df, new_columns

def visualize_data(clean_baseline_df, clean_TEG_df):
    """
    Visualize key statistics and distributions of the given cleaned baseline and TEG values DataFrames.

    Parameters:
    - clean_baseline_df (pd.DataFrame): Cleaned baseline DataFrame.
    - clean_TEG_df (pd.DataFrame): Cleaned TEG values DataFrame.

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly Figure object containing visualizations.

    The function generates visualizations including:
    1. Gender Distribution Pie Chart: Displays the distribution of gender in the dataset.
    2. Ethnicity Distribution Pie Chart: Displays the distribution of ethnicity in the dataset.
    3. Thrombotic Event Histogram: Displays the distribution of thrombotic events.
    4. BMI Histogram: Displays the distribution of BMI (Body Mass Index).
    5. Age Histogram: Displays the distribution of patient ages.
    6. Data Summary Table: Provides a summary of unique patients and total data points.

    Custom colors are used for visualizations, and the function utilizes the Plotly library to create subplots.

    Example Usage:
    ```python
    # Assuming clean_baseline_df and clean_TEG_df are cleaned DataFrames
    visualization_figure = visualize_data(clean_baseline_df, clean_TEG_df)
    ```
    """

    fig_df = clean_baseline_df.copy()

    # Define custom colors
    male_colors = ['#d9ed92', '#5c8254'] 
    white_colors = ['#184e77', '#76acc5'] 
    events_colors = '#1a759f'
    age_histogram_color = '#52b69a' 
    bmi_histogram_color = '#009900'

    # Count binary values in the "Male" column
    male_counts = fig_df['Is Male'].value_counts()
    male_labels = ['Male' if male_counts.index[0] else 'Female', 'Male' if not male_counts.index[0] else 'Female']
    # Create a pie chart for "Male" with custom colors
    sex_pie = go.Pie(labels=male_labels, values=male_counts, marker=dict(colors=male_colors))

    # Count binary values in the "White" column
    white_counts = fig_df['White'].value_counts()
    white_labels = ['White' if white_counts.index[0] else 'Non-White', 'White' if not white_counts.index[0] else 'Non-White']

    # Create a pie chart for "White" with custom colors
    white_pie = go.Pie(labels=white_labels, values=white_counts, marker=dict(colors=white_colors))

    # BMI histogram
    bmi_hist =  go.Histogram(x=fig_df["BMI"], name="BMI", marker=dict(color=bmi_histogram_color))

    # Age histogram
    age_hist=  go.Histogram(x=fig_df["Age"], name="Age", marker=dict(color=age_histogram_color))

    # The following metrics are bsed on the total number of TEG test values
    # Copy TEG df to find metrics
    fig_df = clean_TEG_df.copy()

    # Events histogram 
    events_hist =  go.Histogram(x=fig_df["Events"], name="Events", marker=dict(color=events_colors))

    # Create a summary table
    unique_patients = fig_df['Record ID'].nunique()
    total_data_points = len(fig_df)

    data_summary = pd.DataFrame({
        'Category': ['Unique Patients', 'Total Data Points'],
        'Count': [unique_patients, total_data_points]
    })

    patients_table = go.Table(
        header=dict(values=["Category", "Count"]),
        cells=dict(values=[data_summary['Category'], data_summary['Count']])
    )

    # Create subplots
    fig = make_subplots(rows=2, cols=3,
                        specs=[[{'type':'domain'}, {'type':'domain'},{'type':'xy'}],
                                [{'type':'xy'}, {'type':'xy'},{'type':'domain'}]],
                        subplot_titles=['Gender Distribution', 'Ethnicity Distribution', 'Thrombotic event', 'BMI',
                                        'Age', 'Data Summary'])

    fig.add_trace(sex_pie, row=1, col=1)
    fig.add_trace(white_pie, row=1, col=2)
    fig.add_trace(events_hist, row=1, col=3)
    fig.add_trace(bmi_hist, row=2, col=1)
    fig.add_trace(age_hist, row=2, col=2)
    fig.add_trace(patients_table, row=2, col=3)

    fig.update_layout(width=900, height=600)

    return fig

def train_model(df, target_column, drop_columns, quantile_range=(5,95), param_grid = {
        'xgb_regressor__max_depth': [3, 4, 5],
        'xgb_regressor__gamma': [0, 0.1, 0.2],
        'xgb_regressor__min_child_weight': [1, 2, 5]},
        scoring = 'r2'):
    """
    Trains an XGBoost regression model on the given DataFrame using grid search for hyperparameter tuning.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the features and target variable.
    - target_column (str): The name of the target variable column.
    - drop_columns (list): List of column names to be dropped from the feature set.

    Returns:
    - best_pipeline (Pipeline): The best-performing pipeline after hyperparameter tuning.

    Example:
    best_model = train_model(df=my_dataframe, target_column='target', drop_columns=['column1', 'column2'])
    """

    # Separate features (X) and target (y)
    y = df[target_column]

    drop_columns = drop_columns + [target_column]
    X = df.copy()
    for column in drop_columns:
        if column in X.columns:
            X.drop(column, axis=1,inplace = True)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create transformers for feature scaling
    target_scaler = RobustScaler(quantile_range=quantile_range)
   

    # Scale target
    # Fit the target scaler on training target and transform both training and test target
    y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()


    # Manually adjust the scaled data to center at 0.5
    desired_center = 0.5
    y_train = y_train + (desired_center - np.median(y_train, axis=0))
    y_test = y_test + (desired_center - np.median(y_test, axis=0))

    # Create a pipeline
    pipeline = Pipeline([
        ('feature_scaler', RobustScaler()),  # Robust scaling for features
        ('xgb_regressor', XGBRegressor())    # XGBoost regressor
    ])

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                               scoring=scoring, cv=kf)

    # Fit the model and perform hyperparameter tuning
    grid_search.fit(X_train, y_train)

    
    # Access the best pipeline
    best_pipeline = grid_search.best_estimator_

    # Make predictions on the test data
    y_pred = best_pipeline.predict(X_test)  
    # Evaluate the model using Mean Squared Error
    mse_test = mean_squared_error(y_test, y_pred)
    # Calculate R-squared (R2) score
    r2_test = r2_score(y_test, y_pred)
    print(y_test, y_pred)

    # Make predictions on the train data
    y_pred = best_pipeline.predict(X_train)  
    # Evaluate the model using Mean Squared Error
    mse_train = mean_squared_error(y_train, y_pred)
    # Calculate R-squared (R2) score
    r2_train = r2_score(y_train, y_pred)
    print(y_train, y_pred)
    
    score = {"mse test":mse_test, "r2 test": r2_test, "mse train": mse_train, "r2 train": r2_train}

    return best_pipeline, X_train, score

def feature_importance(best_pipeline, X):
    """
    Generate SHAP (SHapley Additive exPlanations) values and a summary plot for feature importance.

    Parameters:
    - best_pipeline (Pipeline): The best-performing pipeline after hyperparameter tuning. It should have an XGBoost regressor named 'xgb_regressor'.
    - X (pd.DataFrame): Data to be tested, containing features for which SHAP values will be computed.

    Returns:
    - importance_df (pd.DataFrame): DataFrame containing feature names and their importance values.
    - shap_values (numpy.ndarray): SHAP values for the provided data.

    Example:
    importance_df, shap_values = feature_importance(best_pipeline=my_best_pipeline, X=my_test_data)
    
    Note:
    The SHAP (SHapley Additive exPlanations) values provide insights into the contribution of each feature to model predictions. The summary plot and importance DataFrame help identify the most influential features.

    Dependencies:
    - Ensure the 'shap' library is installed. You can install it using 'pip install shap'.

    Usage:
    - For the best results, pass the best-performing pipeline obtained after hyperparameter tuning. The pipeline should include an XGBoost regressor with the name 'xgb_regressor'.

    """
    # Create a SHAP explainer for the XGBoost model
    explainer = shap.Explainer(best_pipeline.named_steps['xgb_regressor'])

    # Generate SHAP values
    shap_values = explainer.shap_values(X)

    # Calculate feature importance using the absolute mean of SHAP values
    feature_importance = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame to associate feature names with their importance values
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

    # Sort the DataFrame by importance in descending order to find the most important features
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Multiply all values by 100
    importance_df['Importance'] *= 100

    return importance_df

def plot_importance(importance_df, title, showlegend = True):

    try:
        # Set 'Feature' column as the index
        importance_df = importance_df.set_index('Feature')
        # Rename the index from 'Feature' to 'Factors'
        importance_df.rename_axis(index='Factors', inplace=True)
    except:
        importance_df = importance_df.copy()

    # Select the first 10 rows
    top_10_df = importance_df.head(10)
    # Create a horizontal bar chart using Plotly Express
    fig = px.bar(
        top_10_df,
        orientation='h',  # Horizontal bars
        title= title,
        labels={'index': 'Factors', 'value': 'Percentage'},
    )

    # Reverse the order of the y-axis (largest value at the top)
    fig.update_layout(yaxis_categoryorder='total ascending')

    fig.update_layout(showlegend=showlegend)

    return fig

def user_options(extended_df, tegValues, new_columns, importance_df_TEG, user_extend_data = False):
   
    user_TEG_df = extended_df.copy()


    # Keep only the most important values from teg. No need for extra created ones
    if user_extend_data:
        columns_to_keep = dict.fromkeys(user_TEG_df.columns.difference(tegValues + new_columns), None)
    else:
        columns_to_keep = dict.fromkeys(user_TEG_df.columns.difference(tegValues), None)

    # Iterate through prefixes and select the most important column for each
    for prefix in tegValues:
        # Filter the importance_df_TEG1 for the current prefix
        prefix_columns = importance_df_TEG[importance_df_TEG['Feature'].str.startswith(prefix)]

        if not prefix_columns.empty:
            # Find the column with the maximum importance for the current prefix
            max_importance_row = prefix_columns.loc[prefix_columns['Importance'].idxmax()]

            # Check if the maximum importance value is greater than 0
            if max_importance_row['Importance'] > 0:
                max_importance_column = max_importance_row['Feature']
                columns_to_keep[max_importance_column] =max_importance_row['Importance']

            else:
                columns_to_keep[prefix] = 0


    # Keep only non repeated values
    user_TEG_df = user_TEG_df[columns_to_keep.keys()]

    return columns_to_keep

def user_selection(selected_features, columns_to_keep, collinearity):

    # Extract all values from selected_features and collinearity
    selected_features_values = list(selected_features.values())
    collinearity_values = [item for sublist in collinearity.values() for item in sublist]

    # Find prefixes to drop
    prefix_to_keep = [prefix for selection in selected_features_values for prefix in collinearity_values if selection.startswith(prefix)]
    prefix_to_drop = list(set(collinearity_values) - set(prefix_to_keep))

    # Find list of columns to drop
    columns_to_drop = [column for column in columns_to_keep.keys() if any(column.startswith(prefix) for prefix in prefix_to_drop)]

    return columns_to_drop

def check_column_names(df, model_column):
    """
    model_column (list) with column names originally used in model
    """
    
    # Ensure all columns in teg_columns exist in clean_TEG_df
    for column in model_column:
        if column not in df.columns:
            # If the column is missing, add an empty column
            df[column] = pd.Series(dtype=float)

    # Drop columns in clean_TEG_df that are not in teg_columns
    new_df = df[model_column]

    return new_df


def predict(df, best_pipeline, id_s):

    columns_to_drop = ['Record ID','Events','Visit Timepoint', 'Date of TEG Collection']
    df = df.copy()

    for column in columns_to_drop:
        if column in df.columns:
            df.drop(column, axis=1,inplace = True)

    # Make predictions on the test data
    y_pred = best_pipeline.predict(df)

    # Scale prediction to be percentage
    y_pred = y_pred*100


    # Convert NumPy array to DataFrame with 'Record ID' as the index
    pred_df = pd.DataFrame({'Prediction': y_pred})
   
    # Concatenate horizontally
    id_pred_df = pd.concat([pred_df, id_s], axis=1)

    
    return id_pred_df  

def iterate_importance(df, best_pipeline, ids):

    # Create a list of strings with the format "patient {record id}" or "patient {record id}: {date}"
    string_list = []

    # Check if the 'Date of TEG Collection' column exists in the DataFrame
    if 'Date of TEG Collection' in df.columns:
        # Convert 'Date of TEG Collection' to string format
        df['Date of TEG Collection'] = pd.to_datetime(df['Date of TEG Collection']).dt.strftime('%Y-%m-%d')

        for index, row in df.iterrows():
            record_id = row['Record ID']
            date = row['Date of TEG Collection']
            patient_string = f"Patient {record_id}: {date}"
            string_list.append(patient_string)

    else:
        
        for index, row in df.iterrows():
            record_id = row['Record ID']
            patient_string = f"Patient {record_id}"
            string_list.append(patient_string)

    
    df = df.copy()
    
    columns_to_drop = ['Record ID','Events', 'Visit Timepoint', 'Date of TEG Collection']

    for column in columns_to_drop:
        if column in df.columns:
            df.drop(column, axis=1,inplace = True)

    total_df = pd.DataFrame()

    # Iterate through all rows in the DataFrame
    for index, row in df.iterrows():
        # Convert the row to a DataFrame with a single row
        single_row_df = pd.DataFrame([row])

        # Calculate shap_values for the current row
        importance_df = feature_importance(best_pipeline, single_row_df)

        # Set 'Feature' column as the index
        importance_df.set_index('Feature', inplace=True)

        # Add a new column to total_df with the corresponding index value from ids
        total_df[string_list[index]] = importance_df['Importance']

    # Drop rows where all values are zeros
    total_df = total_df.loc[(total_df != 0).any(axis=1)]

    # Sort columns based on the highest average value
    total_df = total_df.reindex(total_df.mean(axis=1).sort_values(ascending=False).index)

    # Rename the index from 'Feature' to 'Factors'
    total_df.rename_axis(index='Factors', inplace=True)

    return total_df


def plot_pred(TEG_pred,baseline_pred):
    # Convert Date of TEG Collection to string for plotting
    plot_TEG_df = TEG_pred.copy()
    plot_TEG_df['Date of TEG Collection'] = plot_TEG_df['Date of TEG Collection'].dt.strftime('%Y-%m-%d')

    # Plotly Express Scatter Plot for TEG1_pred
    fig = px.line(plot_TEG_df, x='Date of TEG Collection', y='Prediction',
                    color='Record ID', symbol='Record ID',
                    title="Patient's risk of thrombosis based on TEG values and comorbidities",
                    labels={'Prediction': 'Risk(%)'})

    # Get baseline values
    # Find the smallest and largest values
    smallest_value = TEG_pred['Date of TEG Collection'].min()
    largest_value = TEG_pred['Date of TEG Collection'].max()

    # Plotly Express Line Plot for baseline_pred
    double_baseline_pred = pd.concat([baseline_pred, baseline_pred])
    baseline_pred_line = px.line(double_baseline_pred, x=[smallest_value,largest_value] * len(baseline_pred), y='Prediction', line_group='Record ID')
    baseline_pred_line.update_traces(mode='lines', line_dash='dash', name='Baseline')
    fig.add_trace(baseline_pred_line.data[0])

    return fig