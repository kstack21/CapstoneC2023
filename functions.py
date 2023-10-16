"""
Module Name: functions.py
Description: This module contains a collection of functions used throughout the program. 
"""
# Import libraries
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler

def path_back_to(new_folder_name):
    """
    Navigate to a new folder path relative to the current script's location.

    This function takes a `new_folder_name` as input and calculates a new path by
    moving up from the current script's location and appending the specified folder name.

    Args:
        new_folder_name (str or List[str]): The name of the folder(s) to navigate to,
            relative to the current script's location. Can be a single string or a list
            of strings for nested folders.

    Returns:
        str: The newly constructed path based on the provided folder name(s).

    """
    # Get the directory name of the provided path
    directory_name = os.path.dirname(__file__)

    # Split the directory path into components
    directory_components = directory_name.split(os.path.sep)

    # # Remove the last folder 
    # if directory_components[-1]:
    #     directory_components.pop()

    # Add the new folder to the path
    for file in new_folder_name:
        directory_components.append(file)

    # Join the modified components to create the new path
    new_path = os.path.sep.join(directory_components)

    return new_path

def data_demographics_fig(df):
    """
    Create a set of subplots to visualize data distributions.

    This function generates a 2x3 grid of subplots using Plotly's make_subplots. The subplots include two pie charts
    representing the gender and ethnicity distributions, as well as two histograms depicting age and BMI distributions.
    It also includes a pie chart for the "Date of Thrombosis" column and a summary table.

    Parameters:
        - df (pandas.DataFrame): The DataFrame containing the data to be visualized.

    Returns:
        - fig (plotly.graph_objs.Figure): The Plotly figure containing the subplots.
    """
    
    # Define custom colors
    male_colors = ['#d9ed92', '#99d98c'] 
    white_colors = ['#184e77', '#1a759f'] 
    events_colors = ['#184e77', '#1a759f'] 
    age_histogram_color = '#52b69a' 
    bmi_histogram_color = '#1e6091' 

    # Create subplots
    fig = make_subplots(rows=2, cols=3,
                        specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],
                               [{'type':'xy'}, {'type':'xy'},{'type':'domain'}]],
                        subplot_titles=['Gender Distribution', 'Ethnicity Distribution', 'Thrombotic event', 'BMI',
                                        'Age', 'Data Summary'])

    # Count binary values in the "Male" column
    male_counts = df['Male'].value_counts()
    male_labels = ['Male' if male_counts.index[0] else 'Female', 'Male' if not male_counts.index[0] else 'Female']
    # Create a pie chart for "Male" with custom colors
    fig.add_trace(go.Pie(labels=male_labels, values=male_counts, marker=dict(colors=male_colors)), row=1, col=1)

    # Count binary values in the "White" column
    white_counts = df['White'].value_counts()
    white_labels = ['White' if white_counts.index[0] else 'Non-White', 'White' if not white_counts.index[0] else 'Non-White']
    # Create a pie chart for "White" with custom colors
    fig.add_trace(go.Pie(labels=white_labels, values=white_counts, marker=dict(colors=white_colors)), row=1, col=2)
    
    # Count empty and non-empty values in the "Date of Thrombosis" column
    date_of_thrombosis_counts = df['Date of Thrombosis'].isnull().value_counts()
    date_of_thrombosis_labels = ['No event', 'Event']
    # Create a pie chart for "Date of Thrombosis" with custom colors
    fig.add_trace(go.Pie(labels=date_of_thrombosis_labels, values=date_of_thrombosis_counts, marker=dict(colors=events_colors)), row=1, col=3)

    # BMI histogram
    fig.add_trace(go.Histogram(x=df["BMI"], name="BMI", marker=dict(color=bmi_histogram_color)), row=2, col=1)

    # Age histogram
    fig.add_trace(go.Histogram(x=df["Age"], name="Age", marker=dict(color=age_histogram_color)), row=2, col=2)

    # Create a summary table
    unique_patients = df['Record ID'].nunique()
    total_data_points = len(df)

    data_summary = pd.DataFrame({
        'Category': ['Unique Patients', 'Total Data Points'],
        'Count': [unique_patients, total_data_points]
    })

    trace = go.Table(
        header=dict(values=["Category", "Count"]),
        cells=dict(values=[data_summary['Category'], data_summary['Count']])
    )

    fig.add_trace(trace, row=2, col=3)

    # Update layout
    fig.update_layout(title=dict(text="Demographics charts", 
                                 font=dict(family="Source Sans Pro, light" ,size=25), automargin=True))
    return fig

def describe_dataframe(df):
    """
    Perform customized statistical analysis on columns of a pandas DataFrame.

    Parameters:
        - df (pandas.DataFrame): The DataFrame to be analyzed.

    Returns:
        - numerical_and_dates_analysis: A DataFrame containing statistical analysis of numerical values.
        - categorical_analysis: A DataFrame containing statistical analysis of categorical values.
    """
    # Exclude the "Record ID" column
    if 'Record ID' in df.columns:
        df = df.drop(columns=['Record ID'])

    # Separate columns into numerical and categorical
    numerical_and_dates = df.select_dtypes(include=['number', 'datetime'])
    categorical = df.select_dtypes(exclude=['number', 'datetime'])

    # Treat binary columns as categorical
    binary_columns = [col for col in numerical_and_dates.columns if len(df[col].unique()) == 2]
    if binary_columns:
        categorical = pd.concat([categorical, df[binary_columns]], axis=1)
        numerical_and_dates = numerical_and_dates.drop(columns=binary_columns)

    # Describe for numerical and date columns
    numerical_and_dates_analysis = numerical_and_dates.describe()

    # Describe for categorical columns
    categorical_analysis = pd.DataFrame()
    for col in categorical.columns:
        categorical_analysis[col] = [
            categorical[col].count(),
            categorical[col].nunique(),
            categorical[col].mode().values[0] if len(categorical[col].mode()) > 0 else None,
            categorical[col].mode().count() if len(categorical[col].mode()) > 0 else None,
            categorical[col].value_counts().index.tolist()  # Show top 5 values
        ]

    # Rename the rows for categorical analysis
    categorical_analysis.index = ['Count', 'Unique', 'Top', 'Freq', 'Values']

    return numerical_and_dates_analysis, categorical_analysis


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

def feature_importance (pipeline, X_train):
    """
    Calculate feature importance for an XGBoost model within a given pipeline.

    Parameters:
        pipeline (sklearn.pipeline.Pipeline): A scikit-learn pipeline containing an XGBoost classifier.
        X_train (pd.DataFrame): The training data used to train the model.

    Returns:
        pd.DataFrame: A DataFrame containing feature names, importance scores, and their percentage contributions.

    This function calculates feature importance for an XGBoost model within a given scikit-learn pipeline.
    It retrieves the feature importances, feature names, and computes the percentage contribution of each feature to the model's predictions.

    The returned DataFrame is sorted in descending order of importance, making it easy to identify the most influential features in the model.
    """
    
    # Get feature importances from the XGBoost model in the pipeline
    importances = pipeline.named_steps['classifier'].feature_importances_

    # Get the feature names from the original DataFrame
    feature_names = X_train.columns

    # Calculate the total importance
    total_importance = importances.sum()

    # Calculate the percentage of contribution for each feature
    percentage_contributions = (importances / total_importance) * 100

    # Create a DataFrame to store feature names, importance scores, and percentage contributions
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances, 'Percentage Contribution': percentage_contributions})

    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # return the DataFrame with feature names, importance scores, and percentage contributions
    return feature_importance_df