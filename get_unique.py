import pandas as pd
import numpy as np

# Replace 'your_excel_file.xlsx' with the path to your Excel file
excel_file = './Predictive Analysis Jul 2023 CLEAN.xlsx'

# Read the Excel file into a Pandas DataFrame
excel_data = pd.ExcelFile(excel_file)

# Initialize a dictionary to collect unique values
unique_values = {}

# Save unique values to a text file
with open('unique_values.txt', 'w') as file:
    # Iterate through each sheet in the Excel file
    for sheet_name in excel_data.sheet_names:
        sheet = excel_data.parse(sheet_name)
        
        # Iterate through each column in the sheet
        for column in sheet.columns:
            values = sheet[column].unique()

            # Initialize variables to store min and max for numeric values
            min_value = float('inf')
            max_value = float('-inf')

            # Initialize a list to store non-numeric values
            non_numeric_values = []

            # Iterate through the values
            for value in values:
                if isinstance(value, (int, float)):
                    # Check if the value is numeric
                    min_value = min(min_value, value)
                    max_value = max(max_value, value)
                else:
                    # Value is non-numeric, add it to the non-numeric list
                    non_numeric_values.append(value)

            # Save it
            file.write(f"{column}:\n")
            file.write(f"{non_numeric_values}\n")
            if min_value != float('inf'):
                file.write(f"min value {min_value}\n")
            if max_value != float('-inf'):
                file.write(f"max value {max_value}\n")
            file.write("\n")
                
