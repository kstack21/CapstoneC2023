import pandas as pd
from faker import Faker
import random

# Load the original Excel file
input_file = './data/DummyData.xlsx'
df = pd.read_excel(input_file)

# Initialize the Faker object for generating fake data
fake = Faker()

# Create a dictionary to store data type information for each column
data_types = {}

# Iterate through the columns and determine their data types
for column in df.columns:
    if df[column].dtype == 'object':
        data_types[column] = 'string'
    elif df[column].dtype == 'int64':
        data_types[column] = 'integer'
    elif df[column].dtype == 'float64':
        data_types[column] = 'float'

# Create a list to store the new data
new_data_rows = []

# Specify the number of new rows to generate
num_new_rows = 150  # You can change this as needed

# Generate new data based on the original data types
for _ in range(num_new_rows):
    new_row = {}
    for column, data_type in data_types.items():
        if data_type == 'string':
            # For string columns, randomly select from the original data
            new_row[column] = random.choice(df[column])
        elif data_type == 'integer':
            # For integer columns, select a random integer within the original range
            min_val = df[column].min()
            max_val = df[column].max()
            new_row[column] = random.randint(min_val, max_val)
        elif data_type == 'float':
            # For float columns, select a random float within the original range
            min_val = df[column].min()
            max_val = df[column].max()
            new_row[column] = random.uniform(min_val, max_val)
    new_data_rows.append(new_row)

# Convert the list of new data rows to a DataFrame
result_df = pd.DataFrame(new_data_rows, columns=df.columns)


# Save the combined data to a new Excel file
output_file = './data/DummyData_Extended.xlsx'
result_df.to_excel(output_file, index=False)

print(f"Generated {num_new_rows} new rows and saved to '{output_file}'.")
