import pandas as pd

def preprocess(df):
    # Dummy encoding
    df_encoded = pd.get_dummies(df, columns=['Artery affected',
                                             "Extremity",
                                             'Anticoagulation',
                                             'Intervention Classification'],    
                            prefix=['Artery affected','Extremity',
                                    'Anticoagulation','Intervention Classification'])
    
    output_file = './data/Preprocessed_Data.xlsx'
    df_encoded.to_excel(output_file, index=False)

data_path = "./data/DummyData_Extended.xslxs"
df = pd.read_excel(data_path)