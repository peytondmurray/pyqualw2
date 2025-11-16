import pandas as pd




config_file_path = '../Model/W2_con.csv'

config_DF = pd.read_csv(config_file_path,index_col=0)

print(config_DF.head())