import pandas as pd
import numpy as np


data = pd.read_csv('Data/Features.csv')


attention_weights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


processed_data = pd.DataFrame(columns=data.columns)


for _, row in data.iterrows():

	item_features = row[1:].values


	weighted_features = np.multiply(item_features, attention_weights)


	processed_data = processed_data.append(pd.Series([row[0]] + list(weighted_features), index=processed_data.columns),
										   ignore_index=True)


processed_data.to_csv('attention.csv', index=False)
