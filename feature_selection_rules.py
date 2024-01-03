import pandas as pd
import numpy as np
import pyfpgrowth

# Load training data
feature_data = pd.read_csv('features.csv')
feature_data = feature_data.drop(['Id'], axis=1)

# Binarize the Pawpularity variable, making zero the values below the 75th percentile and one the values above the 75th percentile
feature_data['Pawpularity'] = np.where(feature_data['Pawpularity'] >= 46, 1, 0)

# Specify the features you want to include in the association rule mining
features_to_include = feature_data.columns[0:-2]

# Concatenate the selected features with the Pawpularity variable
assoc_data = pd.concat([feature_data[['Pawpularity']], feature_data[features_to_include]], axis=1)

# Convert DataFrame to a list of transactions
transactions = assoc_data.applymap(str).values.tolist()

# Use FP-Growth to mine frequent itemsets
patterns = pyfpgrowth.find_frequent_patterns(transactions, 0.001)

# Display the top N frequent itemsets
top_n = 10
sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]

print(f"Top {top_n} Frequent Itemsets:")
for pattern, support in sorted_patterns:
    print(f"Pattern: {pattern}, Support: {support}")
