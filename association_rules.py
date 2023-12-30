import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load training data
train_data = pd.read_csv('train.csv')
train_data = train_data.drop(['Id'], axis=1)
train_data.head()

bin_train_data = pd.read_csv('features.csv')

# Binarize the Pawpularity variable, making zero the values below the 75th percentile and one the values above the 75th percentile
bin_train_data = train_data.copy()
bin_train_data['Pawpularity'] = np.where(bin_train_data['Pawpularity'] >= 46, 1, 0)

# Specify the features you want to include in the association rule mining
features_to_include = bin_train_data.columns[1:-2]

# Concatenate the selected features with the Pawpularity variable
assoc_data = pd.concat([bin_train_data[['Pawpularity']], bin_train_data[features_to_include]], axis=1)

# Get the frequent itemsets
frequent_itemsets = apriori(assoc_data, min_support=0.01, use_colnames=True)

# Get the association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Sort the rules based on confidence and lift
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# Show the rules that have the Pawpularity variable in the consequent and sort them by confidence and lift
sorted_rules = rules[rules['consequents'].astype(str).str.contains('Pawpularity')].sort_values(['confidence', 'lift'], ascending=[False, False])

# Get the antecedents of the top ten sorted rules and show the frequency of each antecedent
top_ten = sorted_rules.head(10)
top_ten_antecedents = top_ten['antecedents'].astype(str)
top_ten_antecedents = top_ten_antecedents.str.replace('frozenset', '')
top_ten_antecedents = top_ten_antecedents.str.replace('(', '')
top_ten_antecedents = top_ten_antecedents.str.replace(')', '')
top_ten_antecedents = top_ten_antecedents.str.replace('{', '')
top_ten_antecedents = top_ten_antecedents.str.replace('}', '')
top_ten_antecedents = top_ten_antecedents.str.replace("'", '')
top_ten_antecedents = top_ten_antecedents.str.replace(" ", '')
top_ten_antecedents = top_ten_antecedents.str.split(',')
top_ten_antecedents = top_ten_antecedents.explode()
top_ten_antecedents = top_ten_antecedents.value_counts()
top_ten_antecedents = top_ten_antecedents.reset_index()
top_ten_antecedents.columns = ['Feature', 'Frequency']
print(top_ten_antecedents.head(10))  # Display the top ten antecedents