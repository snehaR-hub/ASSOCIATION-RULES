#Mining association rules is a fundamental task in data mining that identifies interesting relationships (or patterns) between variables in large datasets. A common approach to mining association rules is using Apriori or FP-Growth algorithms. These algorithms find frequent itemsets (combinations of items that appear together frequently) and then generate association rules based on these itemsets.

'''Steps to Mine Association Rules Using Python
To mine association rules, the typical steps are:

Prepare the data – This usually involves creating a dataset where each row represents a transaction, and columns represent items (or attributes).
Use the Apriori algorithm – This algorithm finds frequent itemsets in the data.
Generate association rules – These rules represent relationships between items, like "if item A is bought, then item B is likely to be bought."
Python Libraries Required:
pandas: For handling the dataset (if your data is in tabular format).
mlxtend: For implementing the Apriori algorithm and generating association rules.
numpy: For numerical operations (often used with pandas).'''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample data: list of transactions
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter']
]

# Convert the dataset to a format that can be processed by Apriori
encoder = TransactionEncoder()
encoded_data = encoder.fit(dataset).transform(dataset)
encoded_df = pd.DataFrame(encoded_data, columns=encoder.columns_)

# Run Apriori to find frequent itemsets with a minimum support of 0.6
frequent_itemsets = apriori(encoded_df, min_support=0.6, use_colnames=True)

# Generate association rules with a minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the rules
print(rules)
