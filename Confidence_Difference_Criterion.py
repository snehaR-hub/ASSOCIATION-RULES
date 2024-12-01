#The Confidence Difference Criterion is used in association rule mining to filter and prioritize rules based on the difference in confidence between two rules or between antecedents and consequents.

# Install necessary libraries
!pip install mlxtend pandas

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

# Step 1: Convert the dataset to a format that can be processed by Apriori
encoder = TransactionEncoder()
encoded_data = encoder.fit(dataset).transform(dataset)
encoded_df = pd.DataFrame(encoded_data, columns=encoder.columns_)

# Step 2: Run Apriori to find frequent itemsets with a minimum support of 0.6
frequent_itemsets = apriori(encoded_df, min_support=0.6, use_colnames=True)

# Step 3: Generate association rules with a minimum confidence of 0.7
# Calculate num_itemsets from 'itemsets' column length
num_itemsets = frequent_itemsets['itemsets'].apply(len).values  
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7,num_itemsets=num_itemsets)
# Step 4: Calculate Confidence Difference
# We need to check each rule (antecedent -> consequent) and its reverse (consequent -> antecedent)

def confidence_difference(rules):
    # Create a list to store the confidence difference for each rule
    diff_list = []
    
    for _, row in rules.iterrows():
        # Rule A -> B
        confidence_ab = row['confidence']
        
        # Reverse rule B -> A
        antecedent = frozenset(row['antecedents'])
        consequent = frozenset(row['consequents'])
        
        # Calculate confidence of reverse rule B -> A
        reverse_rule = rules[(rules['antecedents'] == consequent) & (rules['consequents'] == antecedent)]
        
        # Check if reverse rule exists (to avoid error if the reverse rule does not exist in the dataset)
        if not reverse_rule.empty:
            confidence_ba = reverse_rule['confidence'].values[0]
        else:
            confidence_ba = 0  # If no reverse rule exists, assume confidence is 0
        
        # Calculate confidence difference: |confidence(A -> B) - confidence(B -> A)|
        diff = abs(confidence_ab - confidence_ba)
        diff_list.append(diff)
    
    # Add the confidence difference as a new column to the rules DataFrame
    rules['confidence_difference'] = diff_list
    
    return rules

# Apply the Confidence Difference Criterion
rules_with_diff = confidence_difference(rules)

# Display the rules with their confidence difference
print(rules_with_diff[['antecedents', 'consequents', 'confidence', 'confidence_difference']])
