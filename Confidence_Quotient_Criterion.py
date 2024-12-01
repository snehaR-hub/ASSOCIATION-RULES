#To apply the Confidence Quotient Criterion in Python, you can calculate the Confidence Quotient (CQ) for association rules generated from a dataset. The Confidence Quotient (CQ) compares the confidence of a rule A→B with the confidence of its reverse B→A. It is defined as:

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
# Pass num_itemsets to association_rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=num_itemsets)  

# Step 4: Calculate Confidence Quotient (CQ) for each rule
def confidence_quotient(rules):
    # Create a list to store the confidence quotient for each rule
    cq_list = []
    
    for _, row in rules.iterrows():
        # Rule A -> B
        confidence_ab = row['confidence']
        
        # Reverse rule B -> A
        antecedent = frozenset(row['antecedents'])
        consequent = frozenset(row['consequents'])
        
        # Find reverse rule B -> A
        reverse_rule = rules[(rules['antecedents'] == consequent) & (rules['consequents'] == antecedent)]
        
        # If the reverse rule exists, get its confidence; otherwise assume it's 0
        if not reverse_rule.empty:
            confidence_ba = reverse_rule['confidence'].values[0]
        else:
            confidence_ba = 0
        
        # Calculate Confidence Quotient: CQ(A -> B) = confidence(A -> B) / confidence(B -> A)
        if confidence_ba > 0:
            cq = confidence_ab / confidence_ba
        else:
            cq = float('inf')  # If confidence_ba is 0, set CQ to infinity
        
        cq_list.append(cq)
    
    # Add the Confidence Quotient as a new column to the rules DataFrame
    rules['confidence_quotient'] = cq_list
    
    return rules

# Apply the Confidence Quotient Criterion
rules_with_cq = confidence_quotient(rules)

# Display the rules with their confidence quotient
print(rules_with_cq[['antecedents', 'consequents', 'confidence', 'confidence_quotient']])    
