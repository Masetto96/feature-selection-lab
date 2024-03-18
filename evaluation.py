#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_json("evaluation_results.json")

# Flatten the list of selected features
flattened_features = [item for sublist in df['selected_features'] for item in sublist]

# Count occurrences of each feature
feature_counts = pd.Series(flattened_features).value_counts()

# Plotting
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.barplot(x=feature_counts.index, y=feature_counts.values)
plt.xlabel('Selected Features')
plt.ylabel('Count')
plt.title('Count of Each Selected Feature')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# %% 
