import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_json("evaluation_results.json")

# Count occurrences of individual features
feature_counts = {}
for row in df['selected_features']:
    for feature in row:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1

# Get Most successful model
best_model = df.iloc[df['score_sequential'].idxmax()]

# Plot feature occurrences
plt.bar(feature_counts.keys(), feature_counts.values())
plt.xlabel('Feature')
plt.ylabel('Occurrences')
plt.title('Occurrences of Features')
plt.show()
