import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def random_forest_pred(path_to_csv, num_features, pred_row):
    df_molecules = pd.read_csv(path_to_csv)
    X = df_molecules.iloc[:, 2:]
    y = df_molecules[pred_row]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_indices = indices[:num_features]
    top_features = X.columns[top_indices]
    X_important = X_scaled[top_features]

    #print(top_features)
    return top_features


top_feature_list = []
for i in range(10):
    top_feat = random_forest_pred('output_descriptors_filtered.csv', 10, 'PKM2_inhibition')
    top_feature_list.extend(top_feat.to_list())

counter = Counter(top_feature_list)
ten_most_common = counter.most_common(10)
print(ten_most_common)

