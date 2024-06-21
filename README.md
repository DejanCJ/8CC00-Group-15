---

## Project Overview

We created a "check_all_thresholds" file to identify optimal thresholds for model optimization. 
After determining the best thresholds, we developed a pipeline 2.0 that runs the entire model using these optimized thresholds,
ensuring improved performance. The other files are part of the data analysis conducted throughout the project.

Feature Importance Ranking and RFE random forest are the two models, which select the best descriptors from the filter_molecules.csv for the data analysis. filtered_molecules results from the pre-process filter.ipynb. PCA and K-means descriptor importance determinance was also tried, but not used. The RFE_top_descriptors and output are the top descriptors based on the two descriptor selecting programs. 

descriptors_and_fingerprints.xlsx are all the unfiltered descriptors and fingerprints resulting from the input SMILES strings. 
