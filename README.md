# Investigating Annotator Bias with a Graph-Based Approach

0. Install requirements.txt
1. Place the files attack_annotated_comments.tsv, attack_annotations.tsv, and attack_worker_demographics.tsv from the Wikipedia Detox Project in folder 00_data

2. Run 20_graph-generation.ipynb to generate the graphs with teh different weight functions
2. Run 21_cluster-and-split.ipynb to generate the data sets (train/test split) for the different groups and weight functions
3. Run 24_distillbert_classifier.ipynb to train the different classifiers

Outputs can be found in 03_results