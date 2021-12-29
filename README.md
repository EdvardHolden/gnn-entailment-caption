# gnn-entailment-caption - Generating problem graph features for captioning task

This is a work in progress.

To run: 
export CUDA_VISIBLE_DEVICES=""

## Pre-train model
To train the initial model on the deepmath dataset run `python3 train.py`


## Compute problem graphs
To compute the graph embeddings we use the 'LTBDataset' class.
We give this class a txt file containing the problem names (no paths, just the file names), and the path to the directory containing the problems.
Comptuing the graphs will take less than 10 minutes for about 7K problems.
There will be a tgraph datapoint (*.pt file) created for each problem which will be stored in 'graph_data/{name_of_id_file}/processed/'.


## Compute problem embeddings

To compute the problem embeddings we run the script 'extract features' which computes the feature vector for each problem.
The set of feature vectors are stored in 'graph_features.pkl'. This might have to be changed for very large datasets.


## Future
Add unsupervised graph training to the mix.
Might also create an option which uses only the conjecture nodes in the embedding.
