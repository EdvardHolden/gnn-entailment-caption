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


## Compute Supervised problem embeddings

To compute the problem embeddings we run the script 'extract_features.py' which computes the feature vector for each problem, with the help of the model in 'model.pt'
The set of feature vectors are stored in 'graph_features_*.pkl'. This might have to be changed for very large datasets.



## Compute Unsupervised problem embeddings

To compute unsupervised embeddings, run the script 'embed_problems_unsupervised.py'.
It may use pre computed datasets and model, depending on the parameters given and what is available.
Will automatically compute embeddings for all, premise and conjecture pooling configurations.
the result is saved as 'embedding_unsupervised_*.pkl'
