python3 train.py --experiment_dir experiments/thesis/transfer/to_premise/ --learning_task premise --transfer_model experiments/thesis/graph_depth/similarity/num_convolutional_layers_7_remove_argument_node_True
python3 train.py --experiment_dir experiments/thesis/transfer/to_similarity/ --learning_task similarity --transfer_model experiments/thesis/graph_depth/premise/num_convolutional_layers_7_remove_argument_node_True


python3 train.py --experiment_dir experiments/thesis/transfer/premise_to_premise/ --learning_task premise --transfer_model experiments/thesis/graph_depth/premise/num_convolutional_layers_7_remove_argument_node_True
python3 train.py --experiment_dir experiments/thesis/transfer/similarity_to_similarity/ --learning_task similarity --transfer_model experiments/thesis/graph_depth/similarity/num_convolutional_layers_7_remove_argument_node_True
