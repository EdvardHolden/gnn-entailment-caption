python3 -u search_hyperparams.py --parameter_space experiments/thesis/graph_depth/params.json --learning_task premise --experiment_dir experiments/thesis/graph_direction/premise
python3 -u search_hyperparams.py --parameter_space experiments/thesis/graph_depth/params.json --learning_task premise --experiment_dir experiments/thesis/graph_direction/premise_rm --graph_remove_argument_node
python3 -u search_hyperparams.py --parameter_space experiments/thesis/graph_depth/params.json --learning_task similarity --experiment_dir experiments/thesis/graph_direction/similarity
python3 -u search_hyperparams.py --parameter_space experiments/thesis/graph_depth/params.json --learning_task similarity --experiment_dir experiments/thesis/graph_direction/similarity_rm --graph_remove_argument_node

#python3 synthesise_results.py experiments/thesis/
