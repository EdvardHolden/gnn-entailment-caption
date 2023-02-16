python3 -u search_hyperparams.py --parameter_space experiments/thesis/graph_direction/params.json --learning_task premise --experiment_dir experiments/thesis/graph_direction/premise
python3 -u search_hyperparams.py --parameter_space experiments/thesis/graph_direction/params.json --learning_task similarity --experiment_dir experiments/thesis/graph_direction/similarity

python3 synthesise_results.py experiments/thesis/graph_direction/
