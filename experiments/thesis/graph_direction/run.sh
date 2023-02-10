python3 -u train.py --learning_task premise --experiment_dir experiments/thesis/graph_direction/premise
python3 -u train.py --learning_task premise --experiment_dir experiments/thesis/graph_direction/premise --graph_bidirectional
python3 -u train.py --learning_task similarity --experiment_dir experiments/thesis/graph_direction/similarity
python3 -u train.py --learning_task similarity --experiment_dir experiments/thesis/graph_direction/similarityn --graph_bidirectional

python3 synthesise_results.py experiments/thesis/
