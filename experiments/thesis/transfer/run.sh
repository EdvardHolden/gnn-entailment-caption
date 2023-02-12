python3 train.py --train_id id_files/validation.txt --epochs 10 --experiment_dir experiments/thesis/transfer/to_premise/ --learning_task premise --transfer_model experiments/similarity/test/
python3 train.py --train_id id_files/validation.txt --epochs 10 --experiment_dir experiments/thesis/transfer/to_similarity/ --learning_task similarity --transfer_model experiments/premise/test/
