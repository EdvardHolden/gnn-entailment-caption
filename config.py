from pathlib import Path

# export CUDA_VISIBLE_DEVICES=""
# device = 'cuda'
device = "cpu"

BENCHMARK_PATHS = {"deepmath": Path(__file__).parent / "nndata"}
BATCH_SIZE = 64
