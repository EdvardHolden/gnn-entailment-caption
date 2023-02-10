import resource
import sys
from pathlib import Path

# Increase u limit
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

PYTHON = sys.executable

# export CUDA_VISIBLE_DEVICES=""
# device = 'cuda'
device = "cpu"

BENCHMARK_PATHS = {"deepmath": Path(__file__).parent / "nndata"}
BATCH_SIZE = 64

EPOCHS = 80
ES_PATIENCE = 5

TRAIN_ID = "id_files/train.txt"
VAL_ID = "id_files/validation.txt"
TEST_ID = "id_files/test.txt"

PROBLEM_DIR = str((Path(__file__).parent / "nndata").absolute())

NODE_TYPE = {
    0: "True",
    1: "False",
    2: "Variable",
    3: "Functor",
    4: "Argument",
    5: "Application",
    6: "Equality",
    7: "Negation",
    8: "And",
    9: "Or",
    10: "Equivalent",
    11: "Forall",
    12: "Exists",
    13: "Axiom",
    14: "Conjecture",
}


dpath = Path.home() / "gnn-entailment-caption"
