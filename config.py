from pathlib import Path

# export CUDA_VISIBLE_DEVICES=""
# device = 'cuda'
device = "cpu"

BENCHMARK_PATHS = {"deepmath": Path(__file__).parent / "nndata"}
BATCH_SIZE = 64
EPOCHS = 80

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
