from pathlib import Path
import re
import os

CLAUSE_PATTERN = b"((cnf|fof|tff)\\((.*\n)*?.*\\)\\.$)"


def _get_clauses(problem_dir, problem):
    # Read the problem file
    path = os.path.join(problem_dir, problem.strip())
    with open(path, "rb") as f:
        text = f.read()

    # Extract all axioms and extract the axiom group
    res = re.findall(CLAUSE_PATTERN, text, re.MULTILINE)
    res = [r[0] for r in res]

    # Convert all cnf, tff clauses to fof (will remove type information later)
    for n in range(len(res)):
        res[n] = res[n].replace(b"cnf(", b"fof(")
        res[n] = res[n].replace(b"tff(", b"fof(")

    return res


def _split_clauses(clauses):
    # Filter all types from tff!
    axioms = []
    conjecture = []

    for clause in clauses:
        if b"conjecture" in clause:
            conjecture += [clause]
        elif b"type" in clause:
            pass  # We discard tff types
        else:
            axioms += [clause]

    return conjecture, axioms


def read_problem_tptp(problem_dir, problem):

    # Extract the clauses from the problem
    clauses = _get_clauses(problem_dir, problem)

    # Set first axiom to be the conjecture
    conjecture, axioms = _split_clauses(clauses)

    return conjecture, axioms


def read_problem_deepmath(problem_dir, problem_name):

    with open(os.path.join(problem_dir, problem_name), "rb") as f:

        conjecture = next(f).strip()[2:]

        premises, targets = zip(
            *[
                (
                    line[2:],
                    1.0 if line.startswith(b"+") else 0.0,
                )
                for line in f
            ]
        )

    return [conjecture], premises, targets
