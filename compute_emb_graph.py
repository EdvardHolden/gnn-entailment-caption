# TODO - check how we can call graph!
from parser import graph

problem_a = b"fof(c1,axiom, ((a & b) | ~c))."
problem_b = b"fof(c1,axiom, ~((a & b) | c))."

problems = [problem_a, problem_b]


def read(prob_text):
    nodes, sources, targets, premise_indices = graph(prob_text, tuple())

    # TODO return as Data later
    print(nodes)
    print(len(set(nodes)))
    print(sources)
    print(targets)
    print(premise_indices)

    return nodes, sources, targets, premise_indices


def main():
    for prob in problems:
        print()
        read(prob)


if __name__ == "__main__":
    main()
