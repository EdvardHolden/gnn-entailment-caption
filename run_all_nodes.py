import subprocess
import argparse
import time
from tqdm import tqdm

# Useful cmd# killall -u holden
# python3 run_all_nodes.py  --delay 3 --node_lbound 3 --nohup_post 'cd gnn-entailment-caption/ ;  python3 -u search_hyperparams.py --experiment_dir experiments/premise/hyperparam/ --parameter_space experiments/premise/hyperparam/params.json '

# TODO if running python - might have to source ./bashrc first due to pyenv not loading on remote like that
# TODO cannot figure out pythoin issue. bashrc is loading and everything .. Just doing HACK
PYTHON_PATH = "/shareddata/homes/holden/.pyenv/shims/python3"

NOHUP_POST = " >> nohup_{0}.out 2>&1  & "  # dev/null makes sure we do not stick around to wait for the process to terminate

BAD_NODES = [2, 8, 13, 14, 20]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=1, help="Delay between each node")
    parser.add_argument("--timeout", type=int, default=30, help="Time out at node")
    parser.add_argument("--node_lbound", type=int, default=2, help="Lower node id")
    parser.add_argument("--node_ubound", type=int, default=45, help="Upper node id")
    parser.add_argument(
        "--nohup_post",
        action="store_true",
        help="Appends nohup stuff to the end of cmd",
    )
    parser.add_argument("cmd", help="The command to run", nargs="+")
    args = parser.parse_args()

    cmd = " ".join(args.cmd)
    if args.nohup_post:
        # if "nohup" not in cmd:
        #    raise ValueError("Forgot to call nohup at front of cmd")
        cmd += NOHUP_POST

    if "python3" in PYTHON_PATH:
        cmd = cmd.replace("python3", PYTHON_PATH)

    print(f'Running cmd: "{cmd}"')

    for i in tqdm(range(args.node_lbound, args.node_ubound)):

        if i in BAD_NODES:
            continue

        node = f"cc{i:02}"

        try:

            if args.nohup_post:
                cmd_run = cmd.format(i)
            else:
                cmd_run = cmd

            proc = subprocess.Popen(
                f"ssh {node} ' {cmd_run} '",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            outs, errs = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            print(f"Timed out at node: {node}")

        if errs != b"":
            print(f"Error on node {node} msg:", errs.decode("utf-8"), outs.decode("utf-8"))
        time.sleep(args.delay)
        print(node, outs)

    print("# Finished")


if __name__ == "__main__":
    main()
