import sys
import subprocess
import argparse
import time

# Useful cmd# killall -u holden


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=1, help="Delay between each node")
    parser.add_argument("--timeout", type=int, default=30, help="Time out at node")
    parser.add_argument("--node_lbound", type=int, default=2, help="Lower node id")
    parser.add_argument("--node_ubound", type=int, default=45, help="Upper node id")
    parser.add_argument("cmd", help="The command to run", nargs="+")
    args = parser.parse_args()

    cmd = " ".join(args.cmd)
    print("Running cmd:", cmd)

    for i in range(args.node_lbound, args.node_ubound):
        node = f"cc{i:02}"

        try:
            proc = subprocess.Popen(
                f"ssh {node} {cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            outs, errs = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            print(f"Timed out at node: {node}")

        if errs != b"":
            print(f"Error on node {node} msg:", errs)
        time.sleep(args.delay)

    print("# Finished")


if __name__ == "__main__":
    main()
