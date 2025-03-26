from os import system
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("l", nargs="?", type=int, default=3)
parser.add_argument("r", nargs="?", type=int, default=26)
args = parser.parse_args()
L, R = args.l, args.r


def bench(program, filename, rng):
    print(program, filename, rng, flush=True)
    with open(filename, 'w') as f:
        for i in rng:
            cnt = max(5, int(2e7 / 2**i))
            system(f"taskset -c 3 ./run {i} {cnt} > tmp.txt")
            tm = float(open("tmp.txt", 'r').read())
            print(i, f"{tm / cnt:.20f}", file=f, flush=True)
            print(f"{tm / cnt:.20f}", "  ", i, cnt, tm, flush=True)
    print()
