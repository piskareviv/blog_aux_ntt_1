from os import path
system = bench = lambda x: None
L = R = None
exec(open(path.join("..", "bench.py")).read())
# not right, but works


system(f"g++ -I./ ../run.cpp -O2 -std=c++20 -o run")
bench("run", "data_ntt.txt", range(L, R))

system(f"g++ -I./ ../run.cpp -O2 -std=c++20 -DONLY_BIT_REVERSE -o run")
bench("run", "data_bit_reverse.txt", range(L, R))

system(f"g++ -I./ ../run.cpp -O2 -std=c++20 -DDO_NOTHING -o run")
bench("run", "data_nothing.txt", range(L, R))


system(f"rm tmp.txt")
