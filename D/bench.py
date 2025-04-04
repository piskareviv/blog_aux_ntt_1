from os import path
system = bench = lambda x: None
L = R = None
exec(open(path.join("..", "bench.py")).read())
# not right, but works


system(f"g++ -I./ ../run.cpp -O2 -std=c++20 -o run")
bench("run", "data_ntt.txt", range(L, R))

system(f"g++ -I./ ../run.cpp -O3 -std=c++20 -o run")
bench("run", "data_ntt_O3.txt", range(L, R))

system(f"g++ -I./ ../run.cpp -O2 -std=c++20 -DSKIP_LOW_UPD -o run")
bench("run", "data_ntt_sl.txt", range(L, R))

system(f"g++ -I./ ../run.cpp -O2 -std=c++20 -DSKIP_TOP_UPD -o run")
bench("run", "data_ntt_st.txt", range(L, R))

system(f"rm run")
system(f"rm tmp.txt")
