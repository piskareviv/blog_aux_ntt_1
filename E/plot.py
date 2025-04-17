from os import path
make_plot = get_data = plot = lambda x: None
exec(open(path.join("..", "plot.py")).read())


make_plot([
    ("data_ntt.txt", "E"),
    ("../D/data_ntt.txt", "D"),
    ("../A/data_bit_reverse.txt", "bit reverse"),
], "plot", y_ticks=2)

make_plot([
    ("data_ntt.txt", "E"),
    ("data_ntt_O3.txt", "E O3"),

], "plot1")
