from os import path
make_plot = get_data = plot = lambda x: None
exec(open(path.join("..", "plot.py")).read())


make_plot([
    ("data_ntt.txt", "B"),
    ("../A3/data_ntt.txt", "A3"),
    ("../A2/data_ntt.txt", "A2"),
    ("../A/data_ntt.txt", "A"),
    ("../A/data_bit_reverse.txt", "bit reverse"),
    ("data_ntt_O3.txt", "B O3"),
], "plot", large_y_ticks=True)
