from os import path
make_plot = get_data = plot = lambda x: None
exec(open(path.join("..", "plot.py")).read())


make_plot([
    ("data_ntt.txt", "C"),
    ("../A3/data_ntt.txt", "A3"),
    ("../B/data_ntt.txt", "B"),
    ("../B/data_ntt_O3.txt", "B O3"),
    ("../A/data_bit_reverse.txt", "bit reverse"),
], "plot", large_y_ticks=True)

make_plot([
    ("data_ntt.txt", "C"),
    ("data_ntt_O3.txt", "C O3"),
    ("../A/data_bit_reverse.txt", "bit reverse"),
], "plot1")
