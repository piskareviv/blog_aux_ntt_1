from os import path
make_plot = get_data = plot = lambda x: None
exec(open(path.join("..", "plot.py")).read())


make_plot([
    ("data_ntt.txt", "A"),
    ("data_bit_reverse.txt", "bit reverse")
], "plot", y_ticks=10)
