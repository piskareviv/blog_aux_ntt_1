from os import path
make_plot = get_data = plot = lambda x: None
exec(open(path.join("..", "plot.py")).read())

make_plot([
    ("data_ntt.txt", "D"),
    ("../C/data_ntt.txt", "C"),
], "plot", y_ticks=2)

make_plot([
    ("data_ntt.txt", "D"),
    ("../C/data_ntt.txt", "C"),
    ("../A/data_bit_reverse.txt", "bit reverse"),
], "plot_br", y_ticks=2)

make_plot([
    ("data_ntt.txt", "D"),
    ("../C/data_ntt.txt", "C"),
    ("data_ntt_sl.txt", "D skip bottom upd"),
    ("data_ntt_st.txt", "D skip top upd"),
], "plot1", y_ticks=2)

make_plot([
    ("data_ntt.txt", "D"),
    ("data_ntt_O3.txt", "D O3"),
], "plot2", y_ticks=1)
