import numpy as np
from matplotlib import pyplot as plt
from os import path

make_plot = get_data = plot = get_name = lambda x: None
exec(open(path.join("..", "plot.py")).read())


make_plot([
    ("data_ntt.txt", "F"),
    ("../E/data_ntt.txt", "E")
], "plot", draw_line="data_ntt.txt",  y_ticks=1)

make_plot([
    ("data_ntt.txt", "F"),
    ("../E/data_ntt.txt", "E"),
    ("../A/data_bit_reverse.txt", "bit reverse"),
], "plot1", y_ticks=2)

make_plot([
    ("data_ntt.txt", "F"),
    ("data_ntt_O3.txt", "F O3"),
    ("../A/data_bit_reverse.txt", "bit reverse")
], "plot2", y_ticks=2)


def make_ratio_plot(file1, file2, out_file):
    file1, name1 = get_name(file1)
    file2, name2 = get_name(file2)

    x1, y1 = get_data(file1)
    x2, y2 = get_data(file2)
    x2, y2 = list(x2), list(y2)

    my_dpi = 200
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.yticks(np.arange(0, 101, 1))

    plt.xticks(np.arange(0, 31, 1))
    plt.grid(linestyle="--")
    plt.axvline(x=13, linestyle="--")
    plt.axvline(x=18, linestyle="--")
    plt.axvline(x=21, linestyle="--")
    plt.axvline(x=0)
    plt.axhline(y=0)

    x, y = [], []
    for i in range(len(x1)):
        if x1[i] in x2:
            j = x2.index(x1[i])
            x += [x1[i]]
            y += [y1[i] / y2[i]]

    plt.plot(x, y, label=f"'{name1}' to '{name2}'  ratio")

    plt.legend()

    plt.ylabel("ratio")
    plt.xlabel("log_2 n")

    plt.savefig(f"{out_file}.svg")


make_ratio_plot(("../A/data_ntt.txt", "A"), ("data_ntt.txt", "F"), "ratio")
