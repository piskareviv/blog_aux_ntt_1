from os import path
import numpy as np
from matplotlib import pyplot as plt


def get_data(file, scale=1.0):
    raw = open(file, 'r').readlines()
    data = list(map(lambda x: list(map(float, x.split())), raw))
    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)
    y /= 2**x
    y *= scale
    return x, y * 1e9


def get_name(file):
    if isinstance(file, list | tuple):
        file, name = file
    else:
        name = file
    return file, name


def plot(file, scale=1.0, **kwargs):
    file, name = get_name(file)

    x, y = get_data(file, scale)
    plt.plot(x, y, label=name)

    if "draw_line" in kwargs and file in kwargs["draw_line"]:
        plt.axline((x[9], y[9]), (x[15], y[15]), color="red", linestyle="--")


def make_plot(files, out_file, show=False, large_y_ticks=False, **kwargs):
    my_dpi = 200
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.yticks(np.arange(0, 201, 10) if large_y_ticks else np.arange(0, 101, 1))

    plt.xticks(np.arange(0, 31, 1))
    plt.grid(linestyle="--")
    plt.axvline(x=13, linestyle="--")
    plt.axvline(x=18, linestyle="--")
    plt.axvline(x=21, linestyle="--")
    plt.axvline(x=0)
    plt.axhline(y=0)

    for fl in files:
        plot(fl, **kwargs)

    plt.legend()
    if show:
        plt.show()

    plt.ylabel("ns per element")
    plt.xlabel("log_2 n")

    plt.savefig(f"{out_file}.svg")
