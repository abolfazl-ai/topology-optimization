import numpy as np
import matplotlib.pyplot as plt

from main.generate_input import generate_input

nan = (np.nan, np.nan)

bcs = {
    '01': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}],
    '02': [{'S': (0.5, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}],
    '03': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}, {'S': (0, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '04': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}, {'S': (0, 0), 'E': (0, 1), 'D': (0, 0), 'F': nan}],
    '05': [{'S': (0.25, 0), 'E': (0.75, 0), 'D': (0, 0), 'F': nan}],
    '06': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}, {'S': (1, 0.5), 'E': (1, 0.5), 'D': (0, 0), 'F': nan}],
    '07': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}, {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '08': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}, {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': nan}],
    '09': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': nan},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '10': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}, {'S': (0.5, 1), 'E': (0.5, 1), 'D': (0, 0), 'F': nan}],
    '11': [{'S': (0, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': nan}, {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '12': [{'S': (0, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': nan}, {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': nan}],
    '13': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan}, {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}],
    '14': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan}, {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '15': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan}, {'S': (0.5, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': nan}],
    '16': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '17': [{'S': (0, 0.5), 'E': (0, 0.5), 'D': (0, 0), 'F': nan},
           {'S': (1, 0.5), 'E': (1, 0.5), 'D': (0, 0), 'F': nan}],
    '18': [{'S': (0.5, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': nan}, {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '19': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan},
           {'S': (0.5, 1), 'E': (0.5, 1), 'D': (0, 0), 'F': nan}],
    '20': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': nan},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '21': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': nan},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '22': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': nan},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan}],
    '23': [{'S': (0, 0.5), 'E': (0, 0.5), 'D': (0, 0), 'F': nan},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': nan},
           {'S': (1, 0.5), 'E': (1, 0.5), 'D': (0, 0), 'F': nan}],
    '24': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': nan},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': nan},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
    '25': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': nan},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': nan},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': nan},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': nan}],
}

forces = {
    'A': [{'S': (0, 1), 'E': (1, 1), 'D': nan, 'F': (0, -1)}],
    'B': [{'S': (0, 1), 'E': (0.5, 1), 'D': nan, 'F': (0, lambda x: -2 * x)},
          {'S': (0.5, 1), 'E': (1, 1), 'D': nan, 'F': (0, lambda x: -2 * (1 - x))}],
    'C': [{'S': (0, 1), 'E': (1, 1), 'D': nan, 'F': (0, lambda x: -2 * x)}],
    'D': [{'S': (0.5, 1), 'E': (1, 1), 'D': nan, 'F': (0, -1)}],
    'E': [{'S': (0.5, 1), 'E': (1, 1), 'D': nan, 'F': (0, lambda x: -2 * (x - 0.5))}],
    'F': [{'S': (1, 1), 'E': (1, 1), 'D': nan, 'F': (0, -1)}],
    'G': [{'S': (0.5, 1), 'E': (0.5, 1), 'D': nan, 'F': (0, -1)}],
    'H': [{'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': nan, 'F': (0, -1)}],
    'I': [{'S': (1, 0.5), 'E': (1, 0.5), 'D': nan, 'F': (0, -1)}],
    'J': [{'S': (0, 1), 'E': (0, 1), 'D': nan, 'F': (0, -1)}, {'S': (1, 1), 'E': (1, 1), 'D': nan, 'F': (0, -1)}],
    'K': [{'S': (0, 1), 'E': (1, 1), 'D': nan, 'F': (1, 0)}],
    'L': [{'S': (0, 1), 'E': (0.5, 1), 'D': nan, 'F': (lambda x: 2 * x, 0)},
          {'S': (0.5, 1), 'E': (1, 1), 'D': nan, 'F': (lambda x: 2 * (1 - x), 0)}],
    'M': [{'S': (0, 1), 'E': (1, 1), 'D': nan, 'F': (lambda x: 2 * x, 0)}],
    'N': [{'S': (0.5, 1), 'E': (1, 1), 'D': nan, 'F': (1, 0)}],
    'O': [{'S': (0.5, 1), 'E': (1, 1), 'D': nan, 'F': (lambda x: 2 * (x - 0.5), 0)}],
    'P': [{'S': (1, 1), 'E': (1, 1), 'D': nan, 'F': (1, 0)}],
    'Q': [{'S': (0.5, 1), 'E': (0.5, 1), 'D': nan, 'F': (1, 0)}],
    'R': [{'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': nan, 'F': (1, 0)}],
    'S': [{'S': (1, 0.5), 'E': (1, 0.5), 'D': nan, 'F': (1, 0)}],
    'T': [{'S': (0, 1), 'E': (0, 1), 'D': nan, 'F': (1, 0)}, {'S': (1, 1), 'E': (1, 1), 'D': nan, 'F': (1, 0)}],
    'U': [{'S': (1, 1), 'E': (1, 1), 'D': nan, 'F': (1, -1)}],
    'V': [{'S': (0.5, 1), 'E': (0.5, 1), 'D': nan, 'F': (1, -1)}],
    'W': [{'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': nan, 'F': (1, -1)}],
    'X': [{'S': (1, 0.5), 'E': (1, 0.5), 'D': nan, 'F': (1, -1)}],
    'Y': [{'S': (0, 1), 'E': (0, 1), 'D': nan, 'F': (1, -1)}, {'S': (1, 1), 'E': (1, 1), 'D': nan, 'F': (1, -1)}],
}

bou = [
    [
        {'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': nan},
        {'S': (0, 1), 'E': (1, 1), 'D': (np.nan, np.nan), 'F': (0, -1)}
    ]
]

import time
import numpy as np
import pandas as pd
from TopOpt import TopOpt
from read_input import get_input
from plot import Plotter


def save_data(save_path, column_name, compliance, volume):
    temp = np.zeros(shape=(50,)) * np.nan
    sheets = ['C', 'V']
    dfs = [pd.read_excel(save_path + '.xlsx', sheet_name=s) for s in sheets]
    for df, sheet_name in zip(dfs, sheets):
        temp[0:len(compliance)] = [compliance, volume][sheets.index(sheet_name)]
        df[column_name] = temp
    with pd.ExcelWriter(save_path + '.xlsx', engine='openpyxl') as writer:
        [df.to_excel(writer, sheet_name=s, index=False) for df, s in zip(dfs, sheets)]


def plot():
    fig, ax = plt.subplots(1, 2)


def run():
    mesh, bc, filter_params, opt_params, materials, pres, mask = generate_input(2, 3, 0.2, bou[0])

    optimizer = TopOpt(mesh, bc, filter_params, opt_params, materials, pres, mask,
                       lambda loop, x, x_ch, c_ch, c, w:
                       print(f'{loop:03}{"":13}{x_ch:<16.6f}{c_ch:<16.6f}{c:<16.4E}{100 * w:<0.2f}%'))
    optimizer.optimize()


run()
