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


def iter_print(loop, x, x_ch, c_ch, c, w, save_path, plotter):
    print(f'{loop:03}{"":13}{x_ch:<16.6f}{c_ch:<16.6f}{c:<16.4E}{100 * w:<0.2f}%')
    np.save(save_path + '.npy', x)
    if plotter is not None: plotter.update(save_path + '.npy', f'Iteration {loop:03}: C={c:0.4e}', True)


def run(input_path, save_path, plot=True):
    mesh, bc, filter_params, opt_params, materials, pres, mask = get_input(input_path)
    np.save(save_path + '.npy', pres)
    plotter = Plotter(save_path + '.npy', materials, zoom=0.9) if plot else None

    start = time.time()
    print('Optimization starting. Mesh={}, Filter={}, FilterBC={}, R={:0.3f}'.
          format(mesh["shape"], *[filter_params[key] for key in ('filter', 'filter_bc', 'radius')]))
    print(('{:<16}' * 5).format("Iteration", "X Change", "C Change", "Compliance", "Weight"))

    optimizer = TopOpt(mesh, bc, filter_params, opt_params, materials, pres, mask,
                       lambda loop, x, x_ch, c_ch, c, w:
                       iter_print(loop, x, x_ch, c_ch, c, w, save_path, plotter))
    optimizer.optimize()

    print(f'Model converged in {(time.time() - start):0.2f} seconds.')
    if plotter is not None: plotter.update(save_path + '.npy', interactive=False)


run('input_3d.xlsx', '3d')
# save_data('runs/data', 'shear-concentrated-sm', cc, vv)
