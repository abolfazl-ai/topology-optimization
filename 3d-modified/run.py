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


def iter_print(loop, x, x_change, c_change, compliance, weight, save_path, plotter):
    title = (f"Iteration {str(loop).rjust(3, '0')}: XChange={x_change:0.6f}, "
             f"CChange={c_change:0.6f}, Compliance={compliance:0.3e}, Weight={weight * 100:0.1f}%")
    print(title)
    np.save(save_path + '.npy', x)
    if plotter is not None: plotter.update(save_path + '.npy', title=title, interactive=True)


def single_run(input_path, save_path, plot=True):
    mesh, bc, filter_params, opt_params, materials, pres, mask = get_input(input_path)
    np.save(save_path + '.npy', np.moveaxis(pres, -1, 0))
    plotter = Plotter(save_path + '.npy', materials, zoom=0.9) if plot else None

    start = time.time()
    print('Optimization starting. Output will be saved in ' + save_path + '.npy')
    print(f'Mesh={mesh["shape"]}, Filter={filter_params["filter"]}, '
          f'FilterBC={filter_params["filter_bc"].upper()}, Rmin={filter_params["radius"]:0.3f}')

    optimizer = TopOpt(mesh, bc, filter_params, opt_params, materials, pres, mask,
                       lambda loop, x, x_ch, c_ch, c, w:
                       iter_print(loop, x, x_ch, c_ch, c, w, save_path, plotter))
    optimizer.optimize()

    title = f'Model converged in {(time.time() - start):0.2f} seconds.'
    print(title)
    if plotter is not None: plotter.update(save_path + '.npy', title=title, interactive=False)


single_run('input.xlsx', '3d')
# save_data('runs/data', 'shear-concentrated-sm', cc, vv)
