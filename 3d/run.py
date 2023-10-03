import time

import numpy as np
import pandas as pd
from read_input import read_pres, read_options, read_materials, read_bc
from top3d_mm import top3d_mm
from plot import Plot3D


def save_data(save_path, column_name, compliance, volume):
    temp = np.zeros(shape=(50,)) * np.nan
    sheets = ['C', 'V']
    dfs = [pd.read_excel(save_path + '.xlsx', sheet_name=s) for s in sheets]
    for df, sheet_name in zip(dfs, sheets):
        temp[0:len(compliance)] = [compliance, volume][sheets.index(sheet_name)]
        df[column_name] = temp
    with pd.ExcelWriter(save_path + '.xlsx', engine='openpyxl') as writer:
        [df.to_excel(writer, sheet_name=s, index=False) for df, s in zip(dfs, sheets)]


def iter_print(loop, x_phys, x_change, c_change, compliance, volume, save_path, plotter=None):
    print(f"Design cycle {str(loop).rjust(3, '0')}: XChange={x_change:0.6f}, "
          f"CChange={c_change:0.6f}, Compliance={compliance:0.3e}, Weight={volume * 100:0.1f}%")
    np.save(save_path, np.moveaxis(x_phys, -1, 0))
    if plotter is not None: plotter.update(save_path, interactive=True)


def single_run(input_path, save_path, options=None, plot=True):
    nx, ny, nz, vf, penalty, ft, filter_bc, max_it, x_con, c_con, r_min, eta, beta, move = read_options(input_path)
    if options is not None: nx, ny, nz, ft, filter_bc, r_min = options
    node_numbers = np.reshape(range((1 + nx) * (1 + ny) * (1 + nz)), (1 + ny, 1 + nz, 1 + nx), order='F')
    pres, mask = read_pres(input_path, nx, ny, nz)
    free, force = read_bc(input_path, nx, ny, nz, node_numbers)
    densities, elasticities, names, colors = read_materials(input_path)
    # [iteration_start, final, iteration_steps, delta]
    penal_cnt, beta_cnt, move_cnt = [5, 3, 1, 0.5], [5, 2, 1, 2], [5, 0.1, 3, -0.01]

    np.save(save_path + '.npy', np.moveaxis(pres, -1, 0))
    plotter = Plot3D(save_path + '.npy', densities, names, colors, zoom=0.9) if plot else None

    start = time.time()
    print('Optimization starting. Output will be saved in ' + save_path + '.npy')
    print(f'Mesh={nx}x{ny}x{nz}, Filter={ft}, FilterBC={filter_bc.upper()}, Rmin={r_min:0.3f}')
    x, com, vol = top3d_mm(nx, ny, nz, node_numbers, vf,
                           free, force, pres, mask,
                           densities, elasticities,
                           ft, filter_bc, r_min, eta, beta,
                           max_it, x_con, c_con, move, penalty, penal_cnt, beta_cnt, move_cnt,
                           lambda l, xx, x_ch, c_ch, c, v:
                           iter_print(l, xx, x_ch, c_ch, c, v, save_path + '.npy', plotter))
    print(f'Model converged in {(time.time() - start):0.2f} seconds. Final compliance = {com[-1]}')
    if plotter is not None: plotter.update(save_path + '.npy', interactive=False)
    return x, com, vol


def auto_run(output_path):
    # (nx, ny, nz, filter, filter_bc, r_min)
    modes = [
        (30, 30, 30, 3, 'constant', 3.00),
        (30, 30, 30, 3, 'reflect', 3.00),
        (30, 30, 30, 3, 'constant', 3.46),
        (30, 30, 30, 3, 'reflect', 3.46),
    ]

    for mode in modes:
        name_format = f'{mode[0]}x{mode[1]}x{mode[2]}-{mode[3]}-{mode[4].upper()[0]}-{mode[5]:0.2f}'
        save_path = output_path + '/' + name_format
        x, compliance, volume = single_run('input.xlsx', save_path, options=mode, plot=False)
        save_data(output_path + '/data', name_format, compliance, volume)


_, cc, vv = single_run('runs/press-concentrated.xlsx', 'runs/press-concentrated')
save_data('runs/data', 'press-concentrated', cc, vv)

# auto_run('runs')
