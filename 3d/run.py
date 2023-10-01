import time

import numpy as np
from read_input import read_pres, read_options, read_materials, read_bc
from top3d_mm import top3d_mm
from plot import Plot3D


def iter_print(loop, x_phys, change, compliance, volume, output_path, plotter=None):
    print(f"Design cycle {str(loop).rjust(3, '0')}: Change={change:0.6f}, "
          f"Compliance={compliance:0.3e}, Volume={volume:0.3f}")
    np.save(output_path + '/x.npy', np.moveaxis(x_phys, -1, 0))
    if plotter is not None: plotter.update(output_path + '/x.npy', interactive=True)


def single_run(input_path, output_path):
    nx, ny, nz, vf, penalty, ft, filter_bc, max_it, r_min, eta, beta, move = read_options(input_path)
    node_numbers = np.reshape(range((1 + nx) * (1 + ny) * (1 + nz)), (1 + ny, 1 + nz, 1 + nx), order='F')
    pres, mask = read_pres(input_path, nx, ny, nz)
    free, force = read_bc(input_path, nx, ny, nz, node_numbers)
    densities, elasticities, names, colors = read_materials(input_path)
    penal_cnt, beta_cnt = [50, 3, 25, 0.25], [250, 16, 25, 2]

    np.save(output_path + '/x.npy', np.moveaxis(pres, -1, 0))
    plotter = Plot3D(output_path + '/x.npy', densities, names, colors, zoom=0.9)

    start = time.time()
    print('Optimization starting.')
    print(f'Mesh={nx}x{ny}x{nz}, Filter={ft}, FilterBC={filter_bc.upper()}, Rmin={r_min:0.3f}')
    x_final, compliance, volume = top3d_mm(nx, ny, nz, node_numbers, vf,
                                           free, force, pres, mask,
                                           densities, elasticities,
                                           ft, filter_bc, r_min, eta, beta,
                                           max_it, move, penalty, penal_cnt, beta_cnt,
                                           lambda l, x, ch, c, v: iter_print(l, x, ch, c, v, output_path, plotter))
    print(f'Model converged in {(time.time() - start):0.2f} seconds. Final compliance = {compliance[-1]}')
    np.save(output_path + '/final.npy', np.moveaxis(x_final, -1, 0))
    plotter.update(output_path + '/final.npy', interactive=False)


# arguments = [
#     (40, 3, 'reflect', 1.74),
#     # (40, 3, 'reflect', 1.80),
#     # (40, 3, 'reflect', 2.00),
#     # (40, 3, 'reflect', 3.00),
#     # (40, 3, 'reflect', 5.00),
#     # (40, 3, 'constant', np.sqrt(3)),
#     # (40, 3, 'constant', 1.80),
#     # (40, 3, 'constant', 2.00),
#     # (40, 3, 'constant', 3.00),
#     # (40, 3, 'constant', 5.00),
#     # (40, 3, 'nearest', np.sqrt(3)),
#     # (40, 3, 'mirror', np.sqrt(3)),
#     # (40, 3, 'wrap', np.sqrt(3)),
#     # (40, 1, 'reflect', np.sqrt(3)),
#     # (40, 2, 'reflect', np.sqrt(3)),
# ]
#
# for nn, fil, fil_bc, r in arguments:
#     name_format = f"{nn}-{fil_bc.upper()[0]}-{r:0.2f}-{fil}"
#     _, comp, vol_f, pri_f = top3d_mm(nn, fil, fil_bc, r, 'runs/' + name_format)
#
#     empty = np.zeros(shape=(500,)) * np.nan
#     sheets = ['Compliance', 'Volume', 'Cost']
#     dfs = [pd.read_excel('runs/40/data.xlsx', sheet_name=s) for s in sheets]
#     for df, sheet_name in zip(dfs, sheets):
#         empty[0:len(comp)] = [comp, vol_f, pri_f][sheets.index(sheet_name)]
#         df[name_format] = empty
#     with pd.ExcelWriter('runs/data.xlsx', engine='openpyxl') as writer:
#         [df.to_excel(writer, sheet_name=s, index=False) for df, s in zip(dfs, sheets)]


single_run('input.xlsx', 'output')
