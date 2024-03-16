from TopOpt import TopOpt
from modes import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generate_input import generate_input
import os
import csv


def save_plot(ux, uy, u, c, save_path):
    titles = ['Abs. Displacement X', 'Abs. Displacement Y', 'Absolute Displacement', 'Compliance']

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    for i, (data, title) in enumerate(zip([np.abs(ux), np.abs(uy), u, c], titles)):
        sns.heatmap(data, ax=ax[i // 2, i % 2], square=True, cmap='viridis', cbar=False,
                    robust=True, xticklabels=False, yticklabels=False).invert_yaxis()
        ax[i // 2, i % 2].set_title(title)

    fig.savefig(save_path)


def save_data(save_path, ux, uy, u, c):
    np.save(arr=ux, file=save_path + '-X')
    np.save(arr=uy, file=save_path + '-Y')
    np.save(arr=u, file=save_path + '-U')
    np.save(arr=c, file=save_path + '-C')
    save_plot(ux, uy, np.sqrt(ux ** 2 + uy ** 2), c, save_path + '-inputs.png')
    np.savetxt(save_path + '-inputs.csv', X=['InputsCode,Penalty,Radius,VolFrac'], fmt='%s', delimiter=',')


def run(output_path='data', n=200):
    counter = 0
    for bc_code, bc in bcs.items():
        bc_path = output_path + '/' + bc_code
        if not os.path.exists(bc_path): os.mkdir(bc_path)
        for load_code, load in loads.items():
            load_path = bc_path + '/' + load_code
            if not os.path.exists(load_path): os.mkdir(load_path)
            for penal in ps:
                for r in rs:
                    for vf in vfs:
                        counter += 1
                        code = f'{bc_code}-{load_code}-P{penal}-R{r:0.3f}-V{vf:0.3f}'
                        print(f'Starting run {counter:06}: {code}')
                        print(f'BC={bc_code}, Load={load_code}, P={penal}, R={r:0.3f}, VF={vf:0.3f}')

                        if not os.path.exists(load_path + '/output/' + code + '.npy'):
                            optimizer = TopOpt(*generate_input(r, penal, vf, bc + load, n),
                                               lambda loop, xx, x_ch, c_ch, cc, ww:
                                               print(f'{loop:03}' + ' ' * 13 +
                                                     f'{x_ch:<16.6f}{c_ch:<16.6f}{cc:<16.4E}{100 * ww:<0.2f}%'))

                            if not os.path.exists(load_path + '/output'):
                                init_u = optimizer.fem.solve(np.ones(optimizer.mesh['shape']).flatten(order='F'))
                                ux = init_u[range(0, 2 * (n + 1) * (n + 1), 2)].reshape(((n + 1), (n + 1)), order='F')
                                uy = init_u[range(1, 2 * (n + 1) * (n + 1), 2)].reshape(((n + 1), (n + 1)), order='F')
                                c = np.sum((init_u[optimizer.mesh['c_mat']] @ optimizer.fem.k) *
                                           init_u[optimizer.mesh['c_mat']], 1).reshape((n, n), order='F')

                                save_data(load_path + '/' + bc_code + load_code, ux, uy, np.hypot(ux, uy), c)
                                os.mkdir(load_path + '/output')

                            print(('{:<16}' * 5).format("It.", "X Change", "C Change", "Compliance", "Weight"))
                            optimizer.optimize()
                            output = optimizer.x
                            np.save(arr=output, file=load_path + '/output/' + code)

                            with open(load_path + '/' + bc_code + load_code + '-inputs.csv', 'a', newline='') as file:
                                writer = csv.writer(file, delimiter=',')
                                writer.writerow([code, penal, r, vf])

                            fig = plt.figure(figsize=(8, 8))
                            sns.heatmap(1 - output, vmin=0, vmax=1, square=True, cmap='gray', cbar=False,
                                        xticklabels=False, yticklabels=False).invert_yaxis()
                            plt.title(f'BC={bc_code}, Load={load_code}, P={penal}, R={r:0.3f}, VF={vf:0.3f}')
                            fig.savefig(load_path + '/output/' + code + '.png')
                            plt.close()

                        else:
                            print('Already exists')


run()
