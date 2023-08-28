import threading
import time

from plot import Plot3D
import multiprocessing as mp

p = Plot3D('final.npy', [0.5, 0.75, 1],
           ['Material A', 'Material B', 'Material C'], ['r', 'b', 'g'])
#
# def update_plot(command_queue):
#     global p
#     while True:
#         command = command_queue.get()
#         print(command)
#         if command == 'init':
#             p = Plot3D('final.npy', [0.5, 0.75, 1],
#                        ['Material A', 'Material B', 'Material C'], ['r', 'b', 'g'])
#             p.init_figure()
#         if command == 'update':
#             p.update_figure('final.npy')
#         elif command == 'exit':
#             break
#
#
# if __name__ == '__main__':
#     update_queue = mp.Queue()
#     update_process = mp.Process(target=update_plot, daemon=True, args=(update_queue,))
#     update_process.start()
#
#     update_queue.put('init')
#     time.sleep(5.0)
#     # update_queue.put('update')
#     p.update_figure('final.npy')
#     time.sleep(30.0)
