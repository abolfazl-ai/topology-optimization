import numpy as np

bcs = {
    '01': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0}],
    '02': [{'S': (0.5, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0}],
    '03': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': 0}],
    '04': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0, 0), 'E': (0, 1), 'D': (0, 0), 'F': 0}],
    '05': [{'S': (0.25, 0), 'E': (0.75, 0), 'D': (0, 0), 'F': 0}],
    '06': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 0.5), 'E': (1, 0.5), 'D': (0, 0), 'F': 0}],
    '07': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '08': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': 0}],
    '09': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '10': [{'S': (0, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 1), 'E': (0.5, 1), 'D': (0, 0), 'F': 0}],
    '11': [{'S': (0, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '12': [{'S': (0, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': 0},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': 0}],
    '13': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0}],
    '14': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '15': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': 0}],
    '16': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '17': [{'S': (0, 0.5), 'E': (0, 0.5), 'D': (0, 0), 'F': 0},
           {'S': (1, 0.5), 'E': (1, 0.5), 'D': (0, 0), 'F': 0}],
    '18': [{'S': (0.5, 0), 'E': (0.5, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '19': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 1), 'E': (0.5, 1), 'D': (0, 0), 'F': 0}],
    '20': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '21': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '22': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': 0},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0}],
    '23': [{'S': (0, 0.5), 'E': (0, 0.5), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': 0},
           {'S': (1, 0.5), 'E': (1, 0.5), 'D': (0, 0), 'F': 0}],
    '24': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (1, 0), 'E': (1, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': 0},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
    '25': [{'S': (0, 0), 'E': (0, 0), 'D': (0, 0), 'F': 0},
           {'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (0, 0), 'F': 0},
           {'S': (0, 1), 'E': (0, 1), 'D': (0, 0), 'F': 0},
           {'S': (1, 1), 'E': (1, 1), 'D': (0, 0), 'F': 0}],
}

loads = {
    'A': [{'S': (0.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)}],
    'B': [{'S': (0.0, 1.0), 'E': (0.5, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -2 * x)},
          {'S': (0.5, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -2 * (1.0 - x))}],
    'C': [{'S': (0.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -x)}],
    'D': [{'S': (0.5, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)}],
    'E': [{'S': (0.5, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -2 * (x - 0.5))}],
    'F': [{'S': (1.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)}],
    'G': [{'S': (0.5, 1.0), 'E': (0.5, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)}],
    'H': [{'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)}],
    'I': [{'S': (1.0, 0.5), 'E': (1.0, 0.5), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)}],
    'J': [{'S': (0.0, 1.0), 'E': (0.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)},
          {'S': (1.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (0.0, -1.0)}],
    'K': [{'S': (0.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)}],
    'L': [{'S': (0.0, 1.0), 'E': (0.5, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (2 * x, 0)},
          {'S': (0.5, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (2 * (1.0 - x), 0)}],
    'M': [{'S': (0.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (2 * x, 0)}],
    'N': [{'S': (0.5, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)}],
    'O': [{'S': (0.5, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (2 * (x - 0.5), 0)}],
    'P': [{'S': (1.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)}],
    'Q': [{'S': (0.5, 1.0), 'E': (0.5, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)}],
    'R': [{'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)}],
    'S': [{'S': (1.0, 0.5), 'E': (1.0, 0.5), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)}],
    'T': [{'S': (0.0, 1.0), 'E': (0.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)},
          {'S': (1.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, 0)}],
    'U': [{'S': (1.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, -1.0)}],
    'V': [{'S': (0.5, 1.0), 'E': (0.5, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, -1.0)}],
    'W': [{'S': (0.5, 0.5), 'E': (0.5, 0.5), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, -1.0)}],
    'X': [{'S': (1.0, 0.5), 'E': (1.0, 0.5), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, -1.0)}],
    'Y': [{'S': (0.0, 1.0), 'E': (0.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, -1.0)},
          {'S': (1.0, 1.0), 'E': (1.0, 1.0), 'D': (np.nan, np.nan), 'F': lambda x, y: (1.0, -1.0)}],
}

ps = [3, 4, 5, 6]
rs = [1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 4.00, 5.00, 6.00, 8.00]
vfs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
