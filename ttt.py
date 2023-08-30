import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mc


def top_os(nelx, nely, X0, volfrac, costfrac, penal, rmin, D, E, P, MColor, MName, MinMove):
    # INITIALIZE
    x = X0 * np.ones((nely, nelx))
    loop = 0
    change = 1.0

    fig, ax, im, bg = init_fig(np.reshape(x, (nelx, nely), 'F'), D, MColor, MName)
    # START ITERATION
    while change > 1.01 * MinMove:
        loop += 1
        xold = np.copy(x)

        # FE-ANALYSIS
        E_, dE_ = ordered_simp_interpolation(nelx, nely, x, penal, D, E)
        P_, dP_ = ordered_simp_interpolation(nelx, nely, x, 1 / penal, D, P)
        U = fe(nelx, nely, E_)

        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        KE = lk()
        c = 0.0
        dc = np.zeros((nely, nelx))
        for ely in range(nely):
            for elx in range(nelx):
                n1 = (nely + 1) * (elx - 1) + ely
                n2 = (nely + 1) * elx + ely
                Ue = U[[2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n1 + 1, 2 * n1 + 2], 0]
                c += E_[ely, elx] * Ue.T @ KE @ Ue
                dc[ely, elx] = -dE_[ely, elx] * Ue.T @ KE @ Ue

        # FILTERING OF SENSITIVITIES
        dc = check(nelx, nely, rmin, x, dc)

        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD
        x = oc(nelx, nely, x, volfrac, costfrac, dc, E_, dE_, P_, dP_, loop, MinMove)

        # PRINT RESULTS
        change = np.max(np.abs(x - xold))
        print(
            f"It.: {loop}  Obj.: {c:.4f}  Mass Fraction: {np.sum(x) / (nelx * nely):.3f}  Cost Fraction: {np.sum(x * P_) / (nelx * nely):.3f}  Ch.: {change:.3f}")

        # RENDER MULTI-MATERIAL TOPOLOGY
        plot(x, loop, change, fig, ax, im, bg)
    plt.show(block=True)


def oc(nelx, nely, x, volfrac, costfrac, dc, E_, dE_, P_, dP_, loop, MinMove):
    dc = -1 * dc
    lV1 = 0
    lV2 = 2 * np.max(dc)
    Temp = P_ + x * dP_
    Temp = dc / Temp
    lP1 = 0
    lP2 = 2 * np.max(Temp)
    move = max(0.15 * 0.96 ** loop, MinMove)

    while (((lV2 - lV1) / (lV1 + lV2) > 1e-6) or ((lP2 - lP1) / (lP1 + lP2) > 1e-6)):
        lmidV = 0.5 * (lV2 + lV1)
        lmidP = 0.5 * (lP2 + lP1)
        Temp = lmidV + lmidP * P_ + lmidP * x * dP_
        Coef = dc / Temp
        Coef = np.abs(Coef)
        xnew = np.maximum(10 ** -5, np.maximum(x - move, np.minimum(1., np.minimum(x + move, x * np.sqrt(Coef)))))

        if np.sum(xnew) - volfrac * nelx * nely > 0:
            lV1 = lmidV
        else:
            lV2 = lmidV

        CurrentCostFrac = np.sum(xnew * P_) / (nelx * nely)
        if CurrentCostFrac - costfrac > 0:
            lP1 = lmidP
        else:
            lP2 = lmidP

    return xnew


def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))
    for i in range(nelx):
        for j in range(nely):
            s = 0.0
            for k in range(max(i - int(rmin), 0), min(i + int(rmin), nelx)):
                for l in range(max(j - int(rmin), 0), min(j + int(rmin), nely)):
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    s += max(0, fac)
                    dcn[j, i] += max(0, fac) * x[l, k] * dc[l, k]
            dcn[j, i] /= x[j, i] * s
    return dcn


def fe(nelx, nely, E_Intopolation):
    KE = lk()
    K = np.zeros((2 * (nelx + 1) * (nely + 1), 2 * (nelx + 1) * (nely + 1)))
    F = np.zeros((2 * (nely + 1) * (nelx + 1), 1))
    U = np.zeros((2 * (nely + 1) * (nelx + 1), 1))

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * (elx - 1) + ely
            n2 = (nely + 1) * elx + ely
            edof = [2 * n1 - 1, 2 * n1, 2 * n2 - 1, 2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n1 + 1, 2 * n1 + 2]
            K[edof, :][:, edof] += E_Intopolation[ely, elx] * KE

            # DEFINE LOADS AND SUPPORTS (Bridge)
            F[2 * (nely + 1) * (nelx // 4 + 1) - 1, 0] = -1
            F[2 * (nely + 1) * (2 * nelx // 4 + 1) - 1, 0] = -2
            F[2 * (nely + 1) * (3 * nelx // 4 + 1) - 1, 0] = -1
            fixeddofs = np.union1d([2 * (nely + 1) - 2, 2 * (nely + 1) - 1], [2 * (nelx + 1) * (nely + 1) - 1])
            alldofs = np.arange(0, 2 * (nely + 1) * (nelx + 1))
            freedofs = np.setdiff1d(alldofs, fixeddofs)

            # SOLVING
            U[freedofs, :] = np.linalg.solve(K[np.ix_(freedofs, freedofs)], F[freedofs, :])
            U[fixeddofs, :] = 0

    return U


def ordered_simp_interpolation(nelx, nely, x, penal, X, Y):
    y = np.zeros((nely, nelx))
    dy = np.zeros((nely, nelx))

    for i in range(nelx - 1):
        for j in range(nely - 1):
            for k in range(len(X) - 1):
                if (X[k] < x[j, i]) and (X[k + 1] >= x[j, i]):
                    A = (Y[k] - Y[k + 1]) / (X[k] ** (1 * penal) - X[k + 1] ** (1 * penal))
                    B = Y[k] - A * (X[k] ** (1 * penal))
                    y[j, i] = A * (x[j, i] ** (1 * penal)) + B
                    dy[j, i] = A * penal * (x[j, i] ** ((1 * penal) - 1))
                    break

    return y, dy


def lk():
    E = 1.0
    nu = 0.3
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE = E / (1 - nu ** 2) / 24 * (np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]]))
    return KE


def init_fig(x, D, colors, names):
    plt.ion()
    fig, ax = plt.subplots()
    cmap = mc.LinearSegmentedColormap.from_list('mesh', list(zip(D, colors)))
    custom_lines = [Line2D([0], [0], marker='o', label='Scatter',
                           lw=0, markerfacecolor=c, markersize=10) for c in colors]
    im = ax.imshow(x, origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax.set_title(F'Iteration: {0}, Change: {1:0.4f}')
    ax.legend(custom_lines, names, ncol=len(colors))  # , bbox_to_anchor=(0.8, -0.15)
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.1)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    return fig, ax, im, bg


def plot(x, loop, ch, fig, ax, im, bg):
    fig.canvas.restore_region(bg)
    im.set_array(x)
    ax.set_title(F'Iteration: {loop}, Change: {ch:0.5f}')
    ax.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()


D = [0, 0.4, 0.7, 1.0]
E = [0, 0.2, 0.6, 1.0]
P = [0, 0.5, 0.8, 1.0]
MName = {'Void' 'A' 'B' 'C'}
MColor = ['w', 'b', 'r', 'k']
top_os(100, 50, 0.5, 0.4, 0.3, 3, 2.5, D, E, P, MColor, MName, 0.001)
