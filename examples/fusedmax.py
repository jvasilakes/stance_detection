import numpy as np
import matplotlib.pyplot as plt


def supmax(x, y):
    max_idx = np.matmul(y, x.T).argmax(0)
    return y[max_idx]


def sparsemax(x, y, gamma=1.0):
    output = []
    for xi in x:
        reg = np.linalg.norm(y - xi/gamma, axis=1)**2
        min_idx = reg.argmin()
        output.append(y[min_idx])
    return np.array(output)


def fusedmax(x, y, gamma=1.0, lam=0.5):
    output = []
    for xi in x:
        reg = 0.5 * np.linalg.norm(y - xi/gamma, axis=1)**2
        tv = lam * np.abs(y[:, 1:] - y[:, :-1]).sum(1)
        min_idx = (reg + tv).argmin()
        output.append(y[min_idx])
    return np.array(output)


if __name__ == "__main__":
    t = np.arange(-4, 4, 0.1)
    x = np.vstack([np.zeros_like(t), t]).T
    y_ = np.linspace(0.0, 1.0, 100)
    y = np.vstack([y_, y_[::-1]]).T

    plt.plot(t, supmax(x, y)[:, 1], label="max")
    plt.plot(t, sparsemax(x, y)[:, 1], label="sparsemax")
    plt.plot(t, fusedmax(x, y)[:, 1], label="fusedmax")
    plt.xlabel('t')
    plt.legend()
    plt.title("*max([0, t])")
    plt.tight_layout()
    plt.show()
