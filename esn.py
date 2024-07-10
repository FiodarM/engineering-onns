import numpy as np
from scipy import integrate as spode
from scipy.io import wavfile
import matplotlib.pyplot as plt


class ESN:
    def __init__(
        self, size, T, eps, G, phi_0, beta=np.pi, rho=np.pi, M=1, seed=1
    ):
        self.size = size
        self.T = T  # Clock cycle
        self.eps = eps
        self.beta = beta
        self.G = G
        self.phi_0 = phi_0
        self.rho = rho
        self.M = M
        self.s_hist = None
        self.t_hist = None
        self.seed = seed
        np.random.seed(self.seed)
        self.Win = np.random.uniform(-1, 1, size=(size,))
        self.sol = None

    def activation(self, x):
        return 0.5 * self.G * (1 + self.M * np.sin(x + self.phi_0))

    def mask(self, s):
        i = np.floor((self.size * s) % (self.size * self.T)).astype(int)
        return self.Win[i]

    def sample_and_hold(self, x):

        def u(t):
            i = int(t // self.T)
            if 0 <= i < len(x):
                return x[i]
            return 0

        return u

    def derivative(self, t, s, u=None):
        prev_s = np.interp(t - 1, self.t_hist, self.s_hist)
        preactivation = self.beta * prev_s
        if u:
            preactivation += self.rho * self.mask(t) * u(t)
        ds_dt = (-s + self.activation(preactivation)) / self.eps

        # Update history
        self.s_hist[:-1] = self.s_hist[1:]
        self.s_hist[-1] = s[0]
        self.t_hist[:-1] = self.t_hist[1:]
        self.t_hist[-1] = t
        return ds_dt

    def run(self, x=0, s0=0, t_eval=None, method='RK45'):
        x = np.atleast_1d(x)
        self.t_hist = np.linspace(-1, 0, self.size)
        self.s_hist = np.full(self.size, s0)
        if t_eval is None:
            t_eval = np.arange(0, len(x), 1 / (2 * self.size))
        self.sol = spode.solve_ivp(
            self.derivative, t_span=[0, t_eval[-1]], t_eval=t_eval, y0=[s0],
            method=method, vectorized=True, args=(self.sample_and_hold(x),)
        )
        return self.sol

    def plot_solution(self, ax=None):
        # plot the results
        if not ax:
            fig, ax = plt.subplots(figsize=(5, 3), layout='tight')
        else:
            fig = ax.gcf()
        ax.plot(self.sol.t, self.sol.y[0], label='$s(t)$', color='k')
        u = self.sample_and_hold(x)
        plain_input = np.array([u(t) for t in self.sol.t])
        ax.plot(self.sol.t, plain_input, label='input', color='b')
        ax.plot(self.sol.t, self.mask(self.sol.t) * plain_input, label='masked input', color='r')
        ax.set_xlabel('t')
        ax.legend(loc='right')
        return fig, ax


if __name__ == "__main__":
    # set parameter values
    T = 1
    G = 0.3  # nonlinearity gain
    beta = np.pi  # feedback scaling
    rho = np.pi  # input scaling
    phi_0 = np.pi * 0.89  # offset phase of the MZM
    size = 50  # number of virtual nodes
    eps = 5 / size  # response time
    M = 0.98
    esn = ESN(size, T, eps, G, phi_0, beta, rho, M)
    t = np.linspace(0, 3, 100)
    x = np.sin(t)
    esn.run(x, s0=0.2)
    fig, ax = esn.plot_solution()
    plt.show()
