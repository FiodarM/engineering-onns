import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def reservoir_system(x, s, eps, beta, mu, phi_0, s_history, x_history):
    # get previous state at s-1 using interpolation
    s_i = len(s_history)
    if s_i > 1:
        prev_x = np.interp(s_i - 1, s_history, x_history, left=0.0, right=0.0)
    else:
        prev_x = x_history[0]

    dxds = (-x + beta * np.sin(mu * prev_x + phi_0) ** 2) / eps

    # append current state to history
    x_history.append(x[0])
    s_history.append(s)

    return dxds


def u(t):
    # define the input signal here
    return 0.0


# calculate time values based on delay time
tau_d = 20.87E-6  # delay time
Tr = 240e-9
eps = Tr / tau_d

# initialize history arrays
s_history = [0.0]
x0 = 0
x_history = [x0]

# set parameter values
beta = 0.3  # nonlinearity gain
mu = 2.5  # feedback scaling
rho = 0.0  # relative weight of input information compared to feedback signal
phi_0 = np.pi * 0.89  # offset phase of the MZM

s = np.linspace(0, 5, 1000)
ds = s[1] - s[0]
t = s * tau_d

# integrate the system over time
x = odeint(reservoir_system, x0, s,
           args=(eps, beta, mu, phi_0, s_history, x_history))

# plot the results
plt.plot(s, x)
plt.xlabel('s')
plt.ylabel('x')
plt.title(
    'Reservoir Computing without External Signal\n'
    fr'($\beta$={beta}, $\mu$={mu}, $\Phi_0$={phi_0:.2f})'
)
plt.show()
