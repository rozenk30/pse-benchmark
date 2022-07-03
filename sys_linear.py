"""
Firstly written by Tae Hoon Oh 2022.05 (oh.taehoon.4i@kyoto-u.ac.jp)
linear system

4 states (x): dummy
2 input (u): dummy
2 outputs (y): x_1, x_3

Abbreviations
x = state, u = input, y = output, p = parameter
dim = dimension / ini = initial / grad = gradient, coeff = coefficient,
"""

import numpy as np
import casadi as ca
import utility as ut
import scipy
from scipy.signal import cont2discrete


class SysLinear(object):
    def __init__(self, seed, disturb):
        self.seed = seed
        np.random.seed(self.seed)
        self.disturb = disturb  # True/False

        self.x_dim = 4
        self.u_dim = 2
        self.y_dim = 2
        self.p_dim = 0

        self.ini_x = np.array([1., 0., 1., 0.])
        self.ini_u = np.array([0., 0.])
        self.ini_y = np.array([1., 1.])
        self.ini_p = np.array([])
        self.time_interval = 0.1  # -

        self.x_min = np.array([-10., -10., -10., -10.])
        self.x_max = np.array([10., 10., 10., 10.])
        self.u_min = np.array([-10., -10.])
        self.u_max = np.array([10., 10.])
        self.y_min = np.array([-10., -10.])
        self.y_max = np.array([10., 10.])
        self.p_min = np.array([])
        self.p_max = np.array([])

        #  y = ax + b, a = 1./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad = 1. / (self.x_max - self.x_min)

        # Disturbance related parameter
        self.para_std = 0.
        if not self.disturb:
            self.para_std = 0.
        self.measure_std = np.array([0.05, 0.05])

        # Fixed parameters
        self.a = np.array([[0., 1., 0., 0.],
                           [0.5, 1.1, 0.1, 0.],
                           [0., 0., 0., 1.],
                           [0., 0., 0.3, 1.5]])
        self.b = np.array([[0., 0.],
                           [1., 0.],
                           [0., 0.],
                           [0., 1.]])
        self.c = np.array([[1., 0., 0., 0.],
                           [0., 0., 1., 0.]])
        self.d = np.zeros((2, 2))

        self.q = np.eye(2)
        self.r = 0.1*np.eye(2)
        self.K = np.array([[3.11684841,  3.66069817,  0.11895173,  0.01905481],
                           [-0.01101439,  0.01273492,  2.83347392,  4.06790213]])
        discrete_linear_system = cont2discrete((self.a, self.b, self.c, self.d), self.time_interval, method='zoh')
        self.ad, self.bd, self.cd, self.dd, _ = discrete_linear_system

    def system_dynamics(self, state, action, para):
        x1, x2, x3, x4 = state
        u1, u2 = action

        x1dot = x2
        x2dot = 0.5*x1 + 1.1*x2 + 0.1*x3 + u1
        x3dot = x4
        x4dot = 0.3*x3 + 1.5*x4 + u2
        xdot = [x1dot, x2dot, x3dot, x4dot]
        return xdot

    def reset(self):
        return self.ini_x, self.ini_u, self.ini_y, self.ini_p

    def make_step_function(self):
        # scaled in - scaled out function
        x_ca = ca.SX.sym('state', self.x_dim)
        u_ca = ca.SX.sym('action', self.u_dim)
        p_ca = ca.SX.sym('para', self.p_dim)
        up_ca = ca.vcat([u_ca, p_ca])

        x_d = ut.descale(x_ca, self.x_min, self.x_max)
        u_d = ut.descale(u_ca, self.u_min, self.u_max)
        p_d = ut.descale(p_ca, self.p_min, self.p_max)

        if x_d.numel() > 1: x_sp = ca.vertsplit(x_d)
        else: x_sp = x_d
        if u_d.numel() > 1: u_sp = ca.vertsplit(u_d)
        else: u_sp = u_d
        if p_d.numel() > 1: p_sp = ca.vertsplit(p_d)
        else: p_sp = p_d

        # Integrating ODE with Casadi with solver cvodes
        xdot = np.multiply(ca.vcat(self.system_dynamics(x_sp, u_sp, p_sp)), self.scale_grad)  # Because of scaling
        ode = {'x': x_ca, 'p': up_ca, 'ode': xdot}
        options = {'t0': 0, 'tf': self.time_interval}
        ode_integrator = ca.integrator('Integrator', 'cvodes', ode, options)
        return ode_integrator

    def step(self, step_fcn, state, action, para):
        scaled_state = ut.scale(state, self.x_min, self.x_max)
        scaled_action = ut.scale(action, self.u_min, self.u_max)
        scaled_para = ut.scale(para, self.p_min, self.p_max)
        scaled_action_para = np.hstack((scaled_action, scaled_para))
        # scaled_state = np.clip(scaled_state, 0, 1)
        # scaled_action_para = np.clip(scaled_action_para, 0, 1)

        result = step_fcn(x0=scaled_state, p=scaled_action_para)
        scaled_next_state = np.squeeze(np.array(result['xf']))

        next_state = ut.descale(scaled_next_state, self.x_min, self.x_max)
        # next_state = np.clip(next_state, self.x_min, self.x_max)
        return next_state

    def observe(self, x):
        y = np.zeros(self.y_dim)
        y[0] = x[0] + np.random.normal(0, self.measure_std[0], 1)
        y[1] = x[2] + np.random.normal(0, self.measure_std[1], 1)
        y = np.clip(y, self.y_min, self.y_max)
        return y

    def cost(self, u, y):
        return u.T @ self.r @ u + y.T @ self.q @ y

    def lqr(self, x):
        q = np.diag([1., 0., 1., 0.])
        r = 0.1 * np.eye(2)
        w = scipy.linalg.solve_discrete_are(self.ad, self.bd, q, r, e=None, s=None, balanced=True)
        gain = np.linalg.inv(r + self.bd.T @ w @ self.bd) @ self.bd.T @ w @ self.ad
        return -gain @ x


if __name__ == '__main__':
    plant = SysLinear(100, False)
    step_fcn = plant.make_step_function()
    N = 100
    x = np.zeros((plant.x_dim, N))
    u = np.zeros((plant.u_dim, N))
    y = np.zeros((plant.y_dim, N))
    p = np.zeros((plant.p_dim, N))
    c = np.zeros((1, N))
    xp = np.zeros((plant.x_dim, N))
    pp = 0.*np.eye(4)

    x0, u0, y0, p0 = plant.reset()
    x[:, 0] = x0
    xp[:, 0] = x0
    u[:, 0] = u0
    y[:, 0] = y0
    p[:, 0] = p0

    for k in range(N-1):
        u[:, k] = plant.lqr(xp[:, k])
        x[:, k+1] = plant.step(step_fcn, x[:, k], u[:, k], p[:, k]) + np.random.normal(0, 0.02, 4)
        y[:, k+1] = plant.observe(x[:, k+1])
        xp[:, k+1], pp = ut.kalman_filter(xp[:, k], pp, u[:, k], y[:, k+1], plant.ad, plant.bd, plant.cd,
                                          0.02*np.eye(4), 0.05*np.eye(2))
        c[:, k] = plant.cost(u[:, k], y[:, k])

    print(np.sum(c))

    import matplotlib.pyplot as plt
    time = range(N)
    plt.plot(time, y[0, :])
    plt.ylabel('y1')
    plt.show()

    plt.plot(time, y[1, :])
    plt.ylabel('y2')
    plt.show()

