"""
Firstly written by Tae Hoon Oh 2022.05 (oh.taehoon.4i@kyoto-u.ac.jp)
CSTR dynamics [van der Vusse reaction] refers to
"Nonlinear predictive control of a benchmark CSTR (1995)"

6 states (x): Concentrations of A (x1) and B (x2) [mol / L],
              Temperatures of reactor (x3) and cooling jacket (x4) [degree]
              Normalized flow rate (x5) [1 / h] and Heat removal (x6) [kJ / h]
2 actions (u): Change of normalized flow rate (u1) [1 / h] and Change of heat removal (u2) [kJ / h]
4 outputs (y): Concentration of B (y1) [mol / L] and Temperature of reactor (y2) [degree]
            Normalized flow rate (y3) [1 / h] and Heat removal (y4) [kJ / h]
1 parameters (p): Temperature of feed (p1) [degree]

Abbreviations
x = state, u = input, y = output, p = parameter, ref = reference
dim = dimension / ini = initial / grad = gradient, coeff = coefficient,
con = concentration / temp = temperature / var = variance
"""

import numpy as np
import casadi as ca
import utility as ut


class SysCSTR(object):
    def __init__(self, seed, disturb):
        self.seed = seed
        np.random.seed(self.seed)
        self.disturb = disturb  # True/False

        self.x_dim = 6
        self.u_dim = 2
        self.y_dim = 4
        self.p_dim = 1

        self.ini_x = np.array([2., 1., 115., 110., 10, -1000])
        self.ini_u = np.array([0., 0.])
        self.ini_y = np.array([1., 115., 10, -1000])
        self.ini_p = np.array([104.9])
        self.time_interval = 1/60  # hour

        self.x_min = np.array([0., 0., 0., 0., 3., -9000.])
        self.x_max = np.array([5., 5., 200., 200., 35., 0.])
        self.u_min = np.array([-1., -100.])
        self.u_max = np.array([1., 100.])
        self.y_min = np.array([0., 0., 3., -9000.])
        self.y_max = np.array([5., 200., 35., 0.])
        self.p_min = np.array([100])
        self.p_max = np.array([115])

        #  y = ax + b, a = 1./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad = 1. / (self.x_max - self.x_min)

        # Disturbance related parameter
        self.p_std = 0.1
        if not self.disturb:
            self.p_std = 0.
        self.measure_std = np.array([0.01, 0.06, 0., 0.])

        # Fixed parameters
        self.ca0 = 5.10  # mol/L
        self.k10 = 1.287*10**12  # 1/h +- 0.04
        self.k20 = 1.287*10**12  # 1/h +- 0.04
        self.k30 = 9.043*10**9   # 1/(molA*h) +- 0.27
        self.E1 = -9758.3  # K
        self.E2 = -9758.3  # K
        self.E3 = -8560    # K
        self.Hab = 4.2      # kJ/molA  +- 2.36
        self.Hbc = -11.     # kJ/molA  +- 1.92
        self.Had = -41.85   # kJ/molA  +- 1.41
        self.rho_Cp = 2.8119  # kJ/(L*K) +- 0.000016
        self.kw_AR = 866.88  # kJ/(h*K)  +- 25.8
        self.VR = 10  # L
        self.mk = 5.0  # kg
        self.Cpk = 2.0  # kJ/(kg*K) +- 0.05

    def system_dynamics(self, state, action, para):
        x1, x2, x3, x4, x5, x6 = state
        u1, u2 = action
        p1 = para

        k1 = self.k10*np.exp(self.E1/(x3 + 273.15))
        k2 = self.k20*np.exp(self.E2/(x3 + 273.15))
        k3 = self.k30*np.exp(self.E3/(x3 + 273.15))

        heat_by_rxns = k1*x1*self.Hab + k2*x2*self.Hbc + k3*(x1**2)*self.Had

        x1dot = x5*(self.ca0 - x1) - k1*x1 - k3*(x1**2)
        x2dot = -x5*x2 + k1*x1 - k2*x2
        x3dot = x5*(p1 - x3) + self.kw_AR*(x4 - x3)/(self.rho_Cp*self.VR) - heat_by_rxns/self.rho_Cp
        x4dot = (x6 + self.kw_AR*(x3 - x4))/(self.mk*self.Cpk)
        x5dot = u1/self.time_interval
        x6dot = u2/self.time_interval
        xdot = [x1dot, x2dot, x3dot, x4dot, x5dot, x6dot]
        return xdot

    def reset(self):
        ref = np.array([1., 114.])
        return self.ini_x, self.ini_u, self.ini_y, self.ini_p, ref

    def disturbance_generation(self, p):
        p_next = p + np.random.normal(0, self.p_std, 1)
        if p_next > self.p_max:
            return p_next, self.p_max
        elif p_next < self.p_min:
            return p_next, self.p_min
        else:
            return p_next, p_next

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
        para_next, para_bdd = self.disturbance_generation(para)
        scaled_state = ut.scale(state, self.x_min, self.x_max)
        scaled_action = ut.scale(action, self.u_min, self.u_max)
        scaled_para = ut.scale(para_bdd, self.p_min, self.p_max)
        scaled_action_para = np.hstack((scaled_action, scaled_para))
        # scaled_state = np.clip(scaled_state, 0, 1)
        # scaled_action_para = np.clip(scaled_action_para, 0, 1)

        result = step_fcn(x0=scaled_state, p=scaled_action_para)
        scaled_next_state = np.squeeze(np.array(result['xf']))

        next_state = ut.descale(scaled_next_state, self.x_min, self.x_max)
        # next_state = np.clip(next_state, self.state_min, self.state_max)
        return next_state, para_next

    def observe(self, x):
        y = np.zeros(self.y_dim)
        y[0] = x[1] + np.random.normal(0, self.measure_std[0], 1)
        y[1] = x[2] + np.random.normal(0, self.measure_std[1], 1)
        y[2] = x[4] + np.random.normal(0, self.measure_std[2], 1)
        y[3] = x[5] + np.random.normal(0, self.measure_std[3], 1)
        y = np.clip(y, self.y_min, self.y_max)
        return y

    def cost(self, u, y, ref):
        # Cost parameters
        action_weight = 0.1*np.array([[1., 0.],
                                      [0., 0.01]])
        output_weight = np.array([[1., 0.],
                                  [0., 0.01]])
        action_cost = u.T @ action_weight @ u
        err = y[0:2] - ref
        output_cost = err.T @ output_weight @ err
        return action_cost + output_cost


if __name__ == '__main__':
    plant = SysCSTR(100, True)
    step_fcn = plant.make_step_function()
    N = 60*10
    x = np.zeros((plant.x_dim, N))
    u = np.zeros((plant.u_dim, N))
    y = np.zeros((plant.y_dim, N))
    p = np.zeros((plant.p_dim, N))
    c = np.zeros((1, N))
    r = np.zeros((2, N))
    x0, u0, y0, p0, ref = plant.reset()
    x[:, 0] = x0
    u[:, 0] = u0
    y[:, 0] = y0
    p[:, 0] = p0
    c[:, 0] = plant.cost(u0, y0, ref)
    r[:, 0] = ref

    PID_flow = ut.PID(1, 0, 0, 0)
    PID_temp = ut.PID(30, 5, 0, 0)

    for k in range(N-1):
        if k == 60*5:
            ref = np.array([1.1, 112])

        u[0, k] = PID_flow.control(ref[0] - y[0, k], plant.time_interval)
        u[1, k] = PID_temp.control(ref[1] - y[1, k], plant.time_interval)

        x[:, k+1], p[:, k+1] = plant.step(step_fcn, x[:, k], u[:, k], p[:, k])
        y[:, k+1] = plant.observe(x[:, k+1])
        c[:, k+1] = plant.cost(u[:, k], y[:, k], ref)
        r[:, k+1] = ref

    print('Total Cost : ', np.round(np.sum(c), 4))

    import matplotlib.pyplot as plt
    time = np.arange(N)/60
    plt.step(time, y[0, :])
    plt.step(time, x[1, :])
    plt.step(time, r[0, :])
    plt.xlabel('Time [hour]')
    plt.ylabel('Concentration of B [mol / L]')
    plt.show()

    plt.step(time, y[1, :], time, x[2, :], time, x[3, :])
    plt.step(time, r[1, :])
    plt.xlabel('Time [hour]')
    plt.ylabel('Reactor Temperature [degree]')
    plt.show()

    plt.step(time, p[0, :])
    plt.xlabel('Time [hour]')
    plt.ylabel('Feed Temperature [degree]')
    plt.show()

    plt.step(time, y[2, :])
    plt.xlabel('Time [hour]')
    plt.ylabel('Normalized flow rate [1 / h]')
    plt.show()

    plt.step(time, y[3, :])
    plt.xlabel('Time [hour]')
    plt.ylabel('Heat removal [kJ / h]')
    plt.show()

    plt.step(time, c[0, :])
    plt.xlabel('Time [hour]')
    plt.ylabel('Cost')
    plt.show()

