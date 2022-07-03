"""
Firstly written by Tae Hoon Oh 2021.05 (oh.taehoon.4i@kyoto-u.ac.jp)
SMB model refers to
"Transition model for simulated moving bed under nonideal conditions (2019)"
"Automatic control of simulated moving bed process with deep Q-network (2021)"
"Simulated Moving Bed Chromatography for the Separation of Enantiomers (2009)"

# states (1614): Concentrations, time step, mode, previous action, concentration of extact and raffiante tanks
# actions (8): velocity of 8 sections
# outputs (10): concentration of 4 ports + purities of extract and raffinate tanks

Abbreviations
x = state, u = input, y = output, p = parameter
dim = dimension / ini = initial / para = parameter / grad = gradient
con = concentration / temp = temperature / var = variance / coeff = coefficient / equili = equilibrium
const = constant /
"""


import numpy as np
import casadi as ca
import utility as ut


class SysSMB(object):
    def __init__(self, seed, disturb):
        self.seed = seed
        np.random.seed(self.seed)
        self.disturb = disturb  # True/False

        self.grid_num = 50  # Number of grid for a single column
        self.column_num = 8

        self.x_dim = 4*self.column_num*self.grid_num + 1 + 1 + 8 + 4
        self.u_dim = self.column_num
        self.y_dim = 4
        self.p_dim = 0

        self.ini_x = np.zeros(self.x_dim)
        self.ini_x[-1], self.ini_x[-4] = 0.01, 0.01
        self.ini_u = np.array([0.022, 0.022, 0.013, 0.013, 0.014, 0.014, 0.0100, 0.0100]) # np.array([0.022, 0.022, 0.0125, 0.0125, 0.0145, 0.0145, 0.0100, 0.0100])
        self.ini_y = np.array([0.01, 0., 0., 0.01])
        self.ini_p = np.array([])
        self.time_interval = 10  # sec [delt]

        self.x_min = np.zeros(self.x_dim)
        self.x_max = np.ones(self.x_dim)
        self.u_min = np.zeros(self.u_dim)
        self.u_max = np.ones(self.u_dim)
        self.y_min = np.zeros(self.y_dim)
        self.y_max = np.ones(self.y_dim)
        self.p_min = np.array([])
        self.p_max = np.array([])

        #  y = ax + b, a = 1./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad = 1. / (self.x_max - self.x_min)

        # Disturbance related parameter
        self.measure_var = np.zeros(self.y_dim)
        if self.disturb:
            self.measure_var = 0.0001*np.ones(self.y_dim)

        # Fixed parameters of column [Abbreviation]
        self.length = 1.  # m [l]
        self.diameter = 0.1  # m [Dia]
        self.porosity = 0.66  # [e]
        # self.pe = 1000  # Non-dimensional number
        self.diffusion_coeff = 0.00001  # [D]
        # H = 2.25, 0.8 corresponding v = 0.0180 0.0122, del 0.1 is 0.0088
        self.equili_const = np.array([0.5, 0.2])  # [K]
        self.langmuir_coeff = np.array([5, 5])  # [qm]
        # Henry constant = self.equili_const * self.qm
        self.mass_transfer_coeff = np.array([2, 2])  # [k]
        self.switch_time = 120  # sec
        self.feed_con = np.array([1.0, 1.0])  # [feed_con]

        # Additional values
        self.area = np.pi*self.diameter*self.diameter/4  # [A]
        self.volume = self.area * self.length  # [V]
        self.grid_length = self.length / self.grid_num  # [delz]
        self.time_grid_num = self.switch_time / self.time_interval  # Number of grid in time axis

        # Cost related parameters
        self.Product = -1
        self.Desorbent = 0.1
        self.Extract_purity_penalty = 0.01
        self.Raffinate_purity_penalty = 0.01

    def _column_dynamics(self, state, action, para):

        y, v = state, action
        N, D, delz, e = self.grid_num, self.diffusion_coeff, self.grid_length, self.porosity
        K, k, qm = self.equili_const, self.mass_transfer_coeff, self.langmuir_coeff
        ee = (1 - e)/e
        xdot = 4*N*[0.]

        # Boundary condition
        xdot[0*N] = 0  # (v**2/self.D)*(c0[0] - y[0]) + self.D*(y[1] - y[0])/(self.delz**2)
        xdot[1*N-1] = D*(y[N-1] - 2*y[N-1] + y[N-2])/(delz**2) - v*(y[N-1] - y[N-2])/(delz) \
                     - ee*k[0]*(qm[0]*K[0]*y[N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[2*N-1])
        xdot[1*N] = k[0]*(qm[0]*K[0]*y[0]/(1+K[0]*y[0]+K[1]*y[2*N]) - y[N])
        xdot[2*N-1] = k[0]*(qm[0]*K[0]*y[N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[2*N-1])
        xdot[2*N] = 0  # (v**2/self.D)*(c0[1] - y[2*self.N]) + self.D*(y[2*self.N + 1] - y[2*self.N])/(self.delz**2)
        xdot[3*N-1] = D*(y[3*N-1] - 2*y[3*N-1] + y[3*N-2])/(delz**2) - v*(y[3*N-1] - y[3*N-2])/(delz) \
                     - ee*k[1]*(qm[1]*K[1]*y[3*N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[4*N-1])
        xdot[3*N] = k[1]*(qm[1]*K[1]*y[2*N]/(1+K[0]*y[0]+K[1]*y[2*N]) - y[3*N])
        xdot[4*N-1] = k[1]*(qm[1]*K[1]*y[3*N-1]/(1+K[0]*y[N-1]+K[1]*y[3*N-1]) - y[4*N-1])

        # Internal dynamics
        for i in range(N-2):
            xdot[0*N+i+1] = D*(y[i+2] - 2*y[i+1] + y[i])/(delz**2) - v*(y[i+1] - y[i])/delz \
                           - ee*k[0]*(qm[0]*K[0]*y[i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[N+i+1])
            xdot[1*N+i+1] = k[0]*(qm[0]*K[0]*y[i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[N+i+1])
            xdot[2*N+i+1] = D*(y[2*N+i+2] - 2*y[2*N+i+1] + y[2*N+i])/(delz**2) - v*(y[2*N+i+1] - y[2*N+i])/delz \
                           - ee*k[1]*(qm[1]*K[1]*y[2*N+i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[3*N+i+1])
            xdot[3*N+i+1] = k[1]*(qm[1]*K[1]*y[2*N+i+1]/(1+K[0]*y[i+1]+K[1]*y[2*N+i+1]) - y[3*N+i+1])
        return xdot

    def make_column_step_function(self):
        # scaled in - scaled out function
        x_ca = ca.SX.sym('state', 4*self.grid_num)
        u_ca = ca.SX.sym('action', 1)  # Column flowrate
        p_ca = ca.SX.sym('para', 0)
        up_ca = ca.vcat([u_ca, p_ca])

        x_d = ut.descale(x_ca, self.x_min[0:4*self.grid_num], self.x_max[0:4*self.grid_num])
        u_d = ut.descale(u_ca, self.u_min[0], self.u_max[0])
        p_d = ut.descale(p_ca, self.p_min, self.p_max)
        scale_grad = 1. / (self.x_max[0:4*self.grid_num] - self.x_min[0:4*self.grid_num])

        if x_d.numel() > 1: x_sp = ca.vertsplit(x_d)
        else: x_sp = x_d
        if u_d.numel() > 1: u_sp = ca.vertsplit(u_d)
        else: u_sp = u_d
        if p_d.numel() > 1: p_sp = ca.vertsplit(p_d)
        else: p_sp = p_d

        # Integrating ODE with Casadi with solver cvodes
        xdot = np.multiply(ca.vcat(self._column_dynamics(x_sp, u_sp, p_sp)), scale_grad)  # Because of scaling
        ode = {'x': x_ca, 'p': up_ca, 'ode': xdot}
        options = {'t0': 0, 'tf': self.time_interval}
        column_ode_integrator = ca.integrator('Integrator', 'cvodes', ode, options)
        return column_ode_integrator

    def _column_step(self, column_step_fcn, state, concen_in, action, para):

        state[0] = concen_in[0]
        state[2*self.grid_num] = concen_in[1]

        scaled_state = ut.scale(state, self.x_min[0:4*self.grid_num], self.x_max[0:4*self.grid_num])
        scaled_action = ut.scale(action, self.u_min[0], self.u_max[0])
        scaled_para = ut.scale(para, self.p_min, self.p_max)
        scaled_action_para = np.hstack((scaled_action, scaled_para))
        # scaled_state = np.clip(scaled_state, 0, 1)
        # scaled_action_para = np.clip(scaled_action_para, 0, 1)

        result = column_step_fcn(x0=scaled_state, p=scaled_action_para)
        scaled_next_state = np.squeeze(np.array(result['xf']))

        next_state = ut.descale(scaled_next_state, self.x_min[0:4*self.grid_num], self.x_max[0:4*self.grid_num])
        # next_state = np.clip(next_state, self.x_min[0:4*self.grid_num], self.x_max[0:4*self.grid_num])
        terminal_con = np.array([next_state[self.grid_num - 1], next_state[3*self.grid_num - 1]])
        return next_state, terminal_con

    def step(self, column_step_fcn, state, action, para):

        N = self.grid_num

        time_step = state[4*self.column_num*N]
        mode = int(round(state[4*self.column_num*N + 1]*7))
        extract_con = state[4*self.column_num*N + 10: 4*self.column_num*N + 12]
        raffinate_con = state[4*self.column_num*N + 12: 4*self.column_num*N + 14]

        role_index = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        role_index = np.concatenate((role_index[8 - mode:8], role_index[0:8 - mode]))

        s1, s2, s3, s4 = state[0*N:4*N], state[4*N:8*N], state[8*N:12*N], state[12*N:16*N]
        s5, s6, s7, s8 = state[16*N:20*N], state[20*N:24*N], state[24*N:28*N], state[28*N:32*N]

        s1_in = np.array([s8[N - 1], s8[3*N - 1]])
        s2_in = np.array([s1[N - 1], s1[3*N - 1]])
        s3_in = np.array([s2[N - 1], s2[3*N - 1]])
        s4_in = np.array([s3[N - 1], s3[3*N - 1]])
        s5_in = np.array([s4[N - 1], s4[3*N - 1]])
        s6_in = np.array([s5[N - 1], s5[3*N - 1]])
        s7_in = np.array([s6[N - 1], s6[3*N - 1]])
        s8_in = np.array([s7[N - 1], s7[3*N - 1]])

        c_in = np.array([s1_in, s2_in, s3_in, s4_in, s5_in, s6_in, s7_in, s8_in])

        c_in[np.where(role_index == 0)] = c_in[np.where(role_index == 0)] * action[7] / action[0]
        c_in[np.where(role_index == 1)] = c_in[np.where(role_index == 1)] * action[0] / action[1]
        c_in[np.where(role_index == 2)] = c_in[np.where(role_index == 2)]
        c_in[np.where(role_index == 3)] = c_in[np.where(role_index == 3)] * action[2] / action[3]
        c_in[np.where(role_index == 4)] = (c_in[np.where(role_index == 4)] * action[3]
                                           + self.feed_con*(action[4] - action[3]))/action[4]
        c_in[np.where(role_index == 5)] = c_in[np.where(role_index == 5)] * action[4] / action[5]
        c_in[np.where(role_index == 6)] = c_in[np.where(role_index == 6)]
        c_in[np.where(role_index == 7)] = c_in[np.where(role_index == 7)] * action[6] / action[7]

        next_c1, s1_out = self._column_step(column_step_fcn, s1, c_in[0], action[role_index[0]], para)
        next_c2, s2_out = self._column_step(column_step_fcn, s2, c_in[1], action[role_index[1]], para)
        next_c3, s3_out = self._column_step(column_step_fcn, s3, c_in[2], action[role_index[2]], para)
        next_c4, s4_out = self._column_step(column_step_fcn, s4, c_in[3], action[role_index[3]], para)
        next_c5, s5_out = self._column_step(column_step_fcn, s5, c_in[4], action[role_index[4]], para)
        next_c6, s6_out = self._column_step(column_step_fcn, s6, c_in[5], action[role_index[5]], para)
        next_c7, s7_out = self._column_step(column_step_fcn, s7, c_in[6], action[role_index[6]], para)
        next_c8, s8_out = self._column_step(column_step_fcn, s8, c_in[7], action[role_index[7]], para)
        s_out = np.array([s1_out, s2_out, s3_out, s4_out, s5_out, s6_out, s7_out, s8_out])
        extract_con += s_out[np.where(role_index == 1)][0]*action[1]*self.area
        raffinate_con += s_out[np.where(role_index == 5)][0]*action[5]*self.area

        time_step += 0.08

        if time_step == 0.08*12:
            mode += 1
            time_step = 0

        if mode == 8:
            mode = 0

        next_s = np.concatenate((next_c1, next_c2, next_c3, next_c4, next_c5, next_c6, next_c7, next_c8,
                                 [round(time_step, 2)], [1/7*mode], action, extract_con, raffinate_con))
        return next_s

    def cost(self, state, action):
        puri_e = state[-4] / (state[-4] + state[-3])
        puri_r = state[-1] / (state[-2] + state[-1])
        if puri_e > 0.995:
            penalty_extract = 0.1*(puri_e - 0.995)
        elif puri_e < 0.99:
            penalty_extract = self.Extract_purity_penalty
        else:
            penalty_extract = 0

        if puri_r > 0.995:
            penalty_raffinate = 0.1*(puri_r - 0.995)
        elif puri_r < 0.99:
            penalty_raffinate = self.Raffinate_purity_penalty
        else:
            penalty_raffinate = 0
        cost = self.Product*(action[4] - action[3]) + self.Desorbent*(action[0] - action[7]) \
               + penalty_extract + penalty_raffinate
        return cost

    def reset(self):
        return self.ini_x, self.ini_u, self.ini_y, self.ini_p

    def observe(self, state):
        output = np.zeros(self.y_dim)
        output[0] = state[-4] + np.random.normal(0, self.measure_var[0], 1)
        output[1] = state[-3] + np.random.normal(0, self.measure_var[1], 1)
        output[2] = state[-2] + np.random.normal(0, self.measure_var[2], 1)
        output[3] = state[-1] + np.random.normal(0, self.measure_var[3], 1)
        output = np.clip(output, self.y_min, self.y_max)
        return output


if __name__ == '__main__':
    plant = SysSMB(100, False)
    column_step_fcn = plant.make_column_step_function()
    N = 12*8*50
    x = np.zeros((plant.x_dim, N))
    u = np.zeros((plant.u_dim, N))
    y = np.zeros((plant.y_dim, N))
    p = np.zeros((plant.p_dim, N))
    c = np.zeros((1, N))

    x0, u0, y0, p0 = plant.reset()
    x[:, 0] = x0
    u[:, 0] = u0
    y[:, 0] = y0
    p[:, 0] = p0

    for k in range(N-1):
        print(k)
        u[:, k] = u0  # np.array([0.022, 0.022, 0.0118, 0.0118, 0.01521, 0.01521, 0.0100, 0.0100])
        x[:, k+1] = plant.step(column_step_fcn, x[:, k], u[:, k], p[:, k])  # + np.random.normal(0, 0.02, 4)
        y[:, k+1] = plant.observe(x[:, k+1])
        c[:, k] = plant.cost(x[:, k], u[:, k])

    print(np.sum(c))

    import matplotlib.pyplot as plt
    plt.plot(x[0:1600, -1])
    plt.ylabel('State')
    plt.show()

    time = range(N)
    plt.plot(time, y[0, :]/(y[0, :] + y[1, :]), time, y[3, :]/(y[2, :] + y[3, :]))
    plt.ylabel('Purities')
    plt.show()

    plt.plot(time, y[0, :], time, y[1, :])
    plt.ylabel('Extract Tank')
    plt.show()

    plt.plot(time, y[2, :], time, y[3, :])
    plt.ylabel('Raffinate Tank')
    plt.show()

    plt.plot(c[0])
    plt.ylabel('Cost')
    plt.show()
