"""
Firstly written by Tae Hoon Oh 2021.04 (oh.taehoon.4i@kyoto-u.ac.jp)
Penicillin product fed-batch bioreactor dynamics refers to
"The development of an industrial-scale fed-batch fermentation simulation (2015)"

# 29 states : Time, A_0, A_1, A_3, A_4, Integral_X, S, P, V, T, H, n0 - n9 nm, DO, DCO2, viscosity, PAA, O2, CO2, Ni
# 1 action : F (-> F_S, F_oil by feed_ratio)
# 5 outputs : Time, X (A_0 + A_1 + A_3 + A_4), S, P, V
# We fixed the RPM and F_dis.

*** 'sys_penicillin' simulation with PID controllers  ***

Abbreviations
x = state, u = input, y = output, p = parameter, ref = reference
dim = dimension / ini = initial / grad = gradient, coeff = coefficient,
con = concentration / temp = temperature / var = variance
"""


import numpy as np
import utility as ut
from sys_penicillin import SysPenicillin
from utility import PID


class SysPeniWithPID(object):
    def __init__(self, seed, disturb):

        # Define the plant with seed
        self.plant = SysPenicillin(seed=seed, disturb=disturb)
        self.step_fcn = self.plant.make_step_function()

        self.u_min = 10
        self.u_max = 240

        # Define the target value for PID
        self.temp_ref = 298.
        self.pH_ref = 6.5
        self.paa_ref = 1000  # Should be in 200 - 2000
        self.viscosity_ref = 100
        self.ni_upper = 400  # > 150
        self.dissolved_o2_upper = 8  # > 6.6

        # Define the PID controller
        self.PID_cooling = PID(4000, 80, 1000, 2500)
        self.PID_heating = PID(5, 0, 0, 0)
        self.PID_acid = PID(0.01, 0, 0, 0)
        self.PID_base = PID(1.5, 0.1, 0, 0)
        self.PID_water = PID(1, 0, 0, 0)
        self.PID_paa = PID(0.12, 0, 0, 0)
        self.PID_dissolved_o2 = PID(0.08, 0, 0, 0)

        self.previous_action = np.zeros((self.plant.u_dim, self.plant.horizon_length + 1))

    def reset(self):
        x0, u0, y0, p0 = self.plant.reset()
        return x0, u0, y0, p0

    def path_cost(self, action, output):
        action_cost_coeff1 = 0.0005
        action_cost_coeff2 = 0.001
        scaled_input = ut.scale(action, self.u_min, self.u_max)
        path_cost = action_cost_coeff1*scaled_input + action_cost_coeff2*scaled_input**2
        return path_cost

    def terminal_cost(self, action, output):
        terminal_cost_coeff1 = 0.5
        output = ut.scale(output, self.plant.y_min, self.plant.y_max)
        terminal_cost = terminal_cost_coeff1*(1 - output[3]*output[4])  # + terminal_cost_coeff2*(1 - state[7])
        return terminal_cost

    @staticmethod
    def local_input_to_plant_input(local_input, time_index):
        ratio = np.loadtxt('utility_files/sys_peni_feed_ratio.txt')
        if time_index > 459:
            time_index = 459
        f_s = local_input/(1 + 5/3*ratio[time_index])
        f_oil = ratio[time_index]*local_input/(1 + 5/3*ratio[time_index])
        return f_s, f_oil

    @staticmethod
    def plant_input_to_local_input(plant_action):
        local_input = plant_action[0] + 5/3*plant_action[1]
        return local_input

    def step(self, state, action, time_index):

        if time_index > 160:
            self.PID_cooling = PID(1200, 0, 2000, 0)  # Gain scheduling

        if time_index == 0:
            previous_action = np.array([8., 22., 5., 0., 24.6, 0., 0.5, 0., 0., 0.6, 0.])
        else:
            previous_action = self.previous_action[:, time_index-1]

        time, a_0, a_1, a_3, a_4, integral_x, s, p, v, temp, hydro_ion, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, nm, \
        dissolved_o2, dissolved_co2, viscosity, paa, o2, co2, ni = state
        #f_s, f_oil, f_paa, f_a, f_b, f_w, f_g, f_c, f_h, pressure, nh3 = action
        f_s, f_oil = self.local_input_to_plant_input(action[0], time_index)

        f_s_p, f_oil_p, f_paa_p, f_a_p, f_b_p, f_w_p, f_g_p, f_c_p, f_h_p, pressure_p, nh3_p = previous_action
        f_s_l, f_oil_l, f_paa_l, f_a_l, f_b_l, f_w_l, f_g_l, f_c_l, f_h_l, pressure_l, nh3_l = self.plant.u_min
        f_s_u, f_oil_u, f_paa_u, f_a_u, f_b_u, f_w_u, f_g_u, f_c_u, f_h_u, pressure_u, nh3_u = self.plant.u_max

        # PID
        if temp > self.temp_ref - 0.01:
            f_c = self.PID_cooling.control(temp - self.temp_ref, self.plant.time_interval)
            f_h = f_h_l
        elif temp < self.temp_ref - 0.1:
            f_c = f_c_l
            f_h = self.PID_heating.control(self.temp_ref - temp, self.plant.time_interval)
        else:
            f_c = f_c_p
            f_h = f_h_p

        if -np.log10(hydro_ion) > self.pH_ref + 0.03:
            f_a = self.PID_acid.control(-self.pH_ref - np.log10(hydro_ion), self.plant.time_interval)
            f_b = f_b_l
        elif -np.log10(hydro_ion) < self.pH_ref:
            f_a = f_a_l
            f_b = self.PID_base.control(np.log10(hydro_ion) + self.pH_ref, self.plant.time_interval)
        else:
            f_a = f_a_p
            f_b = f_b_p

        if paa < 400:
            f_paa = f_paa_u
        elif paa > 1000:
            f_paa = f_paa_l
        else:
            f_paa = self.PID_paa.control(self.paa_ref - paa, self.plant.time_interval)
            f_paa = np.clip(f_paa, f_paa_l, f_paa_u)

        if viscosity > self.viscosity_ref:
            f_w = self.PID_water.control(viscosity - self.viscosity_ref, self.plant.time_interval)
        else:
            f_w = f_w_l

        if ni < self.ni_upper:
            nh3 = nh3_u
        else:
            nh3 = nh3_p

        if dissolved_o2 < self.dissolved_o2_upper:
            f_g = f_g_p + self.PID_dissolved_o2.control(self.dissolved_o2_upper - dissolved_o2, self.plant.time_interval)
        else:
            f_g = f_g_p

        # Fixed one?
        pressure = 0.6

        action = np.array([f_s, f_oil, f_paa, f_a, f_b, f_w, f_g, f_c, f_h, pressure, nh3])
        self.previous_action[:, time_index] = action
        next_state = self.plant.step(self.step_fcn, state, action, [])
        next_state = np.squeeze(next_state)

        return next_state, action

    def observe(self, state):
        output = self.plant.observe(state)
        return output


if __name__ == '__main__':
    plant = SysPeniWithPID(100, False)
    N = 460
    x = np.zeros((plant.plant.x_dim, N))
    u = np.zeros((1, N))
    y = np.zeros((plant.plant.y_dim, N))
    p = np.zeros((plant.plant.p_dim, N))
    c = np.zeros((1, N))
    x0, u0, y0, p0 = plant.reset()
    x[:, 0] = x0
    y[:, 0] = y0
    p[:, 0] = p0

    for k in range(N-1):
        u[:, k] = 80
        x[:, k+1], _ = plant.step(x[:, k], u[:, k], k)
        y[:, k+1] = plant.observe(x[:, k+1])
        c[:, k] = plant.path_cost(u[:, k], y[:, k])
    c[:, N-1] = plant.terminal_cost(u[:, N-1], y[:, N-1])
    print('Total Cost:', np.sum(c))

    import matplotlib.pyplot as plt
    plt.plot(y[0, :], y[1, :])
    plt.xlabel('Time'); plt.title('X'); plt.show()
    plt.plot(y[0, :], y[2, :])
    plt.xlabel('Time'); plt.title('S'); plt.show()
    plt.plot(y[0, :], y[3, :])
    plt.xlabel('Time'); plt.title('P'); plt.show()
    plt.plot(y[0, :], y[4, :])
    plt.xlabel('Time'); plt.title('V'); plt.show()
    plt.plot(y[0, :], c[0, :])
    plt.xlabel('Time'); plt.title('Cost'); plt.show()

    # Time, A_0, A_1, A_3, A_4, Integral_X, S, P, V, T, H, n0 - n9, nm, DO, DCO2, viscosity, PAA, O2, CO2, Ni
    plt.plot(y[0, :], x[1, :], y[0, :], x[2, :], y[0, :], x[3, :], y[0, :], x[4, :])
    plt.xlabel('Time'); plt.title('Biomass'); plt.show()

    plt.plot(y[0, :], x[9, :], y[0, :], 298*np.ones(N), 'r')
    plt.xlabel('Time'); plt.title('Temperature'); plt.show()

    plt.plot(y[0, :], -np.log10(x[10, :]), y[0, :], 6.5*np.ones(N), 'r')
    plt.xlabel('Time'); plt.title('pH'); plt.show()

    plt.plot(y[0, :], x[11, :], y[0, :], x[14, :], y[0, :], x[17, :], y[0, :], x[20, :], y[0, :], x[21, :])
    plt.xlabel('Time'); plt.title('n'); plt.show()

    plt.plot(y[0, :], x[22, :], y[0, :], 2.2*np.ones(N), 'r', y[0, :], 6.6*np.ones(N), 'r')
    plt.xlabel('Time'); plt.title('DO'); plt.show()

    plt.plot(y[0, :], x[23, :], y[0, :], 0.035*np.ones(N))
    plt.xlabel('Time'); plt.title('DCo2'); plt.show()

    plt.plot(y[0, :], x[24, :])
    plt.xlabel('Time'); plt.title('Viscosity'); plt.show()

    plt.plot(y[0, :], x[25, :], y[0, :], 200*np.ones(N), 'r', y[0, :], 2000*np.ones(N), 'r')
    plt.xlabel('Time'); plt.title('PAA'); plt.show()

    plt.plot(y[0, :], x[26, :], y[0, :], x[27, :], y[0, :], x[28, :], y[0, :], 150*np.ones(N), 'r')
    plt.xlabel('Time'); plt.title('O2, Co2, Ni'); plt.show()
