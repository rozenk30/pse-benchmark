"""
Firstly written by Tae Hoon Oh 2021.05 (oh.taehoon.4i@kyoto-u.ac.jp)
Polymerization process model refers to
"Combining On-Line Characterization Tools with Modern Software Environments for Optimal
Operation of Polymerization Processes (2016)"
""

# states (3+3+35): Mole of components [mole], moments of dead polymers * Volume [mol],
                   Temperature [K], and chain length fraction
# actions (3): Flow rate of monomer and initiator [ml/min], and temperature [Celsius] (?)
# outputs (2): Monomer conversion [-], and polymer weight average [kg/mol]

Units: mol, J, g, L, K, min

solvent: Ethyl acetate (1997),  Butyl acetate (2016)

Abbreviations
x = state, u = input, y = output, p = parameter
dim = dimension / ini = initial / para = parameter / grad = gradient
con = concentration / temp = temperature / var = variance
coeff = coefficient / act = activation
CT = chain transfer
"""


import numpy as np
import casadi as ca
import utility as ut


class SysPolymer(object):
    def __init__(self, seed, disturb):
        self.seed = seed
        np.random.seed(self.seed)
        self.disturb = disturb  # True/False

        self.x_dim = 7 + 15
        self.u_dim = 3
        self.y_dim = 2
        self.p_dim = 0

        self.ini_x = np.zeros(self.x_dim)
        self.ini_x[0:7] = np.array([0.5, 0.5, 0.01, 0.00001, 0.00001, 0.00001, 65+273.15])
        self.ini_u = np.array([0.8*10**(-3), 1.6*10**(-3), 65+273.15])
        self.ini_y = np.array([0., 0.])
        self.ini_p = np.array([])
        self.time_interval = 2.  # min
        self.terminal_time = 1800  # min

        self.x_min = np.zeros(self.x_dim)
        self.x_max = np.ones(self.x_dim)
        # self.x_max[0:7] = np.array([1., 100., 0.1, 1000., 1000., 1000., 200+273.15])
        self.u_min = np.array([0., 0., 0])
        self.u_max = np.array([5., 5., 500])
        self.y_min = np.array([0., 0.])
        self.y_max = np.array([1., 5000.])
        self.p_min = np.array([])
        self.p_max = np.array([])

        #  y = ax + b, a = 1./(max - min) , dy/dt = a*dx/dt = af(x)
        self.scale_grad = 1. / (self.x_max - self.x_min)

        # Disturbance related parameter
        self.para_var = 0.1
        if not self.disturb:
            self.para_var = 0.
        self.measure_var = 0*np.array([0.02, 0.02])

        # Fixed parameters
        self.pre_coeff_of_decomposition = 1.37*10**15  # +- 1.49*10**14  1/min [Ad] (2016)
        self.pre_coeff_of_propagation = 8.5*10**8  # +- 6.23*10**7  L/(mol min) [Ap] (2016)
        self.pre_coeff_of_termination = 4.56*10**11  # +- 7.12*10**10  L/(mol min) [At] (2016)
        self.pre_coeff_of_CT_to_monomer = 1.75*10**13  # L/(mol min) [Afm]
        self.pre_coeff_of_CT_to_solvent = 6.95*10**10  # L/(mol min) [Afs]

        self.act_energy_of_decomposition = 4.184*34277.  # J/mol [Ed]
        self.act_energy_of_propagation = 4.184*6300.  # J/mol [Ep]
        self.act_energy_of_termination = 4.184*2800.  # J/mol [Et]
        self.act_energy_of_CT_to_monomer = 4.184*17957.  # J/mol [Et]
        self.act_energy_of_CT_to_solvent = 4.184*15702.  # J/mol [Et]

        self.monomer_molar_mass = 100.121  # g/mol
        # self.solvent_molar_mass = 88.11  # Ethly acetate g/mol
        self.solvent_molar_mass = 116.16  # Butyl acetate g/mol
        self.initiator_molar_mass = 192.26  # g/mol

        self.monomer_density_coeff = np.array([-1.1004, 1268.5])  # g/L [rho_m]
        # self.solvent_density_coeff = np.array([-1.2322, 1261.4])  # Ethly acetate g/L [rho_s]
        self.solvent_density_coeff = np.array([-1.2653, 1250.8])  # Butyl acetate g/L [rho_s]
        self.polymer_density = 1150.  # +- 30 g/L [rho_p]
        self.initiator_density = 900.  # +- 100 g/L [rho_i]
        self.initiator_efficiency_factor = 0.56  # +- 0.057 [f] (2016)
        self.gas_constant = 8.3144598  # J/K mol

        self.monomer_feed_monomer_con = 8.9492  # mol/L ?????
        self.monomer_feed_solvent_con = 0.  # mol/L ?????
        self.initiator_feed_initiator_con = 4.6812  # mol/L ?????
        self.initiator_feed_solvent_con = 0.  # mol/L ?????
        self.feed_out_for_sample = 0.3*10**(-3)  # L/min ?????

        self.gel_effect_monomer_coeff = np.array([0.001, 167])
        self.gel_effect_polymer_coeff = np.array([0.00048, 387])
        self.gel_effect_solvent_coeff = np.array([0.001, 149.94])

    def system_dynamics(self, state, action, para):

        nm, ns, ni, l0v, l1v, l2v, temp = state[0], state[1], state[2], state[3], state[4], state[5], state[6]
        fm, fi, temp_ref = action

        temp = temp_ref  # ####################

        wm, ws, wi = self.monomer_molar_mass, self.solvent_molar_mass, self.initiator_molar_mass
        cmfm, cmfs = self.monomer_feed_monomer_con, self.monomer_feed_solvent_con
        cifi, cifs = self.initiator_feed_initiator_con, self.initiator_feed_solvent_con
        fout = self.feed_out_for_sample

        rho_m = self.monomer_density_coeff[0]*temp + self.monomer_density_coeff[1]
        rho_s = self.solvent_density_coeff[0]*temp + self.solvent_density_coeff[1]
        vol = l1v*wm/rho_m + nm*wm/rho_m + ns*ws/rho_s + ni*wi/self.initiator_density  # L
        cm, ci, cs = nm/vol, ni/vol, ns/vol
        kp0 = self.pre_coeff_of_propagation*np.exp(-self.act_energy_of_propagation/(self.gas_constant*temp))
        kd = self.pre_coeff_of_decomposition*np.exp(-self.act_energy_of_decomposition/(self.gas_constant*temp))
        ktc = 0.
        ktd0 = self.pre_coeff_of_termination*np.exp(-self.act_energy_of_termination/(self.gas_constant*temp))
        kfm = self.pre_coeff_of_CT_to_monomer*np.exp(-self.act_energy_of_CT_to_monomer/(self.gas_constant*temp))
        kfs = self.pre_coeff_of_CT_to_solvent*np.exp(-self.act_energy_of_CT_to_solvent/(self.gas_constant*temp))

        # Gel effect
        vfm = 0.025 + self.gel_effect_monomer_coeff[0]*(temp - self.gel_effect_monomer_coeff[1])
        vfs = 0.025 + self.gel_effect_solvent_coeff[0]*(temp - self.gel_effect_solvent_coeff[1])
        vfp = 0.025 + self.gel_effect_polymer_coeff[0]*(temp - self.gel_effect_polymer_coeff[1])
        phi_m = (nm*wm/rho_m) / vol
        phi_s = (ns*ws/rho_s) / vol
        phi_p = (l1v*wm/rho_m) / vol
        vf = vfm*phi_m + vfs*phi_s + vfp*phi_p
        vtc = 0.1856 - 2.965*10**(-4)*temp

        #####
        gt = 0.10575 * np.exp(17.15*vf - 0.01715*(temp - 273.15))
        # if vf > vtc:
        #     gt = 0.10575*np.exp(17.15*vf - 0.01715*temp)
        # else:
        #     gt = 2.3*10**(-6)*np.exp(75*vf)

        gp = 1.
        # if vf > 0.05:
        #     gp = 1.
        # else:
        #     gp = 7.1*10**(-5)*np.exp(171.53*vf)

        # Cage effect
        vfcr = 0.1856 - 2.965*10**(-4)*(temp - 273.15)
        initiator_efficiency = self.initiator_efficiency_factor*np.exp(-0.06*(vf**(-1) - vfcr**(-1)))

        ktd = gt*ktd0
        kp = gp*kp0
        p0 = np.sqrt(2*initiator_efficiency*ci*kd/ktd)
        alpha = kp*cm/(kp*cm + kfm*cm + kfs*cs + ktd*p0)

        # Differential equations
        dnmdt = -(kp+kfm)*p0*nm + fm*cmfm - fout*cm
        dnidt = -kd*ni + fi*cifi - fout*ci
        dnsdt = -kfs*ns*p0 + fi*cifs + fm*cmfs - fout*cs
        dl0vdt = (kfm*nm + ktd*p0*vol + kfs*ns)*alpha*p0 + 0.5*ktc*vol*p0**2
        dl1vdt = ((kfm*nm + ktd*p0*vol + kfs*ns)*(2*alpha - alpha**2) + ktc*p0*vol)*p0/(1 - alpha)  # N -> vol typo
        dl2vdt = ((kfm*nm + ktd*p0*vol + kfs*ns)*(alpha**3 - 3*alpha**2 + 4*alpha)
                  + ktc*p0*vol*(alpha+2)*p0/(1-alpha))*p0/((1 - alpha)**2)
        xdot = [dnmdt, dnsdt, dnidt, dl0vdt, dl1vdt, dl2vdt, 0.]
        for k in range(self.x_dim - 7):
            a = 27
            start_length = 2 + a*k*(k+1)
            end_length = 1 + a*(k+1)*(k+2)
            dummy = (vol/l1v)*kp*cm*p0*((start_length*(1-alpha) + alpha)*alpha**(start_length-2)
                                      - ((end_length+1)*(1-alpha) + alpha)*alpha**(end_length-1))
            xdot.append(dummy - 1/l1v*state[7+k]*dl1vdt)

        print(dnidt, ci, cifi, fout, fi)

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

    def observe(self, state):
        output = np.zeros(self.y_dim)

        output[0] = (self.ini_x[0] - state[0])/self.ini_x[0] + np.random.normal(0, self.measure_var[0], 1)
        output[1] = self.monomer_molar_mass*state[5]/state[4] + np.random.normal(0, self.measure_var[1], 1)
        # output = np.clip(output, self.y_min, self.y_max)
        return output

    def cost(self, u, y):
        return 0


''' 
Validation, data from 
Discrete Optimal Control of Molecular Weight Distribution in a Batch Free Radical Polymerization Process (1997)
'''
'''
# Note! the fraction predicted by model depends on the interval
# No cage effect, no gel effect for propagation
if __name__ == '__main__':
    plant = SysPolymer(100, False)
    plant.pre_coeff_of_decomposition = 1.14*10**19  # 1/min [Ad]
    plant.pre_coeff_of_propagation = 4.2*10**8  # L/(mol min) [Ap]
    plant.pre_coeff_of_termination = 1.06*10**11  # L/(mol min) [At]
    plant.pre_coeff_of_CT_to_monomer = 1.75*10**13  # L/(mol min) [Afm]
    plant.pre_coeff_of_CT_to_solvent = 6.95*10**10  # L/(mol min) [Afs]
    plant.act_energy_of_decomposition = 4.184*34277.  # J/mol [Ed]
    plant.act_energy_of_propagation = 4.184*6300.  # J/mol [Ep]
    plant.act_energy_of_termination = 4.184*2800.  # J/mol [Et]
    plant.act_energy_of_CT_to_monomer = 4.184*17957.  # J/mol [Et]
    plant.act_energy_of_CT_to_solvent = 4.184*15702.  # J/mol [Et]
    plant.solvent_molar_mass = 88.11  # Ethly acetate g/mol
    plant.solvent_density_coeff = np.array([-1.2322, 1261.4])  # Ethly acetate g/L [rho_s]
    plant.initiator_efficiency_factor = 0.21  # [f] with no cage effect
    plant.gel_effect_solvent_coeff = np.array([0.001, 181])
    plant.monomer_feed_monomer_con = 0.  # mol/L 
    plant.monomer_feed_solvent_con = 0.  # mol/L 
    plant.initiator_feed_initiator_con = 0.  # mol/L 
    plant.initiator_feed_solvent_con = 0.  # mol/L 
    plant.feed_out_for_sample = 0.  # L/min
    plant.ini_x[0:7] = np.array([4.6964, 5.0733, 0.0468, 0.00001, 0.00001, 0.00001, 65+273.15])


    step_fcn = plant.make_step_function()
    N = 38 + 40
    x = np.zeros((plant.x_dim, N))
    u = np.zeros((plant.u_dim, N))
    y = np.zeros((plant.y_dim, N))
    p = np.zeros((plant.p_dim, N))
    x0 = np.zeros(plant.x_dim)
    x0[0:7] = np.array([4.6964, 5.0733, 0.0468, 0.00001, 0.00001, 0.00001, 65 + 273.15])
    u0 = np.array([0., 0., 65 + 273.15])
    y0 = plant.observe(x0)
    x[:, 0] = x0
    u[:, 0] = u0
    y[:, 0] = y0
    for k in range(N-1):
        u[:, k] = u0
        if k > 38:
            u0[2] = 50 + 273.15
        x[:, k+1] = plant.step(step_fcn, x[:, k], u[:, k], p[:, k])
        y[:, k+1] = plant.observe(x[:, k+1])

    print(np.sum(x[7:, -1])) # summation of fraction

    import matplotlib.pyplot as plt
    average_chain_length = []
    for k in range(15):
        a = 27
        start_length = 2 + a * k * (k + 1)
        end_length = 1 + a * (k + 1) * (k + 2)
        average_chain_length.append(0.5*(1*start_length + 1*end_length))
    # plt.plot(average_chain_length, x[7:, -1], 'o-')
    data_chain_length = [68, 98, 139, 199, 288, 421, 611, 917, 1397, 2138, 3318, 5276]
    data_fraction = [0.0168, 0.0270, 0.0431, 0.0686, 0.1108, 0.1564, 0.1881, 0.1906, 0.1248, 0.0479, 0.0113, 0.0032]
    model_fraction = np.interp(data_chain_length, average_chain_length, x[7:, -1])
    plt.plot(data_chain_length, model_fraction, 'o')
    plt.plot(data_chain_length, data_fraction, '*')
    plt.ylabel('Average chain length')
    plt.ylabel('Weight fraction in interval')
    plt.axis([0, 6000, 0, 0.2])
    plt.show()

    data_time = np.array([45, 60, 75, 86, 104, 119, 131, 146, 177, 194])
    data_time += -data_time[0]
    data_conversion = [0.0014, 0.0453, 0.0915, 0.1441, 0.1904, 0.2588, 0.2875, 0.3035, 0.3225, 0.3407]
    time = np.arange(0, 2*N, 2)
    plt.plot(time, y[0, :], '-')
    plt.plot(data_time, data_conversion, '*')
    plt.xlabel('Time (min)')
    plt.ylabel('Conversion')
    plt.show()
'''

''' 
Validation, data from 
Combining On-Line Characterization Tools with Modern Software Environments 
for Optimal Operation of Polymerization Processes (2016)
'''

if __name__ == '__main__':
    plant = SysPolymer(100, False)
    step_fcn = plant.make_step_function()
    N = 5
    x = np.zeros((plant.x_dim, N))
    u = np.zeros((plant.u_dim, N))
    y = np.zeros((plant.y_dim, N))
    p = np.zeros((plant.p_dim, N))
    x0, u0, y0, p0 = plant.reset()
    y0 = plant.observe(x0)
    x[:, 0] = x0
    u[:, 0] = u0
    y[:, 0] = y0

    for k in range(N-1):
        u[:, k] = u0
        x[:, k+1] = plant.step(step_fcn, x[:, k], u[:, k], p[:, k])
        y[:, k+1] = plant.observe(x[:, k+1])
        print(plant.system_dynamics(x[:, k], u[:, k], []))

    print(np.sum(x[7:, -1])) # summation of fraction

    import matplotlib.pyplot as plt
    average_chain_length = []
    for k in range(15):
        a = 20
        start_length = 2 + a * k * (k + 1)
        end_length = 1 + a * (k + 1) * (k + 2)
        average_chain_length.append(0.5*(1*start_length + 1*end_length))
    plt.plot(average_chain_length, x[7:, -1], 'o-')
    data_chain_length = [68, 98, 139, 199, 288, 421, 611, 917, 1397, 2138, 3318, 5276]
    data_fraction = [0.0168, 0.0270, 0.0431, 0.0686, 0.1108, 0.1564, 0.1881, 0.1906, 0.1248, 0.0479, 0.0113, 0.0032]
    model_fraction = np.interp(data_chain_length, average_chain_length, x[7:, -1])
    plt.plot(data_chain_length, model_fraction, 'o')
    plt.plot(data_chain_length, data_fraction, '*')
    plt.ylabel('Average chain length')
    plt.ylabel('Weight fraction in interval')
    plt.axis([0, 6000, 0, 0.2])
    plt.show()

    data_time = np.array([45, 60, 75, 86, 104, 119, 131, 146, 177, 194])
    data_time += -data_time[0]
    data_conversion = [0.0014, 0.0453, 0.0915, 0.1441, 0.1904, 0.2588, 0.2875, 0.3035, 0.3225, 0.3407]
    time = np.arange(0, 2*N, 2)
    plt.plot(time, y[0, :], '-')
    plt.plot(data_time, data_conversion, '*')
    plt.xlabel('Time (min)')
    plt.ylabel('Conversion')
    plt.show()

    plt.plot(time, x[2, :])
    plt.xlabel('Time (min)')
    plt.ylabel('states')
    plt.show()



