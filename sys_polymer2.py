"""
Firstly written by Tae Hoon Oh 2021.06 (oh.taehoon.4i@kyoto-u.ac.jp)
Polymerization process model refers to
"Online Optimal Feedback Control of Polymerization Reactors:
Application to Polymerization of Acrylamide−Water−Potassium Persulfate (KPS) System (2017)"
""

# states (3+3+35): Mole of components [mole], moments of dead polymers * Volume [mol],
                   chain length fraction
# actions (3): Flow rate of monomer and initiator [ml/min], and Temperature [K]
# outputs (2): Monomer concentration [mol/m**3], and polymer weight average [kg/mol]

Units: mol, J, kg, m**3, K, min

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

        self.x_dim = 6 + 35
        self.u_dim = 3
        self.y_dim = 2
        self.p_dim = 0

        self.ini_x = np.zeros(self.x_dim)
        self.ini_x[0:6] = np.array([0.3, 50., 0.005, 0., 0.000001, 0.])
        self.ini_u = np.array([1.12*10**(-6), 3.0*10**(-6), 45+273.15])
        self.ini_y = np.array([0., 0.])
        self.ini_p = np.array([])
        self.time_interval = 2.5  # min
        self.terminal_time = 180  # min

        self.x_min = np.zeros(self.x_dim)
        self.x_max = 100*np.ones(self.x_dim)
        # self.x_max[0:7] = np.array([1., 100., 0.1, 1000., 1000., 1000., 200+273.15])
        self.u_min = np.array([0.1*10**(-6), 0.1*10**(-6), 40+273.15])
        self.u_max = np.array([5*10**(-6), 5*10**(-6), 80+273.15])
        self.y_min = np.array([0., 0.])
        self.y_max = np.array([10000., 10000.])
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
        self.pre_coeff_of_decomposition = 7.15*10**12  # 1/min [Ad] (2017)
        self.pre_coeff_of_propagation = 4.08*10**5  # m^3/(mol min) [Ap] (2017)
        self.pre_coeff_of_termination = 4.08*10**9  # m^3/(mol min) [At] (2017)
        self.pre_coeff_of_CT_to_monomer = 0.  # L/(mol min) [Afm]
        self.pre_coeff_of_CT_to_solvent = 0.  # L/(mol min) [Afs]

        self.act_energy_of_decomposition = 101123.182  # J/mol [Ed]
        self.act_energy_of_propagation = 11700.  # J/mol [Ep]
        self.act_energy_of_termination = 11700.  # J/mol [Et]
        self.act_energy_of_CT_to_monomer = 0.  # J/mol [Et]
        self.act_energy_of_CT_to_solvent = 0.  # J/mol [Et]

        self.monomer_molar_mass = 0.07108  # kg/mol  Acrylamide
        self.solvent_molar_mass = 0.01801528  # kg/mol  Water
        self.initiator_molar_mass = 0.270322  # kg/mol  Potassium Persulfate
        self.monomer_density = 1130.  # kg/m**3 [rho_m]
        self.solvent_density_coeff = np.array([-0.031, -0.1437, 1003])  # Water kg/m**3 [rho_s], (2017)
        # self.solvent_density_coeff = np.array([-0.0036, 1.9270, 745.5147])  # Water kg/m**3 [rho_s], fitted by oh
        self.polymer_density = 1302.  # kg/m**3 [rho_p]
        self.initiator_density = 2480.  # kg/m**3 [rho_i]
        self.initiator_efficiency_factor = 0.196  # (2017) [f]
        self.gas_constant = 8.3144598  # J/K mol
        self.monomer_feed_monomer_con = 7175.014  # self.monomer_density/self.monomer_molar_mass  # mol/m**3 Pure ?????
        self.monomer_feed_solvent_con = 0.  # mol/L ?????
        self.initiator_feed_initiator_con = 110.9796  # self.initiator_density/self.initiator_molar_mass  # mol/m**3 Pure ?????
        self.initiator_feed_solvent_con = 0.  # mol/L ?????
        self.feed_out_for_sample = 0.5*10**(-6)  # m**3/min ?????

    def system_dynamics(self, state, action, para):

        nm, ns, ni, l0v, l1v, l2v = state[0], state[1], state[2], state[3], state[4], state[5]
        fm, fi, temp = action

        wm, ws, wi = self.monomer_molar_mass, self.solvent_molar_mass, self.initiator_molar_mass
        cmfm, cmfs = self.monomer_feed_monomer_con, self.monomer_feed_solvent_con
        cifi, cifs = self.initiator_feed_initiator_con, self.initiator_feed_solvent_con
        fout = self.feed_out_for_sample
        rho_m, rho_p, rho_i = self.monomer_density, self.polymer_density, self.initiator_density
        rho_s = self.solvent_density_coeff[0]*(temp-273.15)**2\
                + self.solvent_density_coeff[1]*(temp-273.15) + self.solvent_density_coeff[2]
        #rho_s = self.solvent_density_coeff[0]*temp**2\
        #        + self.solvent_density_coeff[1]*temp + self.solvent_density_coeff[2]
        vol = l1v*wm/rho_p + nm*wm/rho_m + ns*ws/rho_s + ni*wi/rho_i  # m**3
        cm, ci, cs = nm/vol, ni/vol, ns/vol

        kp0 = self.pre_coeff_of_propagation*np.exp(-self.act_energy_of_propagation/(self.gas_constant*temp))
        kd = self.pre_coeff_of_decomposition*np.exp(-self.act_energy_of_decomposition/(self.gas_constant*temp))
        ktc = 0.
        ktd0 = self.pre_coeff_of_termination*np.exp(-self.act_energy_of_termination/(self.gas_constant*temp))
        kfm = self.pre_coeff_of_CT_to_monomer*np.exp(-self.act_energy_of_CT_to_monomer/(self.gas_constant*temp))
        kfs = self.pre_coeff_of_CT_to_solvent*np.exp(-self.act_energy_of_CT_to_solvent/(self.gas_constant*temp))

        # Gel effect
        gt = 1.
        gp = 1.

        # Cage effect
        initiator_efficiency = 1.*self.initiator_efficiency_factor

        ktd = gt*ktd0
        kp = gp*kp0
        p0 = np.sqrt(2*initiator_efficiency*ci*kd/ktd)
        alpha = kp*cm/(kp*cm + kfm*cm + kfs*cs + ktd*p0 + ktc*p0)

        # Differential equations
        dnmdt = -(kp+kfm)*p0*nm + fm*cmfm - fout*cm
        dnidt = -kd*ni + fi*cifi - fout*ci
        dnsdt = -kfs*ns*p0 + fi*cifs + fm*cmfs - fout*cs
        dl0vdt = (kfm*nm + ktd*p0*vol + kfs*ns)*alpha*p0 + 0.5*ktc*vol*p0**2
        dl1vdt = ((kfm*nm + ktd*p0*vol + kfs*ns)*(2*alpha - alpha**2) + ktc*p0*vol)*p0/(1 - alpha)  # N -> vol typo
        dl2vdt = ((kfm*nm + ktd*p0*vol + kfs*ns)*(alpha**3 - 3*alpha**2 + 4*alpha)
                  + ktc*p0*vol*(alpha+2)*p0/(1-alpha))*p0/((1 - alpha)**2)
        xdot = [dnmdt, dnsdt, dnidt, dl0vdt, dl1vdt, dl2vdt]

        for k in range(self.x_dim - 6):
            a = 5
            start_length = 2 + a*k*(k+1)
            end_length = 1 + a*(k+1)*(k+2)
            dummy = (vol/l1v)*kp*cm*p0*((start_length*(1-alpha) + alpha)*alpha**(start_length-2)
                                      - ((end_length+1)*(1-alpha) + alpha)*alpha**(end_length-1))
            xdot.append(dummy - 1/l1v*state[6 + k]*dl1vdt)

        return xdot

    def reset(self):
        return self.ini_x, self.ini_u, self.observe(self.ini_x, self.ini_u), self.ini_p

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

    def observe(self, state, action):
        output = np.zeros(self.y_dim)

        wm, ws, wi = self.monomer_molar_mass, self.solvent_molar_mass, self.initiator_molar_mass
        rho_m, rho_p, rho_i = self.monomer_density, self.polymer_density, self.initiator_density
        rho_s = self.solvent_density_coeff[0]*(action[2]-273.15)**2\
                + self.solvent_density_coeff[1]*(action[2]-273.15) + self.solvent_density_coeff[2]
        vol = state[4]*wm/rho_p + state[0]*wm/rho_m + state[1]*ws/rho_s + state[2]*wi/rho_i
        output[0] = state[0]/vol
        output[1] = wm*state[5]/state[4]

        output += np.random.normal(0, self.measure_var, self.y_dim)
        # output = np.clip(output, self.y_min, self.y_max)
        return output

    def cost(self, u, y):
        return 0


''' 
Validation
'''

# Note! the fraction predicted by model depends on the interval
# No cage effect, no gel effect for propagation
if __name__ == '__main__':
    plant = SysPolymer(100, False)
    step_fcn = plant.make_step_function()
    N = 1000
    x = np.zeros((plant.x_dim, N))
    u = np.zeros((plant.u_dim, N))
    y = np.zeros((plant.y_dim, N))
    p = np.zeros((plant.p_dim, N))
    x0, u0, y0, p0 = plant.reset()
    x[:, 0] = x0
    u[:, 0] = u0
    y[:, 0] = y0
    for k in range(N-1):
        u[:, k] = u0
        if k > 8:
            u[0, k] = 0.38*10**(-6)
        plant.system_dynamics(x[:, k], u[:, k], [])
        x[:, k+1] = plant.step(step_fcn, x[:, k], u[:, k], p[:, k])
        y[:, k+1] = plant.observe(x[:, k+1], u[:, k])

    print(np.sum(x[6:, -1])) # summation of fraction

    import matplotlib.pyplot as plt
    average_chain_length = []
    for k in range(35):
        a = 5
        start_length = 2 + a*k*(k + 1)
        end_length = 1 + a*(k + 1)*(k + 2)
        average_chain_length.append(0.5*(1*start_length + 1*end_length))
    plt.plot(average_chain_length, x[6:, -1], 'o-')
    plt.ylabel('Average chain length')
    plt.ylabel('Weight fraction in interval')
    # plt.axis([0, 6000, 0, 0.2])
    plt.show()

    time = np.arange(0, 2.5*N, 2.5)
    plt.plot(time, y[0, :]*0.07108/1000, 'o-')
    plt.xlabel('Time (min)')
    plt.ylabel('Monomer Concentration')
    plt.show()

    plt.plot(time, y[1, :], 'o-')
    plt.xlabel('Time (min)')
    plt.ylabel('Weight averaged molar mass')
    plt.show()

