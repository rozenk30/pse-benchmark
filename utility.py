import matplotlib.pyplot as plt
import numpy as np

# Maybe made the saving formmat...


# scale X -> [0,1]
def scale(var, var_min, var_max):
    scaled_var = (var - var_min) / (var_max - var_min)
    return scaled_var


# descale [0,1] - > X
def descale(var, var_min, var_max):
    descaled_var = (var_max - var_min) * var + var_min
    return descaled_var


# Formal plotting - Recommand to use MATLAB
def plot(t, x):
    plt.plot(t, x, 'o-', color=[0, 0.4470, 0.7410], linewidth=2, markersize=8)
    plt.title('Title', fontsize=20)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.show()
    # Colors from MATLAB
    # [0, 0.4470, 0.7410]
    # [0.8500, 0.3250, 0.0980]
    # [0.9290, 0.6940, 0.1250]
    # [0.4940, 0.1840, 0.5560]
    # [0.4660, 0.6740, 0.1880]
    # [0.3010, 0.7450, 0.9330]
    # [0.6350, 0.0780, 0.1840]


def kalman_filter(x, p, u, y, a, b, c, q, r):
    n = np.shape(a)[0]
    x_pri = a @ x + b @ u
    p_pri = a @ p @ a.T + q
    p_pri = (p_pri + p_pri.T)/2
    K = p_pri @ c.T @ np.linalg.inv(c @ p_pri @ c.T + r)
    x_pos = x_pri + K @ (y - c @ x_pri)
    p_pos = (np.eye(n) - K @ c) @ p_pri @ (np.eye(n) - K @ c).T + K @ r @ K.T
    p_pos = (p_pos + p_pos.T)/2
    return x_pos, p_pos


class PID(object):
    def __init__(self, p_gain, i_gain, d_gain, bias):
        self.error_prior = 0
        self.integral_prior = 0
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.bias = bias

    def control(self, error, iteration_time):
        integral = self.integral_prior + error * iteration_time
        derivative = (error - self.error_prior) / iteration_time
        self.error_prior = error
        self.integral_prior = integral
        output = self.p_gain * error + self.i_gain * integral + self.d_gain * derivative + self.bias
        return output

    def reset(self):
        self.error_prior = 0
        self.integral_prior = 0


