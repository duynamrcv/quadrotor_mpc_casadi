import os
import casadi as ca
import numpy as np
from quadrotor import Quadrotor
from utils import skew_symmetric, v_dot_q, quaternion_inverse


class Controller:
    def __init__(self, quad:Quadrotor, n_nodes=20, dt=0.1):
        """
        :param quad: quadrotor object
        :type quad: Quadrotor3D
        :param n_nodes: number of optimization nodes until time horizon
        """

        self.N = n_nodes    # number of control nodes within horizon
        self.dt = dt        # time step
        self.x_dim = 13
        self.u_dim = 4

        self.opti = ca.Opti()
        self.opt_states = self.opti.variable(self.x_dim, self.N+1)
        self.opt_controls = self.opti.variable(self.u_dim, self.N)

        self.quad = quad

        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        # Declare model variables
        self.p = self.opt_states[:3,:]      # position
        self.q = self.opt_states[3:7,:]     # angle quaternion (wxyz)
        self.v = self.opt_states[7:10,:]    # velocity
        self.r = self.opt_states[10:13,:]   # angle rate

        f = lambda x_, u_: ca.vertcat(*[
            self.p_dynamics(x_),
            self.q_dynamics(x_),
            self.v_dynamics(x_, u_),
            self.w_dynamics(x_, u_)
        ])

        # Initial condition
        self.opt_x_ref = self.opti.parameter(self.x_dim, self.N+1)
        self.opti.subject_to(self.opt_states[:, 0] == self.opt_x_ref[:, 0])
        for i in range(self.N):
            x_next = self.opt_states[:,i] + f(self.opt_states[:,i], self.opt_controls[:,i])*self.dt
            self.opti.subject_to(self.opt_states[:,i+1] == x_next)

        # Weighted squared error loss function
        q_cost = np.diag([10, 10, 10, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        r_cost = np.diag([0.1, 0.1, 0.1, 0.1])

        # Cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[:,i] - self.opt_x_ref[:,i+1]
            obj = obj + ca.mtimes([state_error_.T, q_cost, state_error_]) \
                      + ca.mtimes([self.opt_controls[:,i].T, r_cost, self.opt_controls[:,i]])
        self.opti.minimize(obj)

        self.opti.subject_to(self.opti.bounded(self.min_u, self.opt_controls[0,:], self.max_u))
        self.opti.subject_to(self.opti.bounded(self.min_u, self.opt_controls[1,:], self.max_u))
        self.opti.subject_to(self.opti.bounded(self.min_u, self.opt_controls[2,:], self.max_u))
        self.opti.subject_to(self.opti.bounded(self.min_u, self.opt_controls[3,:], self.max_u))

        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,}
                        # 'ipopt.acceptable_tol':1e-8,
                        # 'ipopt.acceptable_obj_change_tol':1e-6

        self.opti.solver('ipopt', opts_setting)

    def p_dynamics(self, x):
        v = x[7:10]
        return v

    def q_dynamics(self, x):
        q = x[3:7]
        r = x[10:13]
        return 1 / 2 * ca.mtimes(skew_symmetric(r), q)

    def v_dynamics(self, x, u):
        q = x[3:7]
        f_thrust = u * self.quad.max_thrust
        g = ca.vertcat(0.0, 0.0, 9.81)
        a_thrust = ca.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.quad.mass

        v_dynamics = v_dot_q(a_thrust, q) - g
        return v_dynamics

    def w_dynamics(self, x, u):
        r = x[10:13]
        f_thrust = u * self.quad.max_thrust

        y_f = ca.MX(self.quad.y_f)
        x_f = ca.MX(self.quad.x_f)
        c_f = ca.MX(self.quad.z_l_tau)
        return ca.vertcat(
            ( ca.mtimes(f_thrust.T, y_f) + (self.quad.J[1] - self.quad.J[2]) * r[1] * r[2]) / self.quad.J[0],
            (-ca.mtimes(f_thrust.T, x_f) + (self.quad.J[2] - self.quad.J[0]) * r[2] * r[0]) / self.quad.J[1],
            ( ca.mtimes(f_thrust.T, c_f) + (self.quad.J[0] - self.quad.J[1]) * r[0] * r[1]) / self.quad.J[2])

    def compute_control_signal(self, x_ref):
        # Set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, x_ref)

        sol = self.opti.solve()

        u = sol.value(self.opt_controls)
        # x = sol.value(self.opt_states)
        return u[:,0]