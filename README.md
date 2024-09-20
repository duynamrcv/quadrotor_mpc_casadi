# Neural-based Model Predictive Control for Trajectory tracking with Dynamical Disturbance

## Quadrotor Dynamics 
The full explanation of the quadrotor dynamics is presented in [here](https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf).

The quadrotor state space is described between the inertial frame $I$ and body frame $B$, as $\xi = \left[\begin{array}{cccc}p_{IB} & q_{IB} & v_{IB} & \omega_{B}\end{array}\right]^T$ corresponding to position $p_{IB} ∈ \mathbb{R}^3$, unit quaternion rotation on the rotation group $q_{IB} \in \mathbb{SO}(3)$ given $\left\Vert q_{IB}\right\Vert = 1$, velocity $v_{IB} \in \mathbb{R}^3$, and bodyrate $\omega_B \in \mathbb{R}^3$. The input modality is on the level of collective thrust $T_B = \left[\begin{array}{ccc}0 & 0 & T_{Bz} \end{array}\right]^T$ and body torque $\tau_B$ . From here on we drop the frame indices since they are consistent throughout the description. The dynamic equations are follows:
$$
\dot{p}=v \\
\dot{q}= \dfrac{1}{2}\Lambda(q)\left[\begin{array}{l}0\\\omega\end{array}\right] \\
\dot{v}=g+\dfrac{1}{m}R(q)T + f_d\\
\dot{\omega}=J^{-1}(\tau-\omega\times J\omega) + \tau_d
$$
where $\Lambda$ represents a quaternion multiplication, $R(q)$ the quaternion rotation, $m$ the quadrotor’s mass, and $J$ its inertia. $f_d$, $\tau_d$ are the translation and rotation disturbances, respectively.

The input space given by $T$ and $\tau$ is further decomposed into the single rotor thrusts $u =\left[T_1, T_2, T_3, T_4\right]^T$, where $T_i$ is the thrust at rotor $i \in \{1, 2, 3, 4\}$
$$
T=\left[\begin{array}{c}0\\0\\\sum{T_i}\end{array}\right]
$$
For $\times$ configuration quadrotor model
$$
\tau=\left[\begin{array}{c}l/\sqrt{2}(T_1-T_2-T_3+T_4)\\
                           l/\sqrt{2}(-T_1-T_2+T_3+T_4)\\
                           c_\tau(-T_1+T_2-T_3+T_4)\end{array}\right]
$$
with the quadrotor’s arm length $l$ and the rotor’s torque constant $c_\tau$. The quadrotor’s actuators limit the applicable thrust for each rotor, effectively constraining $T_i$ as:
$$
0\leq T_{min} \leq T_i \leq T_{max}
$$

