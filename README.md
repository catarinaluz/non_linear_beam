# non-linear-beam

## Model

Beam + Magnetic Interaction

$$
C\dot{\eta_n} + f_m \ddot{B} + (k_{l_{mec}} + k_{l_{mag}})\eta_n + (k_{nl_{mec}} + k_{nl_{mag}})\eta_n^3 + m\ddot{\eta_n} = 0
$$

Where:

$$
m = 1
$$

$$
k_{l_{mec}} = \omega_0^2
$$

Note: In the absence of magnetic forces, we can associate $\zeta$ with $C$

$$
C = 2 \zeta \omega_0
$$

Taking:

$$
w_n = \eta_n \psi_n
$$

We have:

$$
C\dot{\left(\frac{w_n}{\psi_n}\right)} + f_m \ddot{B} + (k_{l_{mec}} + k_{l_{mag}}) \frac{w_n}{\psi_n} + (k_{nl_{mec}}+ k_{nl_{mag}}) \left(\frac{w_n}{\psi_n}\right)^3 + m \ddot{\left(\frac{w_n}{\psi_n}\right)} = 0
$$

For a given $p_0$:

$$
C\dot{\left(\frac{w_n}{\psi_n}\right)} \bigg|_{x=p_0} + f_m \ddot{B} \bigg|_{x=p_0} + (k_{l_{mec}} + k_{l_{mag}}) \frac{w_n}{\psi_n} \bigg|_{x=p_0} + (k_{nl_{mec}} + k_{nl_{mag}}) \left(\frac{w_n}{\psi_n}\right)^3 \bigg|_{x=p_0} + m \ddot{\left(\frac{w_n}{\psi_n}\right)} \bigg|_{x=p_0} = 0
$$

Therefore, the equation to be fitted:

$$
C\dot{\eta_n} + f_m \ddot{B} + (k_{l_{mec}} + k_{l_{mag}})\eta_n + (k_{nl_{mec}} + k_{nl_{mag}})\eta_n^3 + m\ddot{\eta_n} = A_{acc}\cos(\Omega t)
$$

Where:
- $C$ is the damping coefficient.
- $k_{l_{mec}}$ and $k_{l_{mag}}$ are the linear elastic constants for mechanical and magnetic forces, respectively.
- $k_{nl_{mec}}$ and $k_{nl_{mag}}$ are the nonlinear elastic constants for mechanical and magnetic forces, respectively.
- $f_m$ is the magnetic force.
- $\Omega$ is the excitation frequency.
- $A_{acc}$ is the acceleration amplitude.


