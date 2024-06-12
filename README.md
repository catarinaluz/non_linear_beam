# Non-Linear-Beam

This repository contains the code necessary for extracting, visualizing, and analyzing experimental data to determine the parameters of a non-linear oscillator system. The non-linear oscillator in question is subject to both mechanical and magnetic forces, with the overall goal of fitting the experimental data to the derived model and extracting relevant physical parameters.

## Contents

- **Experimental Data**: Experimental data for the frequency response in several excitations and for the time response provinient of an impact test.
- **Module**: Tools for extracting the data, visualizing then and extracting the non-linear oscilator parameters.
- **Examples**: Notebook examples on how to use the module.

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
- $f_m$ is the modal force.
- $\Omega$ is the excitation frequency.
- $A_{acc}$ is the acceleration amplitude.


## Getting Started

1. **Clone the Repository**: 
```
git clone https://github.com/catarinaluz/non_linear_beam.git
```

2. **Import the functions**:
```
from non_linear_beam.code.module.nonlinearprocess import get_frequency_data, get_time_data, lowpass_filter, get_parameters, set_file_name, plot_fft, bandpass_filter, get_frf,perform_optimization
```

3. **Install Dependencies**:
Make sure you have Python and necessary libraries installed. You can install the required libraries using:
```
pip install requirements.txt
```


## Using Experimental Data

Now, you have the option to use either the experimental data provided or your own dataset. You can extract and visualize the data, as well as fit the curve parameters using the provided tools. Refer to the examples for detailed usage instructions.

For using experimental data provided:

1. Ensure you have cloned the repository and installed the necessary dependencies as outlined in the "Getting Started" section of the README.

2. Run the data extraction functions available to load and preprocess the experimental data.

3. Utilize the visualization tools in the to plot and inspect the experimental data to gain insights and identify patterns.

4. Apply the parameter estimation scripts in to fit the model to the experimental data and estimate the system parameters.

For using your own dataset:

1. Prepare your dataset in a compatible format (e.g., CSV, Excel).

2. Ensure the data meets the requirements of the data extraction scripts.

3. Follow the same steps outlined above for extracting, visualizing, and fitting the curve parameters using your own dataset.

Refer to the examples provided in the repository for more information on how to use each tool effectively.
