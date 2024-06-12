# Non-Linear-Beam

This repository contains the code necessary for extracting, visualizing, and analyzing experimental data to determine the parameters of a non-linear oscillator system. The non-linear oscillator in question is subject to magnetic forces, with the overall goal of fitting the experimental data to the derived model and extracting relevant physical parameters.

## Contents

- **Experimental Data**: Experimental data for the frequency response in several excitations and for the time response provinient of an impact test.
- **Module**: Tools for extracting the data, visualizing then and extracting the non-linear oscilator parameters.
- **Examples**: Notebook examples on how to use the module.

## Directory Structure

- **code**: Contains the primary codebase.
  - **examples**: Jupyter notebooks demonstrating the usage of the module.
  - **module**: Includes the core module for data extraction and analysis.
    - `__init__.py`
    - `nonlinearprocess.py`
  - `requirements.txt`: Lists the dependencies required to run the code.

- **Experimental Data**: Holds the experimental data used for analysis.
  - **Frequency response**:
    - **mag-frf**: Frequency response data for different excitation levels.
      - `0.04g`, `0.06g`, `0.08g`, `0.10g`, `0.12g`
    - **mec-frf**: Mechanical frequency response data for different excitation levels.
      - `0.04g`, `0.06g`, `0.08g`, `0.10g`, `0.12g`, `0.14g`

  - **Time response**:
    - **mag-time**: Time response data from 3 impact tests for the case with magnetic force.
    - **mec-time**: Time response data from 3 impact tests for the case without magnetic force
      
## Model

Beam + Magnetic Interaction

$$
 m\ddot{\eta_n} + C\dot{\eta_n} + (k_{l_{mec}} + k_{l_{mag}})\eta_n + (k_{nl_{mec}} + k_{nl_{mag}})\eta_n^3  = - f_m \ddot{B} 
$$

After some mathematical manipulations, for a given point $p_0$, the equation to be fitted from experimental data is:

$$
\frac{d^2}{dt^2} w_{n,p_0}(t) + C \frac{d}{dt} w_{n,p_0}(t) + \left( \omega_0^2 + k_{l_{mag}} \right) w_{n,p_0}(t) + \left( k_{nl_{mag}}^{p_0} + k_{nl_{mec}}^{p_0} \right) w_{n,p_0}^3(t) = A_{acc} f_m^{p_0} \cos (\Omega t)
$$

Where:
- $w_{n,p_0}(t)$ is the vertical displacement.
- $C$ is the damping coefficient.
- $k_{l_{mec}}$ and $k_{l_{mag}}$ are the linear elastic constants for mechanical and magnetic forces, respectively.
- $k_{nl_{mec}}$ and $k_{nl_{mag}}$ are the nonlinear elastic constants for mechanical and magnetic forces, respectively.
- $f_m$ is the modal force.
- $\Omega$ is the excitation frequency.
- $A_{acc}$ is the acceleration amplitude.
- $\omega_n$ is the natural frequency.

Therefore, the main goal is to identify the values of $f_m$, $\omega_n $, $ C $ and $k_{nl}$ from experimental data.

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

3. Utilize the visualization tools to plot and inspect the experimental data to gain insights and identify patterns.

4. Apply the parameter estimation scripts in to fit the model to the experimental data and estimate the system parameters.

For using your own dataset:

1. Prepare your dataset in a compatible format.

2. Ensure the data meets the requirements of the data extraction scripts.

3. Follow the same steps outlined above for extracting, visualizing, and fitting the curve parameters using your own dataset.

Refer to the examples provided in the repository for more information on how to use each tool effectively.
