import numpy as np
import matplotlib.pyplot as plt
import zipfile
import sympy

import scipy.signal as signal



def get_frequency_data(dir, sweep_up = True):
    """
    Reads data from a .txt file and returns the frequency and displacement amplitude.

    Parameters:
    -----------
    dir (str): Path to the .txt data file.
    sweep_up (bool): Indicates if the data is for a sweep up. If False, it's for a sweep down.

    Returns:
    -----------
    tuple: Two NumPy arrays, one with the frequency in Hz and the other with the displacement amplitude in meters.

    Explanation:
    -----------
    This function reads a text file containing tab-separated data.
    It replaces commas with dots to ensure correct interpretation of decimal values.
    Depending on the value of `sweep_up`, it selects the appropriate columns (frequency and velocity amplitude).
    It then converts the velocity amplitude to displacement amplitude and returns both arrays.

    Steps:
    -----------
    1. Open and read the data file.
    2. Replace commas with dots and tabs with commas for proper formatting.
    3. Write the modified data to a new file.
    4. Read the new file and extract the specified columns (frequency and velocity amplitude).
    5. Convert the velocity amplitude to displacement amplitude.
    6. Return the frequency and displacement amplitude arrays.
    """

    if sweep_up:
      cols = (0,1)
    else:
      cols = (0,3)

    with open(dir, 'r') as file:
      filedata = file.read()

    filedata = filedata.replace(',', '.').replace('\t', ',')

    with open('modified_data.txt', 'w') as file:
      file.write(filedata)

    data = np.genfromtxt('modified_data.txt', delimiter=',', skip_header=2, usecols=cols)

    frequency_array_hz = data[:,0]

    amplitude_vel = data[:,1]

    amplitude_disp = np.zeros(amplitude_vel.shape)

    for i, vel in enumerate(amplitude_vel):
      amplitude_disp[i] = vel/(frequency_array_hz[i]*2*np.pi)

    return frequency_array_hz, amplitude_disp

def get_time_data(dir):
    cols = (0,1)

    with open(dir, 'r') as file:
      filedata = file.read()

    filedata = filedata.replace(',', '.').replace('\t', ',')

    with open('modified_data.txt', 'w') as file:
      file.write(filedata)

    data = np.genfromtxt('modified_data.txt', delimiter=',', skip_header=6, usecols=cols)

    time_array = data[:,0]

    vel_array = data[:,1]

    return time_array, vel_array

def get_parameters(time_array, time_response):
    """
    Calculates the damping coefficient (gamma) and the undamped natural frequency (w0)
    from the time response of a damped harmonic oscillator.

    Parameters:
    -----------
    time_array : array-like
        The array of time values corresponding to the response measurements.
    
    time_response : array-like
        The array of response values (displacement, velocity, etc.) measured at the times specified
        in time_array.

    Returns:
    --------
    gamma : float
        The damping coefficient of the system, representing the rate of exponential decay of the oscillations.
    
    w0 : float
        The undamped natural frequency of the system, representing the frequency of oscillations in the absence
        of damping.

    Notes:
    ------
    This function assumes that the system is underdamped, meaning it exhibits oscillatory behavior. The peaks
    in the time response are identified and used to calculate the average damped period and the logarithmic
    decrement, which are then used to determine the damping coefficient and the undamped natural frequency.
    
    """

    peaks_array = []
    peaks_time_array = []

    for i in range(1, len(time_response) - 1):
        if time_response[i] > time_response[i - 1] and time_response[i] > time_response[i + 1]:
            peaks_array.append(time_response[i])
            peaks_time_array.append(time_array[i])

    T_d = np.mean(np.diff(peaks_time_array))
    wd = 2 * np.pi / T_d

    amplitude_ratios = np.array(peaks_array[:-1]) / np.array(peaks_array[1:])
    log_decrements = np.log(amplitude_ratios)
    gamma = 2 * np.mean(log_decrements) / T_d

    w0 = np.sqrt(wd**2 + (gamma / 2)**2)

    return gamma, w0

def lowpass_filter(t, signal_data, cutoff_freq, order=5):
    """
    Aplica um filtro passa-baixa Butterworth aos dados do sinal.

    Parâmetros:
    - t: array de tempo
    - signal_data: array de dados do sinal
    - cutoff_freq: frequência de corte do filtro passa-baixa (Hz)
    - order: ordem do filtro Butterworth (default é 5)

    Retorna:
    - signal_filtered: array de dados do sinal filtrado
    """
    # Calcular a taxa de amostragem a partir do vetor de tempo
    sample_rate = 1 / (t[1] - t[0])

    # Calcular a frequência de Nyquist
    nyquist_freq = 0.5 * sample_rate

    # Normalizar a frequência de corte em relação à frequência de Nyquist
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Criar o filtro Butterworth
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Aplicar o filtro ao sinal usando filtfilt para evitar defasagem
    signal_filtered = signal.filtfilt(b, a, signal_data)

    return signal_filtered

