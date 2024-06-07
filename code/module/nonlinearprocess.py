import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
from scipy.signal import find_peaks


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
    """
    Reads time data from a text file.

    Parameters:
    -----------
    - dir: Path to the text file containing time data.

    Returns:
    -----------
    - time_array: Array of time values.
    - vel_array: Array of velocity values corresponding to the time values.
    """
    # Define the columns to extract from the text file
    cols = (0, 1)

    # Open and read the contents of the text file
    with open(dir, 'r') as file:
        filedata = file.read()

    # Replace commas with dots and tabs with commas for proper formatting
    filedata = filedata.replace(',', '.').replace('\t', ',')

    # Write the modified data to a new file ('modified_data.txt')
    with open('modified_data.txt', 'w') as file:
        file.write(filedata)

    # Load the modified data from the new file into a NumPy array
    data = np.genfromtxt('modified_data.txt', delimiter=',', skip_header=7, usecols=cols)

    # Extract time and velocity arrays from the data
    time_array = data[:, 0]
    vel_array = data[:, 1]

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
    
    gamma = c/m

    """
    # Use find_peaks to identify the local maxima in the time response
    peaks_indices, _ = find_peaks(time_response)
    peaks_array = time_response[peaks_indices]
    peaks_time_array = time_array[peaks_indices]

    # Calculate the average damped period (T_d) from the time differences between consecutive peaks
    T_d = np.mean(np.diff(peaks_time_array))
    wd = 2 * np.pi / T_d

    # Calculate the logarithmic decrement
    amplitude_ratios = np.array(peaks_array[:-1]) / np.array(peaks_array[1:])
    log_decrements = np.log(np.abs(amplitude_ratios))
    gamma = 2 * np.mean(log_decrements) / T_d

    # Calculate the undamped natural frequency (w0)
    w0 = np.sqrt(wd**2 + (gamma / 2)**2)

    return gamma, w0
def lowpass_filter(t, signal_data, cutoff_freq, order=5):
    """
    Applies a Butterworth low-pass filter to the signal data.

    Parameters:
    -----------
    - t: array of time
    - signal_data: array of signal data
    - cutoff_freq: cut-off frequency of the low-pass filter (Hz)
    - order: order of the Butterworth filter (default is 5)

    Returns:
    -----------
    - signal_filtered: array of filtered signal data
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

def set_folder_name(main_dir, acc=0, time=True, mec=True):
    """
    Constructs the directory name based on the provided parameters.

    Parameters:
    -----------
    - main_dir: main directory where the directory will be created
    - acc: acceleration value (default is 0)
    - time: indicates if it's a time domain response (default is True)
    - mec: indicates if it's a mechanical response (default is True)

    Returns:
    -----------
    - name: constructed directory name based on the parameters
    """
    if time:
        # If it's a time domain response
        name = main_dir + "Time response/"
        if mec:
            # If it's a mechanical response
            name = name + "mec-time/time x velocity.txt"
        else:
            # If it's a magnetic response
            name = name + "mag-time/time x velocity.txt"
    else:
        # If it's a frequency domain response
        name = main_dir + "Frequency response/"
        if mec:
            # If it's a mechanical response
            name = name + "mec-frf/"
        else: 
            # If it's a magnetic response
            name = name + "mag-frf/"

        # Add the acceleration value to the directory name
        name = name + str(acc) + "g/" + str(acc) + "g.txt"

    return name


