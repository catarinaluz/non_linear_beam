import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
from scipy.optimize import minimize



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

    peaks_array = []
    peaks_time_array = []

    for i in range(1, len(time_response) - 1):
        if time_response[i] > time_response[i - 1] and time_response[i] > time_response[i + 1]:
            peaks_array.append(time_response[i])
            peaks_time_array.append(time_array[i])

    T_d = np.mean(np.diff(peaks_time_array))
    wd = 2 * np.pi / T_d

    amplitude_ratios = np.array(peaks_array[:-1]) / np.array(peaks_array[1:])
    log_decrements = np.log(abs(amplitude_ratios))
    gamma = 2 * np.mean(log_decrements) / T_d

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
    # Calculate the sampling rate from the time vector
    sample_rate = 1 / (t[1] - t[0])
    
    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sample_rate
    
    # Normalize the cutoff frequency with respect to the Nyquist frequency
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Create the Butterworth filter
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    
    # Apply the filter to the signal using filtfilt to avoid phase shift
    signal_filtered = signal.filtfilt(b, a, signal_data)
    
    return signal_filtered

def bandpass_filter(t, signal_data, lowcut_freq, highcut_freq, order=5):
    """
    Applies a Butterworth band-pass filter to the signal data.

    Parameters:
    -----------
    - t: array of time
    - signal_data: array of signal data
    - lowcut_freq: lower cut-off frequency of the band-pass filter (Hz)
    - highcut_freq: upper cut-off frequency of the band-pass filter (Hz)
    - order: order of the Butterworth filter (default is 5)

    Returns:
    -----------
    - signal_filtered: array of filtered signal data
    """
    # Calculate the sampling rate from the time vector
    sample_rate = 1 / (t[1] - t[0])
    
    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sample_rate
    
    # Normalize the cutoff frequencies with respect to the Nyquist frequency
    lowcut_normalized = lowcut_freq / nyquist_freq
    highcut_normalized = highcut_freq / nyquist_freq
    
    # Create the Butterworth band-pass filter
    b, a = signal.butter(order, [lowcut_normalized, highcut_normalized], btype='band', analog=False)
    
    # Apply the filter to the signal using filtfilt to avoid phase shift
    signal_filtered = signal.filtfilt(b, a, signal_data)
    
    return signal_filtered

def set_file_name(main_dir, acc=0, time=True, mec=True, test = 1):
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
            name = name + "mec-time/mec-t" 
        else:
            # If it's a magnetic response
            name = name + "mag-time/mag-t"
        name = name + str(test)
        name= name + "-velocity.txt"
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

def plot_fft(time, amplitude, freq_lim):
    """
    Plots the FFT spectrum of the given signal and returns the limited frequency and amplitude arrays.

    Parameters:
    -----------
    - time: array-like
        Time vector.
    - amplitude: array-like
        Signal amplitude data.
    - freq_lim: float
        Frequency limit for plotting the FFT spectrum.

    Returns:
    -----------
    - limit_freqs_mec: array-like
        Frequencies within the specified limit.
    - limit_fft_mec: array-like
        FFT amplitude values corresponding to the limited frequencies.
    """
    
    # Calculate the time interval
    delta_t = time[1] - time[0]
    
    # Calculate the FFT of the signal
    vel_fft_mec = np.fft.fft(amplitude)
    
    # Calculate the corresponding frequencies
    freqs_mec = np.fft.fftfreq(len(amplitude), d=delta_t)
    
    # Filter only the positive frequencies
    positive_freqs_mec = freqs_mec[freqs_mec >= 0]
    positive_fft_mec = np.abs(vel_fft_mec[freqs_mec >= 0])
    
    # Limit frequencies to the specified frequency limit
    limit_freqs_mec = positive_freqs_mec[positive_freqs_mec <= freq_lim]
    limit_fft_mec = positive_fft_mec[:len(limit_freqs_mec)]
    
    # Plot the FFT spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(limit_freqs_mec, limit_fft_mec)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT Spectrum of the Signal')
    plt.grid(True)
    plt.show()
    
    return limit_freqs_mec, limit_fft_mec

def frequency_response_module(freq,w, k_nl, amplitude, gamma, fm, acc):
    """
    Calculate the frequency response module.

    Parameters:
    -----------
    - freq: array of frequencies
    - k_linear: linear stiffness
    - k_nl: nonlinear stiffness
    - amplitude: array of amplitudes
    - gamma: damping coefficient
    - fm: modal force
    - acc: acceleration

    Returns:
    -----------
    - func: array of frequency response values
    """

    A = ( freq**2 - w**2 - 3/4 *k_nl*amplitude**2 )**2 + (gamma*freq)**2

    A = A*amplitude**2

    B = (fm * acc)**2

    func = np.abs(A-B)

    return func

def coef_eq(freq,beta,w,A,gamma):
    c_6 = (3*beta/(8*w))**2
    c_5 = 0
    c_4 = (3 / 4) * beta * (1 - freq / w)
    c_3 = 0
    c_2 = (freq-w)**2 +(gamma/2)**2
    c_1 = 0
    c_0 = -(freq**2*A/(2*w))**2

    return [c_6,c_5,c_4,c_3,c_2,c_1,c_0]

def get_frf(frequency_array, beta,w,A,gamma):
    a_ = []
    frequencias = []

    for i,f in enumerate(frequency_array):
        rad = 2*np.pi*f
        c = coef_eq(rad,beta,w,A,gamma)
        roots = np.roots(c)
        for raiz in roots:
            if raiz.imag == 0 and raiz.real > 0:
                a_.append(raiz.real)
                frequencias.append(f)

    return np.array(frequencias), np.array(a_)

def objective(params, freq_array, amp_array, w, gamma, acc):
    """
    Objective function to be minimized.

    Parameters:
    -----------
    - params: array of parameters [k_nl, fm]
    - freq_array: array of frequencies
    - amp_array: array of amplitudes
    - k_linear: linear stiffness
    - gamma: damping coefficient
    - acc: acceleration

    Returns:
    -----------
    - error: sum of the squared errors of the frequency response
    """
    k_nl, fm = params
    error = 0
    for i, f in enumerate(freq_array):
        amplitude = amp_array[i]
        error += frequency_response_module(f, w, k_nl, amplitude, gamma, fm, acc)
    return np.sum(error)


def optimization(w, gamma, freq_array, amp_array, acc, initial_guess):
    """
    Optimize to identify knl and fm.

    Parameters:
    -----------
    - k_linear: linear stiffness
    - gamma: damping coefficient
    - freq_array: array of frequencies
    - amp_array: array of amplitudes
    - acc: acceleration
    - initial_guess: initial guess for k_nl and fm

    Returns:
    -----------
    - result.x: optimized parameters (k_nl and fm)
    """
    # TODO: definir os bounds
    result = minimize(objective, 
                      initial_guess, 
                      args=(freq_array, amp_array, w, gamma, acc), 
                      bounds=[(0, None), (1e-5, 1e-4)],
                      method = None)
    return result.x
