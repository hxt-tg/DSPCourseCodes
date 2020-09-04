import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack


def show_amplitude_freq_domain(wave, sampling_rate, freq_range=None):
    X = fftpack.fft(wave)
    freq = fftpack.fftfreq(len(wave)) * sampling_rate
    fig, ax = plt.subplots()
    ax.stem(freq, np.abs(X), use_line_collection=True, markerfmt='C0,')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Frequency magnitude')
    if freq_range is not None:
        ax.set_xlim(*freq_range)
    plt.show()


def _make_short_time_window(wave, window_size, shift):
    n_samples = wave.shape[0]
    n_windows = int(np.ceil((n_samples - window_size) / shift))
    window = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        window[i] = wave[i * shift:i * shift + window_size]
    return window


def plot_amplitude_time_domain(wave, sampling_rate, ax=None, hide_x=False, adobe_like=False):
    x_axis = np.linspace(0, len(wave) - 1, len(wave)) / sampling_rate
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2), dpi=400)
    if adobe_like:
        ax.plot(x_axis, wave, color=(0.27, 0.87, 0.61, 1), lw=0.2)
        ax.set_facecolor((0, 0, 0))
        ax.grid(color=(0, 0.4, 0), which='both')
    else:
        ax.plot(x_axis, wave, lw=2)
        # ax.grid(color=(0.1, 0.1, 0.1), which='both')
    ax.set_xlim(0, x_axis[-1])
    if hide_x:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (second)')
    ax.set_ylabel('Amplitude')
    return ax


def plot_short_time_freq_domain(wave, sampling_rate, window_size=1024, shift=100, freq_range=None, ax=None):
    n_samples = wave.shape[0]
    max_freq = sampling_rate / 2
    window = _make_short_time_window(wave, window_size, shift)
    window = (window * np.hanning(window_size + 1)[:-1]).T
    spectrum = np.fft.fft(window, axis=0)[:window_size // 2 + 1:-1]
    spectrum = np.abs(spectrum)[:, :-2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2))
    spec = np.abs(spectrum)

    # Comment for visualization performance
    # spec = 20 * np.log10(spec / np.max(spec))

    if freq_range is None: freq_range = (0, max_freq)
    freq_min = max(int(spec.shape[0] * freq_range[0] / max_freq), 0)
    freq_max = int(spec.shape[0] * min(freq_range[1] / max_freq, 1))
    spec = spec[freq_min:freq_max]
    if spec.shape[0] < 1:
        print('Warning: you may need adjust freq_range.')

    ax.imshow(spec, origin='lower', cmap='gist_heat',
              extent=(0, n_samples / sampling_rate, freq_range[0], freq_range[1]))
    ax.axis('tight')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (second)')
    return ax


def show_constellation_graph(baud2mod, axis_lim=None):
    data = [(r * np.cos(theta), r * np.sin(theta), lbl)
            for lbl, (r, theta) in baud2mod.items()]
    x, y, labels = zip(*data)
    plt.scatter(x, y, color='r', s=40)
    for x, y, t in data:
        plt.annotate(t, (x + 0.05, y - 0.05), ha='left', va='top')
    plt.gca().set_aspect('equal')
    if axis_lim:
        plt.axis(axis_lim)
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.grid(True)
    plt.gca().set_axisbelow(True)
    plt.show()


def show_IQs_on_constellation(IQ_list, axis_lim=None):
    I, Q = zip(*IQ_list)
    plt.scatter(I, Q, color='b', s=2)
    plt.gca().set_aspect('equal')
    if axis_lim:
        plt.axis(axis_lim)
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.grid(True)
    plt.gca().set_axisbelow(True)
    plt.show()
