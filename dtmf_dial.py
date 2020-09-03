import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pseudo_rng import pseudo_gauss
from scipy import fftpack


class DTMF:
    LOW_FREQ = [697.0, 770.0, 852.0, 941.0]
    HIGH_FREQ = [1209.0, 1336.0, 1477.0, 1633.0]
    KEYPAD = [
        ['1', '2', '3', 'A'],
        ['4', '5', '6', 'B'],
        ['7', '8', '9', 'C'],
        ['*', '0', '#', 'D']
    ]
    KEYMAP = (lambda pad, low, high: {pad[i][j]: (l, h) for i, l in enumerate(low) for j, h in enumerate(high)})(
        KEYPAD, LOW_FREQ, HIGH_FREQ)

    def __init__(self, SR=44100.0, SNR_dB=5, dial_ms_range=(300, 500), interval_ms=500, volume=0.5):
        """
        Initialize dual-tone multi-frequency signaling encoder/decoder.

        :param SR:            sampling rate.
        :param SNR_dB:        dB of signal-to-noise ratio.
        :param dial_ms_range: milliseconds of dial time. Choose value in this range randomly.
        :param interval_ms:   milliseconds of interval time.
        :param volume:        volume value.
        """
        self.SR = SR
        self.SNR_dB = SNR_dB
        self.dial_ms_range = dial_ms_range
        self.interval_ms = interval_ms
        self.volume = volume

    @staticmethod
    def AWGN_func(signal, mean, devi):
        return signal + np.array([pseudo_gauss(mean, devi) for _ in range(signal.shape[0])])

    @staticmethod
    def _encodable(m):
        for letter in m:
            if letter not in DTMF.KEYMAP: return False
        return True

    def _ms_to_samples(self, ms):
        return int(ms / 1000 * self.SR)

    def _gen_tone(self, letter, ms=None):
        if ms is None: ms = np.random.randint(*self.dial_ms_range)
        samples = self._ms_to_samples(ms)
        low, high = DTMF.KEYMAP[letter]
        tone = [0.5 * np.sin(2.0 * np.pi * i * low / self.SR) +
                0.5 * np.sin(2.0 * np.pi * i * high / self.SR)
                for i in range(samples)]
        return tone

    def encode(self, message, letter_ms=None, remove_noise=False):
        """
        Encode message to sound wave.

        :param message:      message to encode.
        :param letter_ms:    if None, random choose time from dial_ms_range.
        :param remove_noise: if remove noise.
        :return:             encoded sound wave.
        """
        if not self._encodable(message): raise ValueError(f'Message "{message}" is not encodable.')
        interval_samples = self._ms_to_samples(self.interval_ms)
        wave = np.zeros(interval_samples)
        tones = np.zeros(0)
        for m in message:
            tone = self._gen_tone(m, letter_ms)
            tones = np.concatenate([tones, tone])
            wave = np.concatenate([wave, tone, np.zeros(interval_samples)])

        AWGM_devi = tones.std() / (10 ** (self.SNR_dB / 20))
        print(f'Message: "{message}"    (SNR = {self.SNR_dB}dB)')
        print(f'  Dial wave (without silent interval) standard deviation: {tones.std():.2f}')
        print(f'  AWGM standard deviation: {AWGM_devi:.2f}')
        if not remove_noise: wave = self.AWGN_func(wave, 0, AWGM_devi)
        return self.volume * wave


def show_amplitude(wave, sampling_rate):
    X = fftpack.fft(wave)
    freq = fftpack.fftfreq(len(wave)) * sampling_rate

    fig, ax = plt.subplots()
    ax.stem(freq, np.abs(X), use_line_collection=True, markerfmt='C0,')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Frequency magnitude')
    ax.set_xlim(0, 2500)
    plt.show()


def _make_short_time_window(wave, window_size, shift):
    n_samples = wave.shape[0]
    n_windows = int(np.ceil((n_samples - window_size) / shift))
    window = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        window[i] = wave[i * shift:i * shift + window_size]
    return window


def plot_wave_amplitude(wave, sampling_rate, ax=None, hide_x=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 2), dpi=400)
    ax.plot(wave, color=(0.27, 0.87, 0.61, 1), lw=0.2)
    ax.set_facecolor((0, 0, 0))
    ax.set_xlim(0, len(wave) - 1)
    n_secs = int(len(wave) / sampling_rate) + 1
    ax.set_xticks(list(range(0, n_secs * int(sampling_rate), 2 * int(sampling_rate))))
    ax.grid(color=(0, 0.4, 0), which='both')
    if hide_x:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels([(lambda tick: f'{int(tick / sampling_rate)}')(t) for t in ax.get_xticks()])
        ax.set_xlabel('Time (second)')
    ax.set_ylabel('Amplitude')
    return ax


def plot_short_time_freq(wave, sampling_rate, window_size=1024, shift=100, display_max_freq=None, ax=None):
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

    display_max_idx = int(spec.shape[0] * min(display_max_freq / max_freq, 1))
    spec = spec[:display_max_idx]

    ax.imshow(spec, origin='lower', cmap='gist_heat',
              extent=(0, n_samples / sampling_rate, 0, display_max_freq))
    ax.axis('tight')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (second)')
    return ax


def construct_dtmf_experiment(message, SNR_dB):
    dial = DTMF(dial_ms_range=(300, 500), interval_ms=500, volume=0.2, SNR_dB=SNR_dB)
    wave = dial.encode(message, remove_noise=False)

    fig, (ax_up, ax_down) = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    plot_wave_amplitude(wave, dial.SR, ax=ax_up, hide_x=True)
    plot_short_time_freq(wave, dial.SR, display_max_freq=2500, ax=ax_down)
    ax_up.set_title(f'Message: "{message}"    (SNR = {SNR_dB}dB)')
    plt.show()


def main():
    construct_dtmf_experiment('13991156869', 3)
    construct_dtmf_experiment('13991156869', -5)
    construct_dtmf_experiment('13991156869', -10)


if __name__ == '__main__':
    main()
