import numpy as np
import matplotlib.pyplot as plt
from pseudo_rng import pseudo_gauss
from scipy import fftpack
from utils import *


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


def construct_dtmf_experiment(message, SNR_dB):
    dial = DTMF(dial_ms_range=(300, 500), interval_ms=500, volume=0.2, SNR_dB=SNR_dB)
    wave = dial.encode(message, remove_noise=False)
    fig, (ax_up, ax_down) = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    plot_amplitude_time_domain(wave, dial.SR, ax=ax_up, hide_x=True)
    plot_short_time_freq_domain(wave, dial.SR, freq_range=(0, 2500), ax=ax_down)
    ax_up.set_title(f'Message: "{message}"    (SNR = {SNR_dB}dB)')
    plt.show()


def main():
    construct_dtmf_experiment('13991156869', 3)
    construct_dtmf_experiment('13991156869', -5)
    construct_dtmf_experiment('13991156869', -10)


if __name__ == '__main__':
    main()
