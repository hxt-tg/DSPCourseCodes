import numpy as np
import matplotlib.pyplot as plt
from pseudo_rng import pseudo_gauss
from scipy import fftpack
from cmath import polar
from utils import *


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class QAM16:
    I_AMP = {'00': -3.0, '01': -1.0, '11': 1.0, '10': 3.0}
    Q_AMP = {'01': -3.0, '11': -1.0, '10': 1.0, '00': 3.0}
    BAUD2MOD = (lambda IA, QA: {I_baud + Q_baud: polar(I_amp + 1j * Q_amp)
                                for I_baud, I_amp in IA.items()
                                for Q_baud, Q_amp in QA.items()}
                )(I_AMP, Q_AMP)
    BITS_PER_BAUD = 4

    def __init__(self, SF=10e6, MF=2e6, SNR_dB=5, baud_width_ms=0.01):
        """
        Initialize dual-tone multi-frequency signaling modulator/demodulator.

        :param SF:            sampling frequency.
        :param MF:            Modulation frequency.
        :param SNR_dB:        dB of signal-to-noise ratio.
        :param baud_width_ms: milliseconds of one baud.
        """
        self.SF = SF
        self.MF = MF
        self.SNR_dB = SNR_dB
        self.baud_width_ms = baud_width_ms

    @staticmethod
    def AWGN_func(signal, mean, devi):
        return signal + np.array([pseudo_gauss(mean, devi) for _ in range(signal.shape[0])])

    @staticmethod
    def _modulatable(m):
        if len(m) % QAM16.BITS_PER_BAUD != 0: return False
        for baud in chunks(m, QAM16.BITS_PER_BAUD):
            if baud not in QAM16.BAUD2MOD: return False
        return True

    def _ms_to_samples(self, ms):
        return int(ms / 1000 * self.SF)

    def _gen_piece(self, baud):
        n_samples = self._ms_to_samples(self.baud_width_ms)
        amp, phase = QAM16.BAUD2MOD[baud]
        t = np.arange(n_samples) / self.SF
        piece = amp * np.sin(2.0 * np.pi * self.MF * t + phase)
        return piece

    def modulate(self, message, remove_noise=False, verbose=False):
        """
        Modulate message to microwave.

        :param message:      message (binary-chains) to modulate.
        :param remove_noise: if remove noise.
        :param verbose:      if print message info.
        :return:             modulated wave.
        """
        if not self._modulatable(message):
            raise ValueError(f'Message "{message}" is not encodable. '
                             f'Length should be divided by 4 and only contains 0 or 1.')
        wave = np.zeros(0)
        for baud in chunks(message, QAM16.BITS_PER_BAUD):
            wave = np.concatenate([wave, self._gen_piece(baud)])

        wave_std_deviation = wave.std()
        AWGM_devi = wave_std_deviation / (10 ** (self.SNR_dB / 20))
        if verbose:
            print(f'Message length: {len(message)} bits, {len(message) // QAM16.BITS_PER_BAUD} bauds'
                  f'    (SNR = {self.SNR_dB}dB)')
            print(f'  Wave standard deviation: {wave_std_deviation:.2f}')
            print(f'  AWGM standard deviation: {AWGM_devi:.2f}')
        if not remove_noise: wave = self.AWGN_func(wave, 0, AWGM_devi)
        return wave

    def _low_pass(self, wave, cutoff_freq=1.0):
        signal = fftpack.rfft(wave)
        freq = fftpack.fftfreq(len(wave)) * self.SF

        cut_signal = signal.copy()
        cut_signal[(np.abs(freq) > cutoff_freq)] = 0

        return fftpack.irfft(cut_signal)

    @staticmethod
    def _closest_baud(I_val, Q_val):
        closest_baud = None
        closest_distance = 1e10
        for I_baud, I_amp in QAM16.I_AMP.items():
            for Q_baud, Q_amp in QAM16.Q_AMP.items():
                if (dis := (I_val - I_amp) ** 2 + (Q_val - Q_amp) ** 2) < closest_distance:
                    closest_distance = dis
                    closest_baud = I_baud + Q_baud
        return closest_baud

    def _find_IQ(self, piece):
        t = np.arange(piece.size) / self.SF
        LO_I = 2 * np.sin(2.0 * np.pi * self.MF * t)
        LO_Q = 2 * np.cos(2.0 * np.pi * self.MF * t)
        return self._low_pass(piece * LO_I).mean(), self._low_pass(piece * LO_Q).mean()

    def demodulate(self, wave):
        """
        Demodulate message from microwave.

        :param wave:   microwave to demodulate.
        :return:       demodulated message (binary-chains) .
        """
        n_piece_samples = self._ms_to_samples(self.baud_width_ms)
        result = dict(
            message=[],
            IQ=[]
        )
        for piece in chunks(wave, n_piece_samples):
            I, Q = self._find_IQ(piece)
            result['message'].append(self._closest_baud(I, Q))
            result['IQ'].append((I, Q))
        return result


def string_hamming_distance(str1, str2, by_baud=False):
    if by_baud:
        return sum(1 for c1, c2 in zip(
            chunks(str1, QAM16.BITS_PER_BAUD),
            chunks(str2, QAM16.BITS_PER_BAUD)) if c1 != c2)
    else:
        return sum(1 for c1, c2 in zip(str1, str2) if c1 != c2)


def main_test_modulation(message):
    q = QAM16(10e6, 2e6, 10, baud_width_ms=0.01)
    show_constellation_graph(QAM16.BAUD2MOD, axis_lim=[-3.9, 3.9, -3.9, 3.9])

    wave = q.modulate(message, remove_noise=True, verbose=True)

    fig, (ax_up, ax_down) = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    plot_amplitude_time_domain(wave, q.SF, ax=ax_up, hide_x=True, adobe_like=True)
    plot_short_time_freq_domain(wave, q.SF, freq_range=(0, 2.5e6), ax=ax_down)
    ax_up.set_title(f'Message length: {len(message)} bits, {len(message) // QAM16.BITS_PER_BAUD} bauds'
                    f'    (SNR = {q.SNR_dB}dB)')
    plt.show()


def main_test_demodulation(message, prefix_bauds=10, suffix_bauds=8, prefix_IQ_graph=20):
    q = QAM16(10e6, 2e6, 10, baud_width_ms=0.01)
    wave = q.modulate(message)
    decoded_result = q.demodulate(wave)

    decoded_message = ''.join(decoded_result['message'])
    split = lambda s: ' '.join(chunks(s, QAM16.BITS_PER_BAUD))
    print(f'    Raw: {split(message[:prefix_bauds*QAM16.BITS_PER_BAUD])} ... '
          f'{split(message[-suffix_bauds*QAM16.BITS_PER_BAUD:])}')
    print(f'Decoded: {split(decoded_message[:prefix_bauds*QAM16.BITS_PER_BAUD])} ... '
          f'{split(decoded_message[-suffix_bauds*QAM16.BITS_PER_BAUD:])}')

    diff_bits = string_hamming_distance(message, decoded_message)
    diff_bauds = string_hamming_distance(message, decoded_message, True)
    print(f'[Demodulate result]\n'
          f' Error bits: {diff_bits}    '
          f'({diff_bits / len(message) * 100:.2f}%)')
    print(f'Error bauds: {diff_bauds}    '
          f'({diff_bauds / (len(message) // QAM16.BITS_PER_BAUD) * 100:.2f}%)')

    plt.title(f'SNR={q.SNR_dB} dB')
    show_IQs_on_constellation(decoded_result['IQ'], axis_lim=[-3.9, 3.9, -3.9, 3.9])
    show_decoded_IQ_waves(q, message[:prefix_IQ_graph * QAM16.BITS_PER_BAUD],
                       wave[:prefix_IQ_graph*q._ms_to_samples(q.baud_width_ms)])


# noinspection PyProtectedMember
def show_decoded_IQ_waves(qam16, message, wave):
    wave_I = np.zeros(0)
    wave_Q = np.zeros(0)
    n_piece_samples = qam16._ms_to_samples(qam16.baud_width_ms)
    t = np.arange(n_piece_samples) / qam16.SF
    const = 0 * t + 1

    for piece in chunks(wave, n_piece_samples):
        I, Q = qam16._find_IQ(piece)
        wave_I = np.concatenate([wave_I, I * const])
        wave_Q = np.concatenate([wave_Q, Q * const])

    fig, (ax_up, ax_down) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
    plot_amplitude_time_domain(wave_I, qam16.SF, ax=ax_up, hide_x=True)
    plot_amplitude_time_domain(wave_Q, qam16.SF, ax=ax_down)

    bauds = list(chunks(message, QAM16.BITS_PER_BAUD))
    ax_up.set_title(f'Amplitude wave of I and Q   '
                    f'(first {len(bauds)} bauds, '
                    f'SNR={qam16.SNR_dB} dB)\n'
                    'Message: ' + ' '.join(bauds[:10]) +
                    (' ...' if len(bauds) > 10 else ''))
    ax_up.set_ylabel('Amplitude (I)')
    ax_down.set_ylabel('Amplitude (Q)')
    ax_down.set_xlabel('Time (s)')
    plt.show()


def main():
    n_bauds = 512
    message = list('0' * n_bauds * 2 + '1' * n_bauds * 2)
    np.random.shuffle(message)
    message = ''.join(message)

    main_test_modulation(message)
    main_test_demodulation(message)


if __name__ == '__main__':
    main()
