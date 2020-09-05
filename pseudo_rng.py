import numpy as np
import matplotlib.pyplot as plt
import time


class PseudoRNG:
    def __init__(self, seed=None, a=21454747, b=578907041, c=6851689087):
        if seed is None:
            seed = time.time_ns() % c
        self._a = a
        self._b = b
        self.rand_max = self._c = c
        self._seed = seed
        self._r = self._seed

    def rand(self):
        self._r = (self._a * self._r + self._b) % self._c
        return self._r


RNG = PseudoRNG()


def rand():
    return RNG.rand()


def random():
    return RNG.rand() / RNG.rand_max


def randint(low: int, high: int):
    """high is unreachable."""
    if low >= high: raise ValueError("low should less than high.")
    return low + (RNG.rand() % (high - low))


def pseudo_gauss(mean=.0, std_deviation=1.0):
    std_gauss = np.sqrt(-2 * np.log(random())) * np.cos(2 * np.pi * random())
    return std_gauss * std_deviation + mean


def main():
    n = 100000
    arr = np.array([pseudo_gauss(0.5) for _ in range(int(n))])
    print(f'Among {n} samples:')
    print(f'  Standard deviation: {np.std(arr):.2f}')
    print(f'  Mean: {np.mean(arr):.2f}')
    plt.plot(arr[:1000], lw=0.4)
    plt.hlines(0.5, xmin=0, xmax=1000, colors='r', lw=2)
    plt.hlines([-0.5, 1.5], xmin=0, xmax=1000, colors='y', lw=2)
    plt.axis([0, 1000, -3.5, 4.5])

    plt.xlabel('Time step (first 1000 numbers)')
    plt.ylabel('Random variable X')
    plt.show()
    plt.hist(arr, bins=1000)
    plt.vlines(0.5, ymin=0, ymax=5e4, colors='r', lw=0.8)
    plt.vlines([-0.5, 1.5], ymin=0, ymax=5e4, colors='y', lw=0.8)
    plt.axis([-4.5, 5.5, 0, 50000])
    plt.gca().set_yticks([])
    plt.xlabel('Random variable X')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == '__main__':
    main()
