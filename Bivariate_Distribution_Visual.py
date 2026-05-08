from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np


class MultiVariateNormal:
    """Generalized normal bell curve for multiple dimensions."""

    means: tuple[float]
    variances: tuple[float]

    def __init__(self, means: tuple[float], var: tuple[float]) -> None:
        self.means = means
        self.var = var

    def sample(self) -> tuple[float]:
        """Return one draw from this Distribution."""
        raise NotImplementedError

    def pmf(self, start: float, stop: float) -> float:
        """Return the probability that all RVs of this Distribution
        will like within <start> and <stop> inclusive."""
        raise NotImplementedError


class BivariateNormal(MultiVariateNormal):
    """Distribution of joint probabilities of
    two Normal(0, 1) Random Variables."""

    def __init__(
        self, means: tuple[float] = [0, 0], var: tuple[float] = [1, 1]
    ) -> None:
        """Precondition: means and var have exactly
        2 valid positive integers."""
        super().__init__(means, var)

    def sample(self) -> tuple[float, float]:
        """Return one pair of draws."""
        x = np.random.normal(self.means[0], np.sqrt(self.var[0]))
        y = np.random.normal(self.means[1], np.sqrt(self.var[1]))
        return x, y

    def sampling(self, n: int) -> np.ndarray:
        """Return an array of <n> samples"""
        return np.array([self.sample() for i in range(n)])

    def cov(self) -> float:
        """Return the covariance between both RVs."""
        pairs = np.array([self.sample() for i in range(1000)])
        expected_xy = np.array([(p[0] * p[1]) for p in pairs])
        return expected_xy.mean() - (self.means[0] * self.means[1])

    def rho(self):
        """Return the correlational coefficient Rho between both RVs."""
        cov_xy = self.cov()
        sigma1, sigma2 = np.sqrt(self.var[0]), np.sqrt(self.var[1])
        return cov_xy / (sigma1 * sigma2)

    def pdf(self, x: float, y: float) -> float:
        """Return the probability density of a sample being close to (x, y)."""
        mu1, mu2 = self.means[0], self.means[1]
        sigma1, sigma2 = self.var[0], self.var[1]
        rho = self.rho()

        z1 = (x - mu1) / sigma1
        z2 = (y - mu2) / sigma2

        denom = 2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - rho**2)
        exponent = -1 / (2 * (1 - rho**2)) * (z1**2 - 2 * rho * z1 * z2 + z2**2)

        return round(np.exp(exponent) / denom, 4)


def simulate_bivariate(
    means: tuple[float] = (0, 0), var: tuple[float] = (1, 1), n: int = 1000
) -> None:
    """Plot <n> samples of a multivariate distributions of variances
    in <var> its respective mean in <means>.

    Precondition: len(means) == len(var) == 2 and all elements are non-negative
    """
    bm = BivariateNormal(means, var)
    samples = bm.sampling(n)
    x, y = [sample[0] for sample in samples], [sample[1] for sample in samples]
    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    simulate_bivariate()
