import matplotlib.pyplot as plt
from numpy import ndarray, dtype, float64
import numpy as np
from Bivariate_Distribution_Visual import BivariateNormal


class AssetGBM:
    """A Geometric Brownian Motion for a single asset.

    Public Attributes:
        - dt: How much time passes with each step
        - mu: The expected value of this asset
        - sigma: This asset's volatility
    """

    def __init__(
        self,
        weight: float,
        name: str,
        dt: float,
        mu: float = 0,
        sigma: float = 1,
    ) -> None:
        """Create a new GBM.

        Preconditions: dt > 0
        """
        self.weight = weight
        self.name = name
        self.dt = dt
        self.mu = mu
        self.sigma = sigma
        self.path = [(0, 1)]

    def step(self, z: float) -> float:
        """Take a step forward with fixed noise, <noise>.
        As well as return newest height."""
        step_factor = np.exp(
            (self.mu - (1 / 2) * (self.sigma**2)) * self.dt
            + self.sigma * np.sqrt(self.dt) * z
        )
        new_pos = self.path[-1][0] * step_factor
        self.path.append((self.path[-1][0] + self.dt, new_pos))
        return new_pos

    def simulate_paths(
        self, time: float = 1, n: int = 1
    ) -> ndarray[tuple[int, int], dtype[float64]]:
        """Return an array of <n> simulations of <time> duration."""

        steps = int(time / self.dt)
        paths = np.zeros((steps, n))
        paths[0] = 1

        for i in range(1, steps):
            z = np.random.normal(0, 1, n)
            for j in range(n):
                noise = self.sigma * (self.dt ** (1 / 2)) * z[j]
                step_factor = np.exp(
                    (self.mu - (1 / 2) * (self.sigma**2)) * self.dt + noise
                )

                paths[i, j] = paths[i - 1, j] * step_factor
        return paths


class MultiAssetGBM:
    """A GBM of multiple assets

    Public Attributes:
        - dt: How much time passes with each step
        - path: A collection of what this MultiAssetGBM was priced
                at for all valid moments in time.

    Private Attributes:
        - _assets: A collection of all the assets to be tracked
    """

    dt: float
    path: list[tuple[float, float]]
    _assets: list[AssetGBM]

    def __init__(self, dt: float) -> None:
        """Create a new MultiAssetGBM.

        Precondition: dt > 0
        """
        self.dt = dt
        self.path = [(0, 1)]
        self._assets = []

    def add_asset(self, ticker: str, weight: float, mu: float, sigma: float) -> bool:
        """Return True if this ticker is not already an asset
        and the weight was valid. Otherwise, do nothing and return False.

        Precondition: <ticker> is a valid stock ticker and weight > 0.
        """
        if weight <= 0 or ticker in self._assets:
            return False
        else:
            self._assets.append(AssetGBM(weight, ticker, mu, sigma))
            return True

    def step(self) -> None:
        """Take one step forward and add the newest position to self.path."""

        covs = self.get_correlation_matrix()
        z = np.random.multivariate_normal(np.zeros(len(self._assets)), covs)

        new_pos = 0
        for i in range(len(self._assets)):
            asset = self._assets[i]
            new_pos += asset.step(z[i])

        self.path.append((self.path[-1][0] + self.dt, new_pos))

    def run(self, n: int) -> None:
        """Take <n> steps forward."""
        for i in range(n):
            self.step()

    def cov(self, asset1: AssetGBM, asset2: AssetGBM) -> float:
        """Return the covariance between both Assets."""

        means = asset1.mu, asset2.mu
        var = asset1.sigma**1 / 2, asset2.sigma**1 / 2
        bvn = BivariateNormal(means, var)

        pairs = np.array([bvn.sample() for i in range(10000)])
        expected_xy = np.array([(p[0] * p[1]) for p in pairs])

        return expected_xy.mean() - (asset1.mu * asset2.mu)

    def _corr_row(self, i: int) -> list[float]:
        """Helper function to get_covariance_matrix,

        Return the covariance of the ith asset
        in self._assets with all the other assets.

        Including itself, in which case:
        Cov(asset_i, asset_j) = Var(asset_i).

        Precondition: 0 <= i <= len(self._assets) - 1
        """

        row_i = []
        asset1 = self._assets[i]
        for asset2 in self._assets:
            if asset2.name == asset1.name:
                row_i.append(1)
            else:
                row_i.append(self.cov(asset1, asset2) / (asset1.sigma * asset2.sigma))
        return row_i

    def get_correlation_matrix(self) -> list[list[float]]:
        """Return a matrix of covariances between all assets.
        Such that matrix[i, j] = Cov(ith asset, jth asset),
        with 0 <= i, j <= len(self._assets) - 1.
        """
        matrix = []
        for i in range(len(self._assets)):
            matrix.append(self._corr_row(i))
        return matrix

    def visualize(self) -> None:
        """Visualize the path of this MultiAssetGBM."""
        print(self.path)
        x = [point[0] for point in self.path]
        y = np.array([point[1] for point in self.path])
        plt.plot(x, y.flatten())
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()


if __name__ == "__main__":
    ma = MultiAssetGBM(1 / 252)
    ma.add_asset("test1", 1, 0.08, 0.125)
    ma.add_asset("test2", 0.5, 0.08, 0.1)
    ma.add_asset("test3", 0.25, 0.08, 0.15)
    ma.run(1000)
    ma.visualize()
