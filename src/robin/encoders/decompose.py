import warnings
from typing import Iterable, Optional

import numpy as np
import polars as pl
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from torch import Tensor

from robin.encoders.base import BaseEncoder
from robin.encoders.utils import inverse_token_frequency


class LogTransform:
    """z = log(x + c); inverse: x = exp(z) - c"""

    def __init__(self, offset=0.0):
        self.c = float(offset)

    def fit(self, X):
        if np.min(X) + self.c <= 0:
            raise ValueError("log requires X + offset > 0 for all entries.")
        return self

    def encode(self, X):
        return np.log(X + self.c)

    def decode(self, Z):
        return np.exp(Z) - self.c

    def log_abs_det_jacobian(self, X):
        return -np.log(X + self.c).sum(axis=1)


class YeoJohnsonTransform:
    """Uses sklearn PowerTransformer (yeo-johnson)."""

    def __init__(self):
        self.pt = PowerTransformer(
            method="yeo-johnson", standardize=False
        )  # keep scaling separate if desired

    def fit(self, X):
        self.pt.fit(X)
        return self

    def encode(self, X):
        return self.pt.transform(X)

    def decode(self, Z):
        return self.pt.inverse_transform(Z)

    def log_abs_det_jacobian(self, X):
        eps = 1e-6
        n, d = X.shape
        log_det = np.zeros(n)
        for j in range(d):
            x = X[:, [j]]
            z = self.encode(x)  # shape (n,1)
            # central difference on f w.r.t. x_j
            x_plus = x + eps
            x_minus = x - eps
            z_plus = self.encode(x_plus)
            z_minus = self.encode(x_minus)
            dzdx = (z_plus - z_minus) / (2 * eps)
            log_det += np.log(np.abs(dzdx[:, 0]) + 1e-15)
        return log_det


class QuantileNormalTransform:
    """Approximately invertible Gaussianizer via quantile mapping to N(0,1)."""

    def __init__(self, n_quantiles=1000, subsample=10_000, random_state=0):
        self.qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            subsample=subsample,
            random_state=random_state,
        )

    def fit(self, X):
        self.qt.fit(X)
        return self

    def encode(self, X):
        return self.qt.transform(X)

    def decode(self, Z):
        return self.qt.inverse_transform(Z)

    def log_abs_det_jacobian(self, X):
        raise NotImplementedError(
            "QuantileNormalTransform does not implement log|J|."
        )


class Standardize:
    def __init__(self):
        self.ss = StandardScaler(with_mean=True, with_std=True)

    def fit(self, X):
        self.ss.fit(X)
        return self

    def encode(self, X):
        return self.ss.transform(X)

    def decode(self, Z):
        return self.ss.inverse_transform(Z)

    def log_abs_det_jacobian(self, X):
        # Linear transform z = (x-mu)/sigma => |det J| = prod(1/sigma_j)
        # log|det J| is constant per row
        s = self.ss.scale_
        return np.full(X.shape[0], -np.sum(np.log(np.abs(s) + 1e-15)))


class MetaDeecomposer(BaseEncoder):
    tranformers = [
        None,
        Standardize,
        LogTransform,
        YeoJohnsonTransform,
        QuantileNormalTransform,
    ]

    def __init__(
        self,
        data: pl.Series,
        name: str,
        max_components: int = 10,
        verbose: bool = False,
        learn_rounding=False,
        enforce_min_max=False,
        use_token_weights: bool = False,
        max_iter: int = 100,
        weight_threshold=0.005,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.enforce_min_max = enforce_min_max
        self.use_token_weights = use_token_weights
        self.max_iter = max_iter
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.seed = seed if seed is not None else 12345
        self.size = None
        self.encoding = "decomposed"
        self.slot_size = 2

    def fit_and_encode(self, data: pl.Series) -> Tensor:
        best_likelihood = -np.inf
        best_encoded = None

        for transformer in self.transformers:
            self.transformer = transformer()
            try:
                self.transformer.fit(data.to_numpy().reshape(-1, 1))
                transformed_data = self.transformer.encode(
                    data.to_numpy().reshape(-1, 1)
                )
                transformed_series = pl.Series(transformed_data.flatten())
                if self.verbose:
                    print(
                        f">>> MetaDecomposer: Using {transformer.__name__} for {self.name} <<<"
                    )
                break
            except Exception as e:
                if self.verbose:
                    print(
                        f">>> MetaDecomposer: Transformer {transformer} failed with error: {e} <<<"
                    )

            gmm_encoder = GMMEncoder()
            encoded = gmm_encoder.fit_and_encode(data)
            score = gmm_encoder.score(data)
            # todo

        if self.use_token_weights:
            self._token_weights = inverse_token_frequency(encoded[:, 1].long())
            if self.verbose:
                print(
                    f">>> GMMEncoder: Using token weights for {self.name} with shape {self._token_weights.shape} <<<"
                )

        self.dtype = data.dtype
        self.size = sum(self.threshold_mask)

        return encoded


class GMMEncoder(BaseEncoder):
    """Encoder (Transformer) for numerical data using a Bayesian Gaussian Mixture Model.

    This transformation takes a numerical value and transforms it using a Bayesian GMM
    model. It generates two outputs, a discrete value which indicates the selected
    'component' of the GMM and a continuous value which represents the normalized value
    based on the mean and std of the selected component.

    Args:
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        max_clusters (int):
            The maximum number of mixture components. Depending on the data, the model may select
            fewer components (based on the ``weight_threshold``).
            Defaults to 10.
        weight_threshold (int, float):
            The minimum value a component weight can take to be considered a valid component.
            ``weights_`` under this value will be ignored.
            Defaults to 0.005.

    Attributes:
        transformer:
            An instance of sklearn`s ``BayesianGaussianMixture`` class.
        threshold_indices:
            An array indicating the valid components. If the weight of a component is greater
            than the ``weight_threshold``, it's indicated with True, otherwise it's set to False.
    """

    def __init__(
        self,
        data: pl.Series,
        name: str,
        max_components: int = 10,
        verbose: bool = False,
        learn_rounding=False,
        enforce_min_max=False,
        use_token_weights: bool = False,
        max_iter: int = 100,
        weight_threshold=0.005,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.enforce_min_max = enforce_min_max
        self.use_token_weights = use_token_weights
        self.max_iter = max_iter
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.seed = seed if seed is not None else 12345
        self.size = None

        self.encoding = "decomposed"
        self.slot_size = 2

    def fit_and_encode(self, data: pl.Series) -> Tensor:
        self._fit(data)
        encoded = self.encode(data)
        if self.use_token_weights:
            self._token_weights = inverse_token_frequency(encoded[:, 1].long())
            if self.verbose:
                print(
                    f">>> GMMEncoder: Using token weights for {self.name} with shape {self._token_weights.shape} <<<"
                )

        self.dtype = data.dtype
        self.size = sum(self.threshold_mask)

        return encoded

    def __str__(self):
        return f"{self.__class__.__name__}: ({self.name}) {self.size}/{self.max_components} components."

    def _fit(self, data: pl.Series):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.transformer = BayesianGaussianMixture(
            n_components=self.max_components,
            max_iter=self.max_iter,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=None,
            random_state=self.seed,
        )

        if self.enforce_min_max:
            self._min_value = data.min()
            self._max_value = data.max()

        if self._learn_rounding:
            self._rounding_digits = self.get_rounding(data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer.fit(data.to_numpy().reshape(-1, 1))

        self.threshold_mask = self.transformer.weights_ > self.weight_threshold

    def score(self, data: pl.Series) -> float:
        """Calculate the Log Likelihood for the fitted transformation.
        Args:
            data (pl.Series):
                Data to calculate the log likelihood for.

        Returns:
            float: The Log Likelihood value.
        """
        data = data.to_numpy().reshape(-1, 1)
        return self.transformer.score(X=data)

    def encode(self, data: pl.Series) -> Tensor:
        """Transform the numerical data.

        Args:
            data (pl.Series):
                Data to transform.

        Returns:
            torch.Tensor.
        """
        data = data.to_numpy().reshape(-1, 1)

        # data = data.reshape((len(data), 1))
        means = self.transformer.means_.reshape((1, self.max_components))
        means = means[:, self.threshold_mask]
        vars = self.transformer.covariances_.reshape((1, self.max_components))
        stds = np.sqrt(vars)
        stds = stds[:, self.threshold_mask]

        # Multiply stds by 4 so that a value will be in the range [-1,1] with 99.99% probability
        normalized_values = (data - means) / (4 * stds)
        component_probs = self.transformer.predict_proba(data)
        component_probs = component_probs[:, self.threshold_mask]
        component_probs = (component_probs + 1e-6) / component_probs.sum(
            axis=1, keepdims=True
        )

        r = np.expand_dims(np.random.rand(len(data)), axis=1)
        selected_component = (data.cumsum(axis=1) > r).argmax(axis=1)

        aranged = np.arange(len(data))
        normalized = normalized_values[aranged, selected_component]
        normalized = normalized.reshape([-1, 1])
        normalized = np.clip(normalized, -0.99, 0.99)
        normalized = normalized[:, 0]
        rows = [normalized, selected_component]
        encoded = np.stack(rows, axis=1)
        encoded = Tensor(encoded)
        assert encoded.shape[1] == 2
        return encoded

    def decode(self, data: Iterable) -> pl.Series:
        """Convert data back into the original format.

        Args:
            data (Iterable): Data to transform.

        Returns:
            pl.Series.
        """
        assert data.shape[1] == 2
        data = np.array(data)

        means = self.transformer.means_.reshape([-1])
        stds = np.sqrt(self.transformer.covariances_).reshape([-1])

        # first col [:,0] is component value, second col [:,1] is component index
        selected_component = data[:, 1]
        selected_component = selected_component.round().astype(int)
        selected_component = selected_component.clip(
            0, self.threshold_mask.sum() - 1
        )

        normalized = np.clip(data[:, 0], -1, 1)

        std_t = stds[self.threshold_mask]
        std_t = std_t[selected_component]
        mean_t = means[self.threshold_mask][selected_component]
        decoded = normalized * 4 * std_t + mean_t

        if self.enforce_min_max:
            decoded = decoded.clip(self._min_value, self._max_value)

        if self._learn_rounding and self._rounding_digits is not None:
            decoded = decoded.round(self._rounding_digits)

        return pl.Series(decoded)
