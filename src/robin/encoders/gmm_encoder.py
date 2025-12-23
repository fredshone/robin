from typing import Iterable, Optional

import numpy as np
import polars as pl
from sklearn.mixture import BayesianGaussianMixture
from torch import Tensor

from robin.encoders.base import BaseEncoder


class GMMEncoder(BaseEncoder):
    """Transformer for numerical data using a Bayesian Gaussian Mixture Model.

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
        verbose: bool = False,
        learn_rounding=False,
        enforce_min_max=False,
        max_components=10,
        weight_threshold=0.005,
        seed: Optional[int] = None,
    ):
        self.learn_rounding = learn_rounding
        self.enforce_min_max = enforce_min_max
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.seed = seed if seed is not None else 12345
        self._fit(data)

        self.dtype = data.dtype
        self.encoding = "decomposed"
        self.size = sum(self.threshold_indices) + 1

        if verbose:
            print(
                f"{self.__class__.__name__}: ({name}) max_clusters: {self.max_components}, weight_threshold: {self.weight_threshold}"
            )

    def _fit(self, data: pl.Series):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.transformer = BayesianGaussianMixture(
            n_components=self.max_components,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.001,
            random_state=self.seed,
        )

        if self.enforce_min_max:
            self._min_value = data.min()
            self._max_value = data.max()

        if self.learn_rounding:
            self._rounding_digits = self.get_rounding(data)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        data = data.to_numpy().reshape(-1, 1)
        self.transformer.fit(data)  # .reshape(-1, 1)

        self.threshold_indices = (
            self.transformer.weights_ > self.weight_threshold
        )

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
        means = means[:, self.threshold_indices]
        vars = self.transformer.covariances_.reshape((1, self.max_components))
        stds = np.sqrt(vars)
        stds = stds[:, self.threshold_indices]

        # Multiply stds by 4 so that a value will be in the range [-1,1] with 99.99% probability
        normalized_values = (data - means) / (4 * stds)
        component_probs = self.transformer.predict_proba(data)
        component_probs = component_probs[:, self.threshold_indices]
        # component_probs = (component_probs + 1e-6) / component_probs.sum(
        #     axis=1, keepdims=True
        # )
        component_probs = component_probs / component_probs.sum(
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
        return Tensor(encoded)

    def decode(self, data: Iterable) -> pl.Series:
        """Convert data back into the original format.

        Args:
            data (Iterable): Data to transform.

        Returns:
            pl.Series.
        """
        data = np.array(data)

        means = self.transformer.means_.reshape([-1])
        stds = np.sqrt(self.transformer.covariances_).reshape([-1])

        # first col [:,0] is value, second col [:,1] is component
        normalized = np.clip(data[:, 0], -1, 1)
        selected_component = data[:, 1].round().astype(int)

        std_t = stds[self.threshold_indices][selected_component]
        mean_t = means[self.threshold_indices][selected_component]
        decoded = normalized * 4 * std_t + mean_t

        if self.enforce_min_max:
            data = data.clip(self._min_value, self._max_value)

        if self.learn_rounding and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)

        return pl.Series(decoded)
