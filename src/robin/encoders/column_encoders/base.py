import sys
from typing import Iterable, Optional

import polars as pl
from torch import Tensor

from robin.encoders.utils import inverse_token_frequency, tokenize

MAX_DECIMALS = sys.float_info.dig


class BaseEncoder:
    dtype = None
    encoding = None
    size = None
    _rounding_digits = None
    _token_weights = None

    def __init__(self, name: Optional[str] = None) -> None:
        raise NotImplementedError(
            "BaseEncoder is an abstract class. Please use a concrete encoder."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def read_polars(
        self, data: Iterable, name: Optional[str] = None
    ) -> pl.Series:
        if not isinstance(data, pl.Series):
            print(
                f"Attempting to convert data ({type(data)}) to polars Series."
            )
            data = pl.Series(data)

        return data

    def fit_and_encode(self, data: Iterable) -> Tensor:
        """Fit the encoder to the data.

        Args:
            data (Iterable): input data to fit to.

        Returns:
            Tensor: token weights.
        """
        raise NotImplementedError("Fit method not implemented.")

    def encode(self, data: Iterable) -> Tensor:
        """Encode the data. Returns encoded data and weights.

        Args:
            data (Iterable): input data to be encoded.

        Returns:
            tuple[Tensor, Tensor]: encoded data, token_weights.
        """
        raise NotImplementedError("Encode method not implemented.")

    def decode(self, data: Iterable) -> pl.Series:
        """Decode the data.

        Args:
            data (Iterable): input data to be decoded.

        Returns:
            pl.Series: decoded data.
        """
        raise NotImplementedError("Decode method not implemented.")

    def get_rounding(self, data: pl.Series) -> Optional[int]:
        """Learn the number of digits to round data to.

        Args:
            data (pl.Series):
                Data to learn the number of digits to round to.

        Returns:
            int or None:
                Number of digits to round to.
        """

        roundable_data = data.filter(data.is_finite())
        if len(roundable_data) == 0:
            name = data.name if data.name is not None else "unknown"
            print(
                f"No finite data found for column '{data.name}'. Cannot learn rounding scheme."
            )
            return None

        # Try to round to fewer digits
        highest_int = int(roundable_data.abs().max())
        most_digits = len(str(highest_int)) if highest_int != 0 else 0
        max_decimals = max(0, MAX_DECIMALS - most_digits)
        if (roundable_data == roundable_data.round(max_decimals)).all():
            for decimal in range(max_decimals + 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        # Can't round, not equal after MAX_DECIMALS digits of precision
        name = data.name if data.name is not None else "unknown"
        print(
            f"No rounding scheme detected for column '{name}'. Data will not be rounded."
        )
        return self._token_weights


class MinMaxEncoder(BaseEncoder):

    def __init__(
        self,
        name: Optional[str] = None,
        verbose: bool = False,
        learn_rounding: bool = False,
        **kwargs,
    ):
        """ContinuousEncoder is used to encode continuous data to a range between -1 and 1.

        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or float.
        """

        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.encoding = "continuous"
        self.slot_size = 1
        self.size = 1

    def fit_and_encode(self, data: Iterable) -> Tensor:

        if not data.dtype.is_numeric():
            raise UserWarning(
                "ContinuousEncoder only supports numeric data types."
            )

        self.mini = data.min()
        self.maxi = data.max()
        self.range = self.maxi - self.mini
        self.mean = data.mean()
        self.std = data.std()
        if self.range == 0:
            raise UserWarning("Data has no range. Cannot encode.")
        self.dtype = data.dtype

        if self._learn_rounding:
            self._rounding_digits = self.get_rounding(data)

        return self.encode(data)

    def __str__(self):
        return f"{self.__class__.__name__}: ({self.name}) min: {self.mini}, max: {self.maxi}, range: {self.range}, dtype: {self.dtype}"

    def encode(self, data: Iterable) -> Tensor:
        data = Tensor(data).unsqueeze(-1)
        return (2 * (data - self.mini) / self.range) - 1

    def decode(self, data: Iterable) -> pl.Series:
        data = pl.Series(data.squeeze(-1))
        data = ((data + 1.0) * self.range / 2.0) + self.mini
        data = data.cast(self.dtype)
        if self._learn_rounding and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)
        return data


class StandardScalerEncoder(BaseEncoder):

    def __init__(
        self,
        name: Optional[str] = None,
        verbose: bool = False,
        learn_rounding: bool = False,
        **kwargs,
    ):

        self.name = name
        self.verbose = verbose
        self._learn_rounding = learn_rounding
        self.encoding = "continuous"
        self.slot_size = 1
        self.size = 1

    def fit_and_encode(self, data: Iterable) -> Tensor:

        if not data.dtype.is_numeric():
            raise UserWarning(
                "StandardScalerEncoder only supports numeric data types."
            )

        self.mean = data.mean()
        self.std = data.std()
        if self.std == 0:
            raise UserWarning("Data has no variance. Cannot encode.")
        self.dtype = data.dtype

        if self._learn_rounding:
            self._rounding_digits = self.get_rounding(data)

        return self.encode(data)

    def __str__(self):
        return f"{self.__class__.__name__}: ({self.name}) mean: {self.mean}, std: {self.std}, dtype: {self.dtype}"

    def encode(self, data: Iterable) -> Tensor:
        data = Tensor(data).unsqueeze(-1)
        return 0.25 * (data - self.mean) / self.std

    def decode(self, data: Iterable) -> pl.Series:
        data = pl.Series(data.squeeze(-1))
        data = (data * self.std) * 4 + self.mean
        data = data.cast(self.dtype)
        if self._learn_rounding and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)
        return data


class CategoricalTokeniser(BaseEncoder):
    def __init__(
        self,
        name: Optional[str] = None,
        verbose: bool = False,
        use_token_weights: bool = False,
        **kwargs,
    ):
        """CategoricalEncoder is used to encode categorical data as integers from 0 to N.
        Where N is the number of unique categories.
        Args:
            data (Iterable): input data to be encoded
            name (str, optional): name of the encoder. Defaults to None.
            verbose (bool, optional): print the encoder configuration. Defaults to False.
            use_token_weights (bool, optional): whether to use token weights based on inverse token frequency. Defaults to False.
        Raises:
            UserWarning: If the data is not of type int or object.
        """
        self.name = name
        self.verbose = verbose
        self.encoding = "categorical"
        self.slot_size = 1
        self.use_token_weights = use_token_weights

    def fit_and_encode(self, data: Iterable) -> Tensor:

        self.dtype = data.dtype
        encoded, self.mapping = tokenize(data)

        if self.use_token_weights:
            self._token_weights = inverse_token_frequency(encoded)
            if self.verbose:
                print(
                    f">>> CategoricalTokeniser: Using token weights for {self.name} with shape {self._token_weights.shape} <<<"
                )

        self.size = len(self.mapping)

        if self.verbose:
            if self.size > 20:
                print(
                    f">>> Warning: {self} has more than 20 categories ({self.size})). <<<"
                )
        return encoded.unsqueeze(-1)

    def __str__(self):
        if self.verbose:
            return f"{self.__class__.__name__}: ({self.name}) size: {self.size}, categories: {self.mapping}, dtype: {self.dtype}"
        return f"{self.__class__.__name__}: ({self.name}) size: {self.size}"

    def encode(self, data: Iterable) -> Tensor:
        return tokenize(data, self.mapping)[0].unsqueeze(-1)

    def decode(self, data: Iterable, safe: bool = True) -> pl.Series:
        data = pl.Series(data.squeeze(-1)).cast(pl.Int8)
        reverse_mapping = {v: k for k, v in self.mapping.items()}
        if safe:
            missing = set(data.unique()) - set(reverse_mapping.keys())
            if missing:
                raise UserWarning(
                    f"Missing categories in data: {missing}. Please check your encoding."
                )
        data = data.replace_strict(reverse_mapping, return_dtype=self.dtype)
        return data
