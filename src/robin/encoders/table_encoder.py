from typing import List, Optional, Union

import pandas as pd
import pandas.api.types as ptypes
import polars as pl
from torch import Tensor, cat

from robin.encoders.base import (
    CategoricalTokeniser,
    MinMaxEncoder,
    StandardScalerEncoder,
)
from robin.encoders.decompose import GMMEncoder
from robin.encoders.table_datasets import XDataset


class TableEncoder:

    continuous_encoders = {
        "minmax": MinMaxEncoder,
        "standard": StandardScalerEncoder,
        "decomposed": GMMEncoder,
    }

    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame],
        include: Optional[list] = None,
        exclude: Optional[list] = None,
        verbose: bool = False,
        continuous_encoding: str = "minmax",
        max_components: Optional[int] = None,
        learn_rounding_scheme: bool = False,
        enforce_min_max_values: bool = False,
        use_token_weights: bool = False,
    ):
        """Encode a dataframe into a Tensor,
        and initialise mapping for further encoding and decoding.
        Args:
            data (Union[pl.DataFrame, pd.DataFrame]): input dataframe to tokenise.
            include (list, optional): columns to include. Defaults to None.
            exclude (list, optional): columns to exclude. Defaults to None.
            verbose (bool, optional): print the configuration. Defaults to False.
            continuous_encoding (str, optional): continuous encoding scheme. Defaults to "minmax".
            max_components (int, optional): maximum number of GMM components. Defaults to None.
            learn_rounding_scheme (bool, optional): learn rounding scheme for continuous columns. Defaults to False.
            enforce_min_max_values (bool, optional): enforce min and max values for continuous columns.
            use_token_weights (bool, optional): enable token weighting based on inverse token frequency. Defaults to False.
        """

        self.verbose = verbose
        self.continuous_encoder = self.continuous_encoders.get(
            continuous_encoding
        )
        if self.continuous_encoder is None:
            raise ValueError(
                f"Continuous encoding '{continuous_encoding}' not recognised. "
                f"Available options: {list(self.continuous_encoders.keys())}"
            )
        self.max_components = max_components
        self.learn_rounding = learn_rounding_scheme
        self.enforce_min_max = enforce_min_max_values
        self.use_token_weights = use_token_weights

        columns = data.columns
        columns = [col for col in columns if col not in ["pid", "iid", "hid"]]
        if include is not None:
            columns = [col for col in columns if col in include]
        if exclude is not None:
            columns = [col for col in columns if col not in exclude]

        if not columns:
            raise UserWarning("No columns found to encode in table.")

        self.columns = columns

    def fit_and_encode(self, data: Union[pl.DataFrame, pd.DataFrame]) -> None:
        self.encoders = {}
        self.mode = type(data)
        self.initialise_encoders(data)

        encoded = []
        for name, encoder in self.encoders.items():
            if name not in data.columns:
                raise UserWarning(
                    f"Expected column '{name}' based on configuration, but not found in data"
                )
            x = encoder.fit_and_encode(data[name])
            encoded.append(x)

        if self.verbose:
            print(str(self))

        if not encoded:
            raise UserWarning("No encodings found.")

        encoded = cat(encoded, dim=-1).float()
        dataset = XDataset(encoded)
        return dataset

    def __repr__(self):
        return f"{self.__class__.__name__}: ({len(self.encoders)} encoders)"

    def __str__(self):
        return f"{self.__repr__()}:\n" + "\n".join(
            [f"\t--> {e}" for e in self.encoders.values()]
        )

    def initialise_encoders(
        self, data: Union[pl.DataFrame, pd.DataFrame]
    ) -> None:
        if isinstance(data, pd.DataFrame):
            self.configure_pandas(data)
        elif isinstance(data, pl.DataFrame):
            self.configure_polars(data)
        else:
            raise ValueError("Data must be a pandas or polars dataframe")

    def configure_polars(self, data: pl.DataFrame) -> None:
        """Configure the tokeniser by encoding the dataframe columns.
        Args:
            data (pl.DataFrame): input dataframe to configure.
            verbose (bool, optional): print the configuration. Defaults to False.
        """

        for column in self.columns:
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")
            values = data[column]
            dtype = values.dtype
            if (
                dtype == pl.Utf8
                or dtype == pl.Object
                or dtype == pl.Categorical
                or dtype == pl.Boolean
                or dtype == pl.Enum
            ):
                self.encoders[column] = CategoricalTokeniser(
                    name=column,
                    verbose=self.verbose,
                    use_token_weights=self.use_token_weights,
                )

            elif dtype.is_numeric():
                self.encoders[column] = self.continuous_encoder(
                    name=column,
                    verbose=self.verbose,
                    max_components=self.max_components,
                    learn_rounding=self.learn_rounding,
                    enforce_min_max=self.enforce_min_max,
                    use_token_weights=self.use_token_weights,
                )

            else:
                raise UserWarning(
                    f"Column '{column}' not supported for encoding: {values.dtype}"
                )

    def configure_pandas(self, data: pd.DataFrame) -> None:
        """Configure the tokeniser by encoding the dataframe columns.
        Args:
            data (pd.DataFrame): input dataframe to configure.
            verbose (bool, optional): print the configuration. Defaults to False.
        """

        for column in self.columns:
            if column not in data.columns:
                raise UserWarning(f"Column '{column}' not found in attributes")
            values = data[column]
            if (
                ptypes.is_string_dtype(values)
                or ptypes.is_object_dtype(values)
                or ptypes.is_categorical_dtype(values)
            ):
                self.encoders[column] = CategoricalTokeniser(
                    name=column,
                    verbose=self.verbose,
                    use_token_weights=self.use_token_weights,
                )
            elif ptypes.is_numeric_dtype(values):
                self.encoders[column] = self.continuous_encoder(
                    name=column,
                    verbose=self.verbose,
                    max_components=self.max_components,
                    learn_rounding=self.learn_rounding,
                    enforce_min_max=self.enforce_min_max,
                    use_token_weights=self.use_token_weights,
                )
            else:
                raise UserWarning(
                    f"Column '{column}' not supported for encoding: {values.dtype}"
                )

    def encode(self, data: Union[pl.DataFrame, pd.DataFrame]) -> Tensor:
        """Encode the dataframe into a Tensor.
        Args:
            data (Union[pl.DataFrame, pd.DataFrame]): input dataframe to encode.
        Returns:
            Tensor: encoded dataframe.
        """
        encoded = []
        for column, encoder in self.encoders.items():
            if column not in data.columns:
                raise UserWarning(
                    f"Expected column '{column}' based on configuration, but not found in data"
                )
            x = encoder.encode(data[column])
            encoded.append(x)

        if not encoded:
            raise UserWarning("No encodings found.")

        encoded = cat(encoded, dim=-1).float()
        dataset = XDataset(encoded)
        return dataset

    def decode(self, data: List[Tensor]) -> pd.DataFrame | pl.DataFrame:
        """Decode Tensor of tokens back into dataframe.

        Args:
            data (List[Tensor]): input Tensor of tokens to decode.

        Returns:
            Union[pd.DataFrame, pl.DataFrame]: decoded dataframe.
        """
        assert data.ndim == 2, "Data must be a 2D Tensor"
        assert data.shape[1] == sum(
            self.slot_sizes()
        ), "Data shape does not match encoder configuration"

        decoded = {}
        for (name, encoder), (i, j) in zip(
            self.encoders.items(), self.slot_idxs()
        ):
            tokens = data[:, i:j]
            decoded[name] = encoder.decode(tokens)
        decoded = (
            pd.DataFrame(decoded)
            if self.mode == pd.DataFrame
            else pl.DataFrame(decoded)
        )
        return decoded

    def encode_series(self, data: pd.Series) -> Tensor:
        """Encode a pandas series into a 1d Tensor.
        Args:
            data (pd.Series): input series to encode.
        Returns:
            Tensor: encoded series.
        """
        if data.name not in self.encoders.keys():
            raise UserWarning(f"'{data.name}' not found in available encoders")
        encoder = self.encoders[data.name]
        column_encoded = encoder.encode(data)
        return column_encoded

    def names(self) -> List[str]:
        """Get the names of the encoders.
        Returns:
            List[str]: list of encoder names.
        """
        return list(self.encoders.keys())

    def types(self) -> List[str]:
        """Get the types of the embeddings.
        Returns:
            List[str]: list of types of the embeddings.
        """
        return [encoder.encoding for encoder in self.encoders.values()]

    def slot_sizes(self) -> List[int]:
        """Get the slot sizes of the embeddings.
        Returns:
            List[int]: list of slot sizes of the embeddings.
        """
        return [encoder.slot_size for encoder in self.encoders.values()]

    def slot_idxs(self) -> List[int]:
        """Get the slot locations of the embeddings.
        Returns:
            List[int]: list of slot locations of the embeddings.
        """
        idxs = [0]
        for s in self.slot_sizes():
            idxs.append(idxs[-1] + s)
        starts = idxs[:-1]
        ends = idxs[1:]
        return list(zip(starts, ends))

    def sizes(self) -> List[int]:
        """Get the sizes of the embeddings.
        Returns:
            List[int]: list of sizes of the embeddings.
        """
        return [encoder.size for encoder in self.encoders.values()]

    def token_weights(self) -> list[Tensor]:
        """Get the token weights for each encoder."""
        return [encoder._token_weights for encoder in self.encoders.values()]
