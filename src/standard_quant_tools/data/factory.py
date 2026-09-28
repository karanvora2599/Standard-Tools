import logging
from typing import Literal, Optional, get_args

from standard_quant_tools.error import ValidationError

from .base import DataProvider
from .bloomberg_provider import BloombergProvider
from .polygon_provider import PolygonProvider
from .yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)

#: The providers `get_provider` can build. A tool field that selects one is
#: typed with this, so the schema an agent reads lists them, and the
#: factory's own refusal names the same set.
ProviderName = Literal["yfinance", "polygon", "bloomberg", "databento"]
PROVIDER_NAMES = get_args(ProviderName)


class DataFactory:
    """
    Factory class to create DataProvider instances.
    """

    @staticmethod
    def get_provider(
        source: str = "yfinance",
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> DataProvider:
        """
        Returns a DataProvider instance based on the source.

        Args:
            source: The name of the data provider (e.g., 'yfinance', 'bloomberg',
                'polygon', 'alpaca').
            api_key: Optional API key for providers that require it — 'polygon'
                (falls back to SQT_POLYGON_API_KEY if omitted); unused by
                'bloomberg' — Desktop API authenticates via the Terminal login,
                not a credential this process holds.
            host, port: 'bloomberg' only — override SQT_BLOOMBERG_HOST/
                SQT_BLOOMBERG_PORT (and the localhost:8194 Desktop API
                default) for this instance. See data/bloomberg_provider.py.

        Returns:
            DataProvider: An instance of a data provider.

        Raises:
            ValidationError: If the source is unknown or not implemented. A
                ValueError too, so an `except ValueError` still catches it.
            APIError: source='bloomberg' and blpapi isn't installed (see
                BloombergProvider's error message for install instructions),
                or source='polygon' and no API key was found anywhere (see
                PolygonProvider's error message).
        """
        if not isinstance(source, str):
            raise ValidationError(
                f"Unknown data provider source: {source!r}. The providers this "
                f"library serves are {list(PROVIDER_NAMES)}."
            )
        source = source.lower()
        logger.debug("[factory] provider=%s", source)

        if source == "yfinance":
            return YFinanceProvider()
        elif source == "bloomberg":
            return BloombergProvider(host=host, port=port)
        elif source == "polygon":
            return PolygonProvider(api_key=api_key)
        elif source == "databento":
            # The only provider here that serves DEPTH. Its credential comes
            # from DATABENTO_API_KEY rather than `api_key`, for the reason
            # its module docstring gives: a key passed through a spec would
            # be persisted, hashed into a model's lineage, and written into
            # decision records.
            from standard_quant_tools.data.databento_provider import (
                DatabentoProvider,
            )

            return DatabentoProvider(api_key=api_key)
        elif source == "alpaca":
            raise ValidationError(
                "The 'alpaca' data provider is not implemented. The providers "
                f"this library serves are {list(PROVIDER_NAMES)}."
            )
        else:
            # A bare ValueError here reached an agent through eleven tools
            # as an internal failure with no list to choose from.
            raise ValidationError(
                f"Unknown data provider source: '{source}'. The providers this "
                f"library serves are {list(PROVIDER_NAMES)}; leave the source "
                "unset for the default, 'yfinance'."
            )
