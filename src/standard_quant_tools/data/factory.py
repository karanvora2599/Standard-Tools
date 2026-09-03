import logging
from typing import Optional

from .base import DataProvider
from .bloomberg_provider import BloombergProvider
from .polygon_provider import PolygonProvider
from .yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


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
            ValueError: If the source is unknown.
            APIError: source='bloomberg' and blpapi isn't installed (see
                BloombergProvider's error message for install instructions),
                or source='polygon' and no API key was found anywhere (see
                PolygonProvider's error message).
        """
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
            raise NotImplementedError("Alpaca provider is not yet implemented.")
        else:
            raise ValueError(f"Unknown data provider source: '{source}'")
