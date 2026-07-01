import logging
from typing import Optional
from .base import DataProvider
from .yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class DataFactory:
    """
    Factory class to create DataProvider instances.
    """
    
    @staticmethod
    def get_provider(source: str = "yfinance", api_key: Optional[str] = None) -> DataProvider:
        """
        Returns a DataProvider instance based on the source.
        
        Args:
            source: The name of the data provider (e.g., 'yfinance', 'alpaca', 'polygon').
            api_key: Optional API key for providers that require it.
            
        Returns:
            DataProvider: An instance of a data provider.
            
        Raises:
            ValueError: If the source is unknown.
        """
        source = source.lower()
        logger.debug("[factory] provider=%s", source)

        if source == "yfinance":
            return YFinanceProvider()
        elif source == "alpaca":
            raise NotImplementedError("Alpaca provider is not yet implemented.")
        elif source == "polygon":
            raise NotImplementedError("Polygon provider is not yet implemented.")
        elif source == "bloomberg":
            raise NotImplementedError("Bloomberg provider is not yet implemented.")
        else:
            raise ValueError(f"Unknown data provider source: '{source}'")
