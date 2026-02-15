class QuantError(Exception):
    """Base class for all standard-quant-tools exceptions."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception

class DataProviderError(QuantError):
    """Raised when data fetching fails."""
    pass

class DataNotFoundError(DataProviderError):
    """Raised when no data is found for a symbol."""
    pass

class InvalidSymbolError(DataProviderError):
    """Raised when a ticker symbol is invalid."""
    pass

class APIError(DataProviderError):
    """Raised when the external API returns an error or rate limit."""
    pass

class CalculationError(QuantError):
    """Raised when a calculation fails (e.g., division by zero, NaN inputs)."""
    pass

class ValidationError(QuantError):
    """Raised when input validation fails (e.g., negative period, empty DataFrame)."""
    pass

class BacktestError(QuantError):
    """Raised when the backtesting engine encounters an error."""
    pass
