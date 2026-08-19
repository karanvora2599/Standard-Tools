from typing import Optional


class QuantError(Exception):
    """Base class for all standard-quant-tools exceptions."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
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


class NonRetryableAPIError(APIError):
    """An APIError that will never succeed no matter how many times it's
    retried — e.g. an invalid/expired API key (HTTP 401/403). Distinct from
    plain APIError (used for genuinely transient failures like 429/5xx) so
    the shared retry decorator (data/_retry.py) can tell them apart without
    guessing from the HTTP status embedded in the message text. Still an
    APIError subclass, so every existing `except APIError` call site keeps
    working unchanged."""

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


class AuditIntegrityError(QuantError):
    """
    The audit trail's own hash chain is damaged.

    Distinct from ValidationError because it is not a statement about the
    caller's input — it says the tamper-evident log on disk can no longer be
    extended honestly. The writer raises this instead of silently restarting
    the chain at genesis, which is what it used to do when a day file's last
    line was unparsable: a corrupted tail became "no previous record", and
    the trail carried on looking intact while having lost everything before
    the damage.

    Raised only for corruption of an EXISTING chain. A missing or empty file
    is a legitimate genesis and is not an error.
    """
