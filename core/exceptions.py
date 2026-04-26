class AppError(Exception):
    """Base application exception."""


class ConfigurationError(AppError):
    """Raised when application configuration is invalid."""

