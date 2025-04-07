"""Custom exceptions for binaural package."""


class BinauralError(Exception):
    """Base exception for Binaural errors."""


class ConfigFileNotFoundError(BinauralError):
    """Raised when the configuration file is not found."""


class YAMLParsingError(BinauralError):
    """Raised when YAML parsing fails."""


class ConfigurationError(BinauralError):
    """Raised when the YAML configuration is invalid."""


class AudioGenerationError(BinauralError):
    """Raised when an error occurs during audio generation."""


class UnsupportedFormatError(BinauralError):
    """Raised when the specified audio format is unsupported."""
