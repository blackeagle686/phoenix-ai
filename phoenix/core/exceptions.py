class PhoenixAIError(Exception):
    """Base exception for Phoenix AI SDK."""
    pass

class ServiceNotInitializedError(PhoenixAIError):
    """Raised when a service is used before being properly initialized."""
    pass

class ConfigurationError(PhoenixAIError):
    """Raised when there is a configuration issue."""
    pass

