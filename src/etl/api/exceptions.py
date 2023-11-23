class DataAPIError(Exception):
    """Base class for https://api.foodakai.com exceptions."""


class DataAPIResponseError(DataAPIError):
    """Error raised when https://api.foodakai.com does not send the expected response."""
