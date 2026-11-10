"""TheHouse ML: modelo de compatibilidad de Smart Match."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("thehouse-ml")
except PackageNotFoundError:  # pragma: no cover - solo sin instalar el paquete
    __version__ = "0.0.0"

__all__ = ["__version__"]
