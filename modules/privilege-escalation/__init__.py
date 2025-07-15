# modules/privilege-escalation/__init__.py

"""
Privilege Escalation Exploits Package

Auto-discovers all Exploit subclasses in this directory so they can be
iterated, registered or invoked by your simulation engine.
"""

import pkgutil
import importlib
from typing import List, Type

from modules.exploits.exploit_template import Exploit

__all__: List[str] = []


def discover_exploits() -> List[Type[Exploit]]:
    """
    Dynamically load each submodule in this package and collect
    all classes that subclass Exploit (excluding the base class).
    Returns:
        List of Exploit subclasses available under this package.
    """
    exploits: List[Type[Exploit]] = []
    for finder, module_name, is_pkg in pkgutil.iter_modules(__path__):
        # import the module to ensure its classes are defined
        module = importlib.import_module(f"{__name__}.{module_name}")
        # scan module attributes for Exploit subclasses
        for attr in dir(module):
            cls = getattr(module, attr)
            if (
                isinstance(cls, type)
                and issubclass(cls, Exploit)
                and cls is not Exploit
            ):
                exploits.append(cls)
    return exploits


# preload all modules and populate __all__
for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")
    __all__.append(module_name)
