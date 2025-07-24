from abc import ABC, abstractmethod
from typing import Any, Dict, List

class Plugin(ABC):
     name: str
     description: str
+    # optional metadata so GUI can render forms
     parameters: List[Dict[str,Any]] = []

     @abstractmethod
     def run(self, **kwargs) -> Any:
         pass
         
class Plugin(ABC):
    name: str
    description: str
    # optional metadata so GUI can render forms automatically
    parameters: List[Dict[str, Any]] = []

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass
