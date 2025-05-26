import logging
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger(__name__)


class IncidentReflexManager:
    """
    Manages incident reflex actions within the Sentenial-X system.
    Allows registration, deregistration, and execution of reflexes in response to detected incidents.
    """

    def __init__(self) -> None:
        self._reflex_registry: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def register_reflex(self, incident_type: str, reflex: Callable[[Dict[str, Any]], None]) -> None:
        """
        Registers a reflex function for a given incident type.
        
        Args:
            incident_type (str): The type of incident to register the reflex for.
            reflex (Callable): A callable accepting incident data as a dict.
        """
        if not callable(reflex):
            raise ValueError("Reflex must be callable")
        self._reflex_registry[incident_type] = reflex
        logger.debug(f"Registered reflex for incident type '{incident_type}'.")

    def deregister_reflex(self, incident_type: str) -> None:
        """
        Deregisters the reflex for a given incident type.
        
        Args:
            incident_type (str): The type of incident to remove the reflex for.
        """
        if incident_type in self._reflex_registry:
            del self._reflex_registry[incident_type]
            logger.debug(f"Deregistered reflex for incident type '{incident_type}'.")

    def handle_incident(self, incident_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Executes the reflex associated with the incident type, if any.
        
        Args:
            incident_type (str): The type of incident.
            data (Optional[Dict[str, Any]]): Incident data payload.
        """
        data = data or {}
        reflex = self._reflex_registry.get(incident_type)
        if reflex:
            logger.info(f"Handling incident '{incident_type}' with registered reflex.")
            try:
                reflex(data)
            except Exception as exc:
                logger.error(f"Error executing reflex for '{incident_type}': {exc}", exc_info=True)
        else:
            logger.warning(f"No reflex registered for incident type '{incident_type}'.")

    def list_registered_reflexes(self) -> Dict[str, Callable[[Dict[str, Any]], None]]:
        """
        Returns a dictionary of registered incident types and their reflexes.
        """
        return self._reflex_registry.copy()


# Example usage (remove in production, or adapt as test code):
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    irm = IncidentReflexManager()

    def block_ip_reflex(data: Dict[str, Any]) -> None:
        ip = data.get('ip')
        logger.info(f"Blocking IP address: {ip}")

    irm.register_reflex('intrusion_attempt', block_ip_reflex)
    irm.handle_incident('intrusion_attempt', {'ip': '192.168.1.100'})
    irm.handle_incident('malware_detected', {'file': 'evil.exe'})
