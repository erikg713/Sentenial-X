"""
Sentenial-X A.I. Countermeasures Module

This module provides comprehensive threat response and countermeasure capabilities.
It includes adaptive learning, policy-driven responses, and forensic logging.

Classes:
    Countermeasures: Main countermeasure orchestration class with learning capabilities
    CountermeasureEngine: Rule-based threat evaluation and response engine

Authors: Sentenial-X A.I. Team
License: See LICENSE file
"""

import importlib.util
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

try:
    from core.neural_engine import NeuralEngine
except ImportError:
    # Fallback if neural_engine is not available
    class NeuralEngine:
        def predict(self, signature):
            return {'level': 'medium', 'tags': ['suspicious'], 'confidence': 0.5}

try:
    from core.neural_engine.learning_engine import LearningEngine
except ImportError:
    # Fallback if learning_engine is not available
    class LearningEngine:
        def update_with_event(self, event): pass
        def retrain_if_needed(self): pass
        def select_best_action(self, threat): return "monitor"

try:
    from core.telemetry import TelemetryListener
except ImportError:
    # Fallback if telemetry is not available
    class TelemetryListener:
        def __init__(self, callback=None): self.callback = callback
        def start(self): pass
        def stop(self): pass

# Handle feedback collector with the actual filename
try:
    # Try standard import first
    from feedback import FeedbackCollector
except ImportError:
    try:
        # Load feedback module with space in filename
        feedback_path = os.path.join(os.path.dirname(__file__), "feedback py")
        if os.path.exists(feedback_path):
            spec = importlib.util.spec_from_file_location("feedback", feedback_path)
            if spec and spec.loader:
                feedback_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(feedback_module)
                FeedbackCollector = feedback_module.FeedbackCollector
            else:
                raise ImportError("Could not load feedback module spec")
        else:
            raise ImportError("Feedback module not found")
    except ImportError:
        # Fallback if feedback collector is not available
        class FeedbackCollector:
            def store_feedback(self, detection_id, label): pass

# Handle utils import
try:
    from utils.helpers import get_timestamp, load_config
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    try:
        from utils.helpers import get_timestamp, load_config
    except ImportError:
        # Fallback implementations
        import time
        import json
        
        def get_timestamp():
            return time.strftime('%Y-%m-%d %H:%M:%S')
            
        def load_config(config_path='config.json'):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}

# Configure module logger
logger = logging.getLogger(__name__)


class Countermeasures:
    """
    Adaptive countermeasure orchestration system with machine learning capabilities.
    
    This class provides real-time threat response through adaptive policies,
    telemetry ingestion, and analyst feedback integration for continuous improvement.
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for the countermeasure system
        learning_engine (LearningEngine): ML engine for adaptive decision making
        feedback_collector (FeedbackCollector): Collector for analyst feedback
        telemetry_listener (TelemetryListener): Real-time telemetry data processor
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Countermeasures system.
        
        Args:
            config: Configuration dictionary containing system parameters
            
        Raises:
            ValueError: If required configuration parameters are missing
            ConnectionError: If unable to connect to required services
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        self.config = config
        
        try:
            self.learning_engine = LearningEngine()
            self.feedback_collector = FeedbackCollector()
            self.telemetry_listener = TelemetryListener(callback=self.ingest_telemetry)
            logger.info("Countermeasures system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize countermeasures system: {e}")
            raise ConnectionError(f"System initialization failed: {e}")

    def ingest_telemetry(self, event: Dict[str, Any]) -> None:
        """
        Process incoming telemetry data and update the learning engine.
        
        Args:
            event: Telemetry event data containing threat indicators
            
        Raises:
            ValueError: If event data is malformed
        """
        if not isinstance(event, dict):
            logger.warning(f"Invalid telemetry event format: {type(event)}")
            return
            
        try:
            self.learning_engine.update_with_event(event)
            logger.debug(f"Processed telemetry event: {event.get('event_type', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to process telemetry event: {e}")

    def receive_feedback(self, detection_id: str, label: str) -> None:
        """
        Store analyst feedback and trigger retraining if needed.
        
        Args:
            detection_id: Unique identifier for the detection
            label: Analyst-provided label (e.g., 'true_positive', 'false_positive')
            
        Raises:
            ValueError: If detection_id or label is invalid
        """
        if not detection_id or not isinstance(detection_id, str):
            raise ValueError("Detection ID must be a non-empty string")
            
        if not label or not isinstance(label, str):
            raise ValueError("Label must be a non-empty string")
            
        try:
            self.feedback_collector.store_feedback(detection_id, label)
            self.learning_engine.retrain_if_needed()
            logger.info(f"Feedback received for detection {detection_id}: {label}")
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")

    def apply_countermeasure(self, threat: Dict[str, Any]) -> Optional[str]:
        """
        Apply adaptive countermeasure based on threat analysis.
        
        Args:
            threat: Threat data containing indicators and metadata
            
        Returns:
            Action taken as a string, or None if no action was required
            
        Raises:
            ValueError: If threat data is malformed
        """
        if not isinstance(threat, dict):
            raise ValueError("Threat data must be a dictionary")
            
        try:
            action = self.learning_engine.select_best_action(threat)
            if action:
                self.execute_action(action, threat)
                logger.info(f"Applied countermeasure '{action}' for threat: {threat.get('id', 'unknown')}")
                return action
            else:
                logger.debug("No countermeasure action required")
                return None
        except Exception as e:
            logger.error(f"Failed to apply countermeasure: {e}")
            return None

    def execute_action(self, action: str, threat_data: Dict[str, Any]) -> None:
        """
        Execute a specific countermeasure action.
        
        Args:
            action: The countermeasure action to execute
            threat_data: Context data about the threat
            
        Raises:
            ValueError: If action or threat_data is invalid
        """
        if not action or not isinstance(action, str):
            raise ValueError("Action must be a non-empty string")
            
        try:
            logger.debug(f"Executing action '{action}' on target: {threat_data.get('target', 'unknown')}")
            
            if action == "isolate":
                self.isolate_target(threat_data.get('target'))
            elif action == "quarantine":
                self.quarantine_file(threat_data.get('file_path'))
            elif action == "block":
                self.block_ip(threat_data.get('source_ip'))
            else:
                logger.warning(f"Unknown action requested: {action}")
                
        except Exception as e:
            logger.error(f"Failed to execute action '{action}': {e}")
            raise

    def isolate_target(self, target: Optional[str]) -> None:
        """
        Isolate a network target from the rest of the network.
        
        Args:
            target: Target identifier (IP address, hostname, etc.)
        """
        if not target:
            logger.warning("No target specified for isolation")
            return
            
        logger.info(f"Isolating network target: {target}")
        # TODO: Implement actual network isolation logic
        # This would typically involve firewall rules, VLAN changes, etc.

    def quarantine_file(self, file_path: Optional[str]) -> None:
        """
        Quarantine a suspicious file.
        
        Args:
            file_path: Path to the file to quarantine
        """
        if not file_path:
            logger.warning("No file path specified for quarantine")
            return
            
        logger.info(f"Quarantining file: {file_path}")
        # TODO: Implement actual file quarantine logic
        # This would involve moving the file to a secure location

    def block_ip(self, ip_address: Optional[str]) -> None:
        """
        Block an IP address at the network level.
        
        Args:
            ip_address: IP address to block
        """
        if not ip_address:
            logger.warning("No IP address specified for blocking")
            return
            
        logger.info(f"Blocking IP address: {ip_address}")
        # TODO: Implement actual IP blocking logic
        # This would involve firewall rule creation

    def start(self) -> None:
        """Start the telemetry listener and begin processing events."""
        try:
            self.telemetry_listener.start()
            logger.info("Countermeasures system started successfully")
        except Exception as e:
            logger.error(f"Failed to start countermeasures system: {e}")
            raise

    def stop(self) -> None:
        """Stop the telemetry listener and clean up resources."""
        try:
            if hasattr(self.telemetry_listener, 'stop'):
                self.telemetry_listener.stop()
            logger.info("Countermeasures system stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping countermeasures system: {e}")


class CountermeasureEngine:
    """
    Rule-based threat evaluation and countermeasure execution engine.
    
    This class provides deterministic threat assessment using predefined rules
    and neural network predictions, with comprehensive logging and forensics.
    
    Attributes:
        config (Dict[str, Any]): Engine configuration parameters
        engine (NeuralEngine): Neural network for threat prediction
        actions_log (List[Dict[str, Any]]): Log of all executed actions
        rules (List[Dict[str, Any]]): Loaded countermeasure rules
    """
    
    def __init__(self, config_path: str = 'config.json') -> None:
        """
        Initialize the CountermeasureEngine.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        try:
            self.config = load_config(config_path)
            self.engine = NeuralEngine()
            self.actions_log: List[Dict[str, Any]] = []
            self.rules: List[Dict[str, Any]] = []
            
            self.load_predefined_rules()
            logger.info("CountermeasureEngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CountermeasureEngine: {e}")
            raise

    def load_predefined_rules(self) -> None:
        """
        Load countermeasure rules from the configured rules file.
        
        Raises:
            IOError: If unable to read the rules file
            json.JSONDecodeError: If rules file contains invalid JSON
        """
        rules_path = self.config.get('countermeasure_rules_path', 'data/rules/countermeasures.json')
        
        try:
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as file:
                    self.rules = json.load(file)
                    
                if not isinstance(self.rules, list):
                    raise ValueError("Rules file must contain a list of rules")
                    
                logger.info(f"Successfully loaded {len(self.rules)} countermeasure rules from {rules_path}")
            else:
                logger.warning(f"Rules file not found at {rules_path}. Using empty rule set.")
                self.rules = []
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in rules file {rules_path}: {e}")
            self.rules = []
        except Exception as e:
            logger.error(f"Failed to load countermeasure rules: {e}")
            self.rules = []

    def evaluate_threat(self, threat_signature: str) -> Dict[str, Any]:
        """
        Evaluate a threat signature using the neural engine.
        
        Args:
            threat_signature: Unique signature identifying the threat
            
        Returns:
            Dictionary containing threat evaluation results
            
        Raises:
            ValueError: If threat_signature is invalid
        """
        if not threat_signature or not isinstance(threat_signature, str):
            raise ValueError("Threat signature must be a non-empty string")
            
        try:
            prediction = self.engine.predict(threat_signature)
            logger.debug(f"Threat evaluation for '{threat_signature}': {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Failed to evaluate threat signature '{threat_signature}': {e}")
            return {'level': 'unknown', 'tags': [], 'confidence': 0.0}

    def execute_countermeasures(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute countermeasures based on threat data and configured rules.
        
        Args:
            threat_data: Dictionary containing threat information including 'signature'
            
        Returns:
            Dictionary containing execution results and actions taken
            
        Raises:
            ValueError: If threat_data is invalid or missing required fields
        """
        if not isinstance(threat_data, dict):
            raise ValueError("Threat data must be a dictionary")
            
        if 'signature' not in threat_data:
            raise ValueError("Threat data must contain a 'signature' field")
            
        try:
            # Evaluate the threat
            prediction = self.evaluate_threat(threat_data['signature'])
            
            # Build response structure
            response = {
                'threat_level': prediction.get('level', 'unknown'),
                'confidence': prediction.get('confidence', 0.0),
                'actions_taken': [],
                'timestamp': get_timestamp(),
                'details': threat_data.copy()
            }
            
            # Apply matching rules
            prediction_tags = prediction.get('tags', [])
            for rule in self.rules:
                if self._rule_matches(rule, prediction_tags, prediction):
                    action = rule.get('action')
                    if action:
                        try:
                            self.apply_action(action, threat_data)
                            response['actions_taken'].append({
                                'action': action,
                                'rule_trigger': rule.get('trigger'),
                                'timestamp': get_timestamp()
                            })
                            logger.info(f"Rule triggered: {rule.get('trigger')} -> Action: {action}")
                        except Exception as e:
                            logger.error(f"Failed to apply action '{action}': {e}")
                            response['actions_taken'].append({
                                'action': action,
                                'rule_trigger': rule.get('trigger'),
                                'error': str(e),
                                'timestamp': get_timestamp()
                            })
            
            # Log the response
            self.actions_log.append(response)
            return response
            
        except Exception as e:
            logger.error(f"Failed to execute countermeasures: {e}")
            error_response = {
                'threat_level': 'error',
                'actions_taken': [],
                'timestamp': get_timestamp(),
                'error': str(e),
                'details': threat_data.copy()
            }
            self.actions_log.append(error_response)
            return error_response

    def _rule_matches(self, rule: Dict[str, Any], prediction_tags: List[str], 
                      prediction: Dict[str, Any]) -> bool:
        """
        Check if a rule matches the current prediction.
        
        Args:
            rule: Rule dictionary to check
            prediction_tags: Tags from the threat prediction
            prediction: Full prediction dictionary
            
        Returns:
            True if the rule matches, False otherwise
        """
        trigger = rule.get('trigger')
        if not trigger:
            return False
            
        # Check for tag-based triggers
        if trigger in prediction_tags:
            return True
            
        # Check for threshold-based triggers
        min_confidence = rule.get('min_confidence', 0.0)
        if prediction.get('confidence', 0.0) >= min_confidence:
            threat_level = rule.get('threat_level')
            if threat_level and prediction.get('level') == threat_level:
                return True
                
        return False

    def apply_action(self, action: str, threat_data: Dict[str, Any]) -> None:
        """
        Apply a specific countermeasure action.
        
        Args:
            action: The action to execute
            threat_data: Context data about the threat
            
        Raises:
            ValueError: If action is invalid
        """
        if not action or not isinstance(action, str):
            raise ValueError("Action must be a non-empty string")
            
        logger.debug(f"Applying action '{action}' to target: {threat_data.get('target', 'unknown')}")
        
        # TODO: Implement actual countermeasure logic based on action type
        # This is a placeholder for the actual implementation
        time.sleep(0.1)  # Simulate action execution time

    def get_action_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete action log.
        
        Returns:
            List of all executed actions and their results
        """
        return self.actions_log.copy()

    def save_action_log(self, filepath: str = 'logs/countermeasure_log.json') -> None:
        """
        Save the action log to a file.
        
        Args:
            filepath: Path where the log file should be saved
            
        Raises:
            IOError: If unable to write to the specified file
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(self.actions_log, file, indent=2, ensure_ascii=False)
                
            logger.info(f"Action log saved to {filepath} ({len(self.actions_log)} entries)")
            
        except Exception as e:
            logger.error(f"Failed to save action log to {filepath}: {e}")
            raise IOError(f"Could not save action log: {e}")

    def clear_action_log(self) -> None:
        """Clear the in-memory action log."""
        self.actions_log.clear()
        logger.info("Action log cleared")


def main() -> None:
    """
    Main function for testing the countermeasure engine.
    
    This function demonstrates basic usage of the CountermeasureEngine
    and can be used for development and testing purposes.
    """
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize the engine
        cm_engine = CountermeasureEngine()
        
        # Create sample threat data
        sample_threat = {
            'signature': 'anomalous_behavior_sequence_xyz',
            'target': '192.168.1.25',
            'origin': 'endpoint-agent-14',
            'severity': 'high',
            'timestamp': get_timestamp()
        }
        
        # Execute countermeasures
        result = cm_engine.execute_countermeasures(sample_threat)
        
        # Display results
        print("Countermeasure Execution Results:")
        print(json.dumps(result, indent=2))
        
        # Save action log
        cm_engine.save_action_log()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1
        
    return 0


if __name__ == '__main__':
    exit(main())