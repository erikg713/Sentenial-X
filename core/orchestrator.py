"""
Sentenial-X A.I. System Orchestrator

This module provides system orchestration capabilities including Flask monitoring,
RabbitMQ integration, and coordination of various security modules.

This was extracted from the original countermeasures.py to maintain separation of concerns.

Authors: Sentenial-X A.I. Team
License: See LICENSE file
"""

import json
import logging
import time
from multiprocessing import Process
from typing import Dict, Any

import pika
from flask import Flask, jsonify

# Configure logging for the orchestrator
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='sentenial_x_orchestrator.log',
    filemode='a'
)

logger = logging.getLogger(__name__)

# RabbitMQ configuration
RABBITMQ_HOST = 'localhost'
QUEUES = {
    'threat_detection': 'threat_detection_queue',
    'file_monitor': 'file_monitor_queue',
    'internet_scanner': 'internet_scanner_queue',
    'predictive': 'predictive_queue',
    'countermeasures': 'countermeasures_queue'
}

# Flask app for status monitoring
app = Flask(__name__)
system_status = {
    'threat_detection': 'stopped',
    'file_monitor': 'stopped',
    'internet_scanner': 'stopped',
    'predictive': 'stopped',
    'countermeasures': 'stopped'
}


@app.route('/status')
def get_status():
    """Get current system status."""
    return jsonify(system_status)


@app.route('/status/<module>')
def get_module_status(module):
    """Get status of a specific module."""
    if module in system_status:
        return jsonify({module: system_status[module]})
    else:
        return jsonify({'error': 'Module not found'}), 404


class SystemOrchestrator:
    """
    Main system orchestrator for managing security modules.
    
    This class coordinates various security components including threat detection,
    file monitoring, internet scanning, and countermeasures.
    """
    
    def __init__(self):
        """Initialize the system orchestrator."""
        self.processes: Dict[str, Process] = {}
        self.status = system_status.copy()
        logger.info("System orchestrator initialized")
    
    def start_module(self, module_name: str) -> bool:
        """
        Start a specific security module.
        
        Args:
            module_name: Name of the module to start
            
        Returns:
            True if module started successfully, False otherwise
        """
        if module_name not in QUEUES:
            logger.error(f"Unknown module: {module_name}")
            return False
            
        if module_name in self.processes and self.processes[module_name].is_alive():
            logger.warning(f"Module {module_name} is already running")
            return True
            
        try:
            # Import and start the appropriate module
            if module_name == 'threat_detection':
                from threat_detection import run_threat_detection
                process = Process(target=run_threat_detection)
            elif module_name == 'file_monitor':
                from file_monitor import start_file_monitoring
                process = Process(target=start_file_monitoring)
            elif module_name == 'internet_scanner':
                from internet_scanner import start_internet_scanning
                process = Process(target=start_internet_scanning)
            elif module_name == 'predictive':
                from predictive_capabilities import run_predictive_capabilities
                process = Process(target=run_predictive_capabilities)
            else:
                logger.error(f"No implementation found for module: {module_name}")
                return False
                
            process.start()
            self.processes[module_name] = process
            self.status[module_name] = 'running'
            logger.info(f"Started module: {module_name}")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to start module {module_name}: {e}")
            return False
    
    def stop_module(self, module_name: str) -> bool:
        """
        Stop a specific security module.
        
        Args:
            module_name: Name of the module to stop
            
        Returns:
            True if module stopped successfully, False otherwise
        """
        if module_name not in self.processes:
            logger.warning(f"Module {module_name} is not running")
            return True
            
        try:
            process = self.processes[module_name]
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                
                if process.is_alive():
                    logger.warning(f"Force killing module {module_name}")
                    process.kill()
                    process.join()
                    
            del self.processes[module_name]
            self.status[module_name] = 'stopped'
            logger.info(f"Stopped module: {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop module {module_name}: {e}")
            return False
    
    def start_all_modules(self) -> None:
        """Start all security modules."""
        logger.info("Starting all security modules")
        for module_name in QUEUES.keys():
            self.start_module(module_name)
    
    def stop_all_modules(self) -> None:
        """Stop all security modules."""
        logger.info("Stopping all security modules")
        for module_name in list(self.processes.keys()):
            self.stop_module(module_name)
    
    def get_system_status(self) -> Dict[str, str]:
        """
        Get current status of all modules.
        
        Returns:
            Dictionary with module names and their current status
        """
        # Update status based on actual process state
        for module_name, process in self.processes.items():
            if process.is_alive():
                self.status[module_name] = 'running'
            else:
                self.status[module_name] = 'stopped'
                
        return self.status.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the system.
        
        Returns:
            Dictionary containing health check results
        """
        health_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status': 'healthy',
            'modules': self.get_system_status(),
            'issues': []
        }
        
        # Check for any stopped modules that should be running
        for module_name, status in health_data['modules'].items():
            if status == 'stopped':
                health_data['issues'].append(f"Module {module_name} is not running")
        
        # Check RabbitMQ connectivity
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
            connection.close()
        except Exception as e:
            health_data['issues'].append(f"RabbitMQ connection failed: {e}")
            health_data['overall_status'] = 'degraded'
        
        if health_data['issues']:
            health_data['overall_status'] = 'degraded' if len(health_data['issues']) < 3 else 'unhealthy'
            
        return health_data


# Flask routes for orchestrator control
@app.route('/orchestrator/start/<module>')
def start_module_endpoint(module):
    """Start a specific module via HTTP endpoint."""
    global orchestrator
    if orchestrator.start_module(module):
        return jsonify({'status': 'success', 'message': f'Module {module} started'})
    else:
        return jsonify({'status': 'error', 'message': f'Failed to start module {module}'}), 500


@app.route('/orchestrator/stop/<module>')
def stop_module_endpoint(module):
    """Stop a specific module via HTTP endpoint."""
    global orchestrator
    if orchestrator.stop_module(module):
        return jsonify({'status': 'success', 'message': f'Module {module} stopped'})
    else:
        return jsonify({'status': 'error', 'message': f'Failed to stop module {module}'}), 500


@app.route('/orchestrator/health')
def health_check_endpoint():
    """Get system health status via HTTP endpoint."""
    global orchestrator
    return jsonify(orchestrator.health_check())


# Global orchestrator instance
orchestrator = None


def main():
    """Main function for running the orchestrator."""
    global orchestrator
    
    try:
        orchestrator = SystemOrchestrator()
        
        # Start Flask monitoring server
        logger.info("Starting Flask monitoring server on port 5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator")
        if orchestrator:
            orchestrator.stop_all_modules()
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        if orchestrator:
            orchestrator.stop_all_modules()


if __name__ == '__main__':
    main()