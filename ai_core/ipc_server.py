import zmq
import threading
import logging
from sentenialx.ai_core.detection import detect_prompt_threat

logger = logging.getLogger("SentenialX.IPC")


def start_ipc_server(port: int = 5555):
    """
    Starts a ZeroMQ REP server to communicate with external components (like GUI).
    """
    def server():
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://127.0.0.1:{port}")
        logger.info(f"IPC server running on port {port}")

        while True:
            try:
                message = socket.recv_json()
                logger.info(f"Received message: {message}")
                
                if message.get("action") == "analyze":
                    text = message.get("text", "")
                    score = detect_prompt_threat(text, source="ipc_request")
                    socket.send_json({"confidence": score})
                else:
                    socket.send_json({"error": "unknown action"})

            except Exception as e:
                logger.error(f"IPC server error: {e}")
                socket.send_json({"error": str(e)})

    threading.Thread(target=server, daemon=True).start()
