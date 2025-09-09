import click
import uvicorn

@click.command("serve")
@click.option("--host", default="0.0.0.0", help="Host to bind API.")
@click.option("--port", default=8000, help="Port for API server.")
def serve_api(host: str, port: int):
    """Run Sentenial-X API service."""
    uvicorn.run("sentenial_x_ai.api:app", host=host, port=port, reload=False)
