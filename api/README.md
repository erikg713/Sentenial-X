# install deps
pip install fastapi uvicorn pydantic python-dotenv

# ensure your CLI modules are importable: `cli/` must be a package in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# set minimal security config
export API_KEY="super-secret-key"

# run
uvicorn api.main:app --host 0.0.0.0 --port 8000
