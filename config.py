# config.py
from dynaconf import Dynaconf
settings = Dynaconf(
    envvar_prefix="SENTENIAL",
    settings_files=["settings.toml", ".secrets.toml"],
)
