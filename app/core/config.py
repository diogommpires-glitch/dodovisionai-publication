from pydantic_settings import BaseSettings
import config as legacy_cfg

class Settings(BaseSettings):
    host: str = legacy_cfg.HOST
    port: int = legacy_cfg.PORT
    app_version: str = legacy_cfg.APP_VERSION

settings = Settings()
