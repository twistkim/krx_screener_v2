from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    APP_NAME: str = "KRX Screener V2"
    ENV: str = "local"

    # ✅ DB (고정 규칙)
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_USER: str = "krxapp"
    DB_PASSWORD: str = "krx_pw_1234"
    DB_NAME: str = "krx_screener_v2"
    DB_ECHO: bool = False

    @property
    def DATABASE_URL(self) -> str:
        # mysql+asyncmy DSN
        return (
            f"mysql+asyncmy://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"
        )

    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def TEMPLATE_DIR(self) -> Path:
        return self.BASE_DIR / "app" / "templates"

    @property
    def STATIC_DIR(self) -> Path:
        return self.BASE_DIR / "app" / "static"
    
settings = Settings()