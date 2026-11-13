from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PLATFORM_", env_file=".env.platform", extra="ignore"
    )
    environment: str = "local"
    database_url: str = (
        "mysql+asyncmy://research:local-research-only@127.0.0.1:53306/research_platform"
    )
    redis_url: str = "redis://127.0.0.1:56379/0"
    rabbit_url: str = "amqp://research:local-research-only@127.0.0.1:55672/"
    object_endpoint: str = "http://127.0.0.1:59000"
    object_bucket: str = "research-platform"
    object_access_key: str = "research-local"
    object_secret_key: str = "local-research-only"
    public_origin: str = "http://localhost:3000"
    require_email_verification: bool = True
    rate_limit_enabled: bool = False
    smtp_host: str = "127.0.0.1"
    smtp_port: int = 51025
    smtp_sender: str = "research@example.test"
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = False
    max_running: int = 2
    max_queued: int = 10
    max_file_bytes: int = 50 * 1024 * 1024
    worker_root: Path = Path(".data/platform-worker")
    task_timeout_seconds: int = 900
    admin_emails: str = ""

    @model_validator(mode="after")
    def production(self):
        if self.environment == "production":
            if not self.require_email_verification or not self.rate_limit_enabled:
                raise ValueError("Production requires verified email and distributed rate limits")
            if not self.public_origin.startswith("https://"):
                raise ValueError("Production requires HTTPS origin")
            if "local-" in self.database_url or self.object_secret_key == "local-research-only":
                raise ValueError("Production requires non-development credentials")
        return self
