"""Database utility functions for CLI commands."""

from pathlib import Path

from chunkhound.core.config.config import Config


def verify_database_exists(config: Config) -> Path:
    """Verify database exists, raising if not found.

    Args:
        config: Configuration with database settings

    Raises:
        FileNotFoundError: If database doesn't exist
        ValueError: If database path not configured
    """
    return config.database.get_db_path(must_exist=True)
