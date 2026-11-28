"""Errors shared by CLI commands."""

from __future__ import annotations


class CliError(Exception):
    """A user-readable failure (exit code 1 unless ``code`` says otherwise)."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code
