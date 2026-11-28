"""API error type: every failure is JSON ``{"error": {"code", "message"}}`` with a proper HTTP status."""

from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
