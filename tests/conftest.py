"""Shared pytest fixtures."""

import socket

import pytest


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch):
    """Make DNS lookups in the feature extractor offline & instant for tests."""
    monkeypatch.setattr(socket, "gethostbyname", lambda host: "127.0.0.1")
