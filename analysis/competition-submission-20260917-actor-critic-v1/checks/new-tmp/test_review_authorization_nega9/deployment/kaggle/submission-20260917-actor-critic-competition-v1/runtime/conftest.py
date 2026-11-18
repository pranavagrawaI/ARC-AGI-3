"""Run this package's plain coroutine tests in the pinned worker Python.

These tests use synchronous fixtures only. No extra package installation is
needed; defer to pytest-asyncio when the caller already has that plugin.
"""
import asyncio
import inspect
import pytest


def pytest_configure(config):
    config.addinivalue_line('markers', 'asyncio: a plain asyncio coroutine test')


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.config.pluginmanager.hasplugin('asyncio'):
        return None
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        args = {name: pyfuncitem.funcargs[name] for name in inspect.signature(pyfuncitem.obj).parameters}
        asyncio.run(pyfuncitem.obj(**args))
        return True
