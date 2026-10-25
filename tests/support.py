from __future__ import annotations

import multiprocessing
import socket
import tempfile
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import TracebackType
from typing import Any

_CONTEXT = multiprocessing.get_context("fork")
_STOP_TIMEOUT = 0.25


def _stop_process(process: BaseProcess, *, timeout: float = _STOP_TIMEOUT) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout)
    if process.is_alive():
        raise RuntimeError(f"could not stop test process {process.pid}")


def _serve_unix_socket(
    path: str,
    handler: Callable[[socket.socket], None],
    stop: Any,
    status: Connection,
) -> None:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection: socket.socket | None = None
    try:
        listener.bind(path)
        listener.listen(1)
        listener.settimeout(0.05)
        status.send(("ready", None))
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            handler(connection)
            break
    except BaseException as exc:
        try:
            status.send(("error", repr(exc)))
        except BaseException:
            pass
    finally:
        if connection is not None:
            connection.close()
        listener.close()
        status.close()


class UnixSocketServer:
    """One-shot Unix fixture isolated in a bounded, killable subprocess."""

    def __init__(
        self,
        handler: Callable[[socket.socket], None],
        *,
        shutdown_timeout: float = _STOP_TIMEOUT,
    ) -> None:
        self.handler = handler
        self.shutdown_timeout = shutdown_timeout
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "fake.sock"
        self._process: BaseProcess | None = None
        self._stop: Any = None
        self._status: Connection | None = None

    def __enter__(self) -> UnixSocketServer:
        stop = _CONTEXT.Event()
        parent, child = _CONTEXT.Pipe(duplex=False)
        process = _CONTEXT.Process(
            target=_serve_unix_socket,
            args=(str(self.path), self.handler, stop, child),
            name="fake-unix-server",
        )
        process.start()
        child.close()
        self._process = process
        self._stop = stop
        self._status = parent
        if not parent.poll(2):
            self._cleanup()
            raise TimeoutError("fake Unix server did not become ready")
        kind, detail = parent.recv()
        if kind != "ready":
            self._cleanup()
            raise RuntimeError(f"fake Unix server failed to start: {detail}")
        return self

    def _cleanup(self) -> str | None:
        error: str | None = None
        if self._stop is not None:
            self._stop.set()
        if self._process is not None:
            _stop_process(self._process, timeout=self.shutdown_timeout)
            self._process.close()
            self._process = None
        if self._status is not None:
            if self._status.poll():
                try:
                    kind, detail = self._status.recv()
                except EOFError:
                    pass
                else:
                    if kind == "error":
                        error = str(detail)
            self._status.close()
            self._status = None
        self._temporary.cleanup()
        return error

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        error = self._cleanup()
        if exception_type is None and error is not None:
            raise RuntimeError(f"fake Unix server handler failed: {error}")


def receive_all(connection: socket.socket, *, chunk_size: int = 64 * 1024) -> bytes:
    chunks: list[bytes] = []
    while chunk := connection.recv(chunk_size):
        chunks.append(chunk)
    return b"".join(chunks)
