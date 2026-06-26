#!/usr/bin/env python3
"""Shared runtime guards for the ACEA detection nodes.

Two recurring *operational* failure modes motivated this module (they are not
detector bugs, they are how the nodes get launched):

1. **Duplicate node instances.** ROS 2 lets you start two nodes with the same
   name; both then subscribe to the camera and publish to the same topics. The
   visible symptom is *alternating* ``processed_frame_count`` and contradictory
   state in ``ros2 topic echo`` (one says ``SCAN``, the other
   ``STOP_AND_LOCALIZE``), and config edits appear to have no effect because an
   orphaned old instance is still running. ``SingleInstanceLock`` makes a second
   instance on the same host refuse to start (race-free ``flock``);
   ``count_named_nodes`` additionally flags cross-host / cross-container copies
   that share the DDS graph (where a file lock cannot see them).

2. **Observability.** ``instance_id()`` (``host:pid``) is stamped into the status
   messages so two publishers are immediately obvious in ``ros2 topic echo``
   without having to count frame numbers by eye.

The module is intentionally ROS-free (only ``os`` / ``socket`` / ``fcntl``) so it
imports cleanly in the nodes' offline unit-test path too.
"""
from __future__ import annotations

import os
import socket
import tempfile

try:
    import fcntl  # POSIX only; ROS 2 targets Linux
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore


class DuplicateInstanceError(RuntimeError):
    """Raised when another instance already holds the single-instance lock."""


def instance_id() -> str:
    """Stable, human-readable id for this process: ``hostname:pid``."""
    return f"{socket.gethostname()}:{os.getpid()}"


class SingleInstanceLock:
    """Best-effort, race-free single-instance guard via ``flock`` on a /tmp file.

    Catches the common "an orphaned old node is still running on the same host"
    case. The lock is held for the lifetime of the file descriptor and is
    released automatically by the OS when the process dies (even on SIGKILL), so
    a stale lock file never blocks a future start. Cross-host duplicates cannot
    be seen through a local file and are handled by ``count_named_nodes``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = os.path.join(tempfile.gettempdir(), f"{name}.lock")
        self.fd = -1
        self.acquired = False
        self.holder = ""
        if fcntl is None:  # cannot lock here -> never block startup
            self.acquired = True
            return
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Held by another live process -> we are the duplicate.
            self.holder = self._read_holder()
            self._close()
            return
        except OSError:
            # Filesystem refuses advisory locks (rare, e.g. some overlay FS) ->
            # degrade gracefully rather than block the node from starting.
            self._close()
            self.acquired = True
            return
        try:
            os.ftruncate(self.fd, 0)
            os.write(self.fd, instance_id().encode())
            os.fsync(self.fd)
        except OSError:
            pass
        self.acquired = True

    def _read_holder(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return handle.read().strip() or "unknown"
        except OSError:
            return "unknown"

    def _close(self) -> None:
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1

    def release(self) -> None:
        if fcntl is not None and self.fd >= 0:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except OSError:
                pass
        self._close()


def count_named_nodes(node) -> int:
    """How many live nodes share this node's ``(name, namespace)`` on the graph.

    Returns 1 on any error so callers never false-alarm when discovery is not
    ready or the node stub is used offline.
    """
    try:
        my_name = node.get_name()
        my_ns = node.get_namespace()
        names = node.get_node_names_and_namespaces()
        return sum(1 for n, ns in names if n == my_name and ns == my_ns)
    except Exception:  # noqa: BLE001 - any discovery failure -> assume unique
        return 1
