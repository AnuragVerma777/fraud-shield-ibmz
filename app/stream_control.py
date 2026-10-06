"""Start and manage the transaction stream as a background subprocess."""

import atexit
import math
import os
import signal
import subprocess
import sys
import threading

from src.config import ALERTS_PATH, PROJECT_ROOT, STREAM_LOG_PATH

try:
    import streamlit as st
except ImportError:  # The controller can still be imported by non-UI tooling.
    st = None


_SESSION_PROCESS_KEY = "stream_process"
_ACTIVE_PROCESSES: dict[int, subprocess.Popen] = {}
_PROCESS_LOCK = threading.Lock()


def _session_state():
    """Return Streamlit session state or explain why it is unavailable."""
    if st is None:
        raise RuntimeError("Streamlit is required to manage a dashboard stream.")
    return st.session_state


def _forget_process(process: subprocess.Popen) -> None:
    with _PROCESS_LOCK:
        _ACTIVE_PROCESSES.pop(id(process), None)


def _request_graceful_stop(process: subprocess.Popen, timeout: float = 5.0) -> None:
    """Ask the stream to stop cleanly, then force-reap it if it does not exit."""
    if process.poll() is not None:
        _forget_process(process)
        return

    try:
        if os.name == "nt":
            # start_stream creates a separate process group so this event is
            # delivered to the stream CLI rather than the Streamlit server.
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=timeout)
    except (AttributeError, OSError, ValueError, subprocess.TimeoutExpired):
        # The CLI handles Ctrl+C/SIGINT by stopping its producer and consumer.
        # If the platform cannot deliver that signal or the process is stuck,
        # terminate and reap it so a child is never left behind on shutdown.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
    finally:
        _forget_process(process)


def _cleanup_processes() -> None:
    """Stop all stream children when the Streamlit server shuts down."""
    with _PROCESS_LOCK:
        processes = list(_ACTIVE_PROCESSES.values())
    for process in processes:
        _request_graceful_stop(process)


atexit.register(_cleanup_processes)


def _clear_old_logs() -> None:
    """Clear prior stream and alert output before a new simulation starts."""
    for path in (STREAM_LOG_PATH, ALERTS_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def start_stream(speed_tps, max_events):
    """Start ``src.stream`` in the background and store its process per session.

    ``max_events`` is required and finite, so even if a browser disconnects
    without a Streamlit callback the simulation exits after its event limit.
    The server's shutdown hook also stops active subprocesses immediately.
    """
    try:
        speed_tps = float(speed_tps)
        max_events = int(max_events)
    except (TypeError, ValueError) as exc:
        raise ValueError("speed_tps and max_events must be numeric") from exc
    if not math.isfinite(speed_tps) or speed_tps <= 0:
        raise ValueError("speed_tps must be a finite value greater than zero")
    if max_events <= 0:
        raise ValueError("max_events must be greater than zero")

    state = _session_state()
    current = state.get(_SESSION_PROCESS_KEY)
    if current is not None and current.poll() is None:
        raise RuntimeError("A stream is already running in this dashboard session.")
    if current is not None:
        _forget_process(current)
        state.pop(_SESSION_PROCESS_KEY, None)

    _clear_old_logs()
    command = [
        sys.executable,
        "-m",
        "src.stream",
        "--tps",
        str(speed_tps),
        "--max-events",
        str(max_events),
        "--reset-log",
    ]
    options = {
        "cwd": PROJECT_ROOT,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "shell": False,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True

    process = subprocess.Popen(command, **options)
    with _PROCESS_LOCK:
        _ACTIVE_PROCESSES[id(process)] = process
    state[_SESSION_PROCESS_KEY] = process
    return process


def stop_stream() -> None:
    """Stop this dashboard session's stream and wait for the child to exit."""
    state = _session_state()
    process = state.get(_SESSION_PROCESS_KEY)
    if process is None:
        return
    _request_graceful_stop(process)
    state.pop(_SESSION_PROCESS_KEY, None)


def is_running() -> bool:
    """Return whether this session's stream process is still alive."""
    state = _session_state()
    process = state.get(_SESSION_PROCESS_KEY)
    if process is None:
        return False
    if process.poll() is None:
        return True
    _forget_process(process)
    state.pop(_SESSION_PROCESS_KEY, None)
    return False
