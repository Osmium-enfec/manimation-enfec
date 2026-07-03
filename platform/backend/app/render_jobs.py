"""Background Manim render jobs with on-disk status for polling."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

JobKind = Literal["preview", "export"]
ProgressCallback = Callable[[int, str], None]
RenderFn = Callable[[ProgressCallback | None], None]

_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()
_active_processes: dict[str, subprocess.Popen] = {}

# Global cap on how many Manim renders may run at once. Each Manim subprocess
# consumes a large amount of RAM (Cairo + NumPy + frame buffers), so without a
# cap a handful of concurrent users can exhaust memory and take the box down.
# Tune with MAX_CONCURRENT_RENDERS; default 2 is safe for a small droplet.
_MAX_CONCURRENT_RENDERS = max(1, int(os.environ.get("MAX_CONCURRENT_RENDERS", "2")))
_render_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_RENDERS)

_STATUS_FILES: dict[JobKind, str] = {
    "preview": ".render_status.json",
    "export": ".export_status.json",
}


def _status_path(renders_dir: Path, kind: JobKind) -> Path:
    return renders_dir / _STATUS_FILES[kind]


def _lock_key(project_id: str, kind: JobKind) -> str:
    return f"{project_id}:{kind}"


def _project_lock(project_id: str, kind: JobKind) -> threading.Lock:
    key = _lock_key(project_id, kind)
    with _registry_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_active_process(project_id: str, kind: JobKind, proc: subprocess.Popen) -> None:
    with _registry_lock:
        _active_processes[_lock_key(project_id, kind)] = proc


def unregister_active_process(project_id: str, kind: JobKind) -> None:
    with _registry_lock:
        _active_processes.pop(_lock_key(project_id, kind), None)


def cancel_render_job(project_id: str, kind: JobKind, renders_dir: Path) -> dict:
    """Request cancellation; kill Manim subprocess if running."""
    key = _lock_key(project_id, kind)
    with _registry_lock:
        proc = _active_processes.get(key)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    unregister_active_process(project_id, kind)
    payload = {
        "status": "cancelled",
        "finished_at": _now_iso(),
        "error": None,
        "phase": "Cancelled",
    }
    current = read_status(renders_dir, kind)
    if current.get("status") == "rendering":
        if "progress" in current:
            payload["progress"] = current.get("progress", 0)
        write_status(renders_dir, kind, payload)
    lock = _project_lock(project_id, kind)
    if lock.locked():
        try:
            lock.release()
        except RuntimeError:
            pass
    return read_status(renders_dir, kind)


def read_status(renders_dir: Path, kind: JobKind = "preview") -> dict:
    path = _status_path(renders_dir, kind)
    if not path.exists():
        return {"status": "idle"}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"status": "idle"}


def write_status(renders_dir: Path, kind: JobKind, payload: dict) -> None:
    renders_dir.mkdir(parents=True, exist_ok=True)
    _status_path(renders_dir, kind).write_text(json.dumps(payload, indent=2))


def update_status(renders_dir: Path, kind: JobKind, **fields: object) -> None:
    current = read_status(renders_dir, kind)
    current.update(fields)
    write_status(renders_dir, kind, current)


def start_render_job(
    project_id: str,
    kind: JobKind,
    renders_dir: Path,
    render_fn: RenderFn,
    *,
    track_progress: bool = False,
) -> dict:
    """Run render_fn in a background thread; return current job status."""
    lock = _project_lock(project_id, kind)
    if not lock.acquire(blocking=False):
        current = read_status(renders_dir, kind)
        if current.get("status") == "rendering":
            return current
        lock.acquire()

    initial: dict = {
        "status": "rendering",
        "started_at": _now_iso(),
        "error": None,
    }
    if track_progress:
        existing = read_status(renders_dir, kind)
        initial["progress"] = existing.get("progress", 0) if existing.get("status") == "rendering" else 0
        initial["phase"] = existing.get("phase") or "Starting Manim"
        if existing.get("started_at") and existing.get("status") == "rendering":
            initial["started_at"] = existing["started_at"]
    write_status(renders_dir, kind, initial)

    def _run() -> None:
        def progress_cb(percent: int, phase: str) -> None:
            if track_progress:
                update_status(
                    renders_dir,
                    kind,
                    progress=max(0, min(100, percent)),
                    phase=phase,
                )

        # Wait for a global render slot before launching the heavy Manim
        # subprocess. If all slots are busy, surface a "Queued" phase so the
        # frontend shows the job is waiting rather than stalled.
        if not _render_slots.acquire(blocking=False):
            if track_progress:
                update_status(renders_dir, kind, status="rendering", phase="Queued", progress=0)
            _render_slots.acquire()

        try:
            # A job can be cancelled while it was waiting for a slot; skip the
            # expensive render entirely if so.
            if read_status(renders_dir, kind).get("status") == "cancelled":
                return
            render_fn(progress_cb if track_progress else None)
            current = read_status(renders_dir, kind)
            if current.get("status") == "cancelled":
                return
            done: dict = {
                "status": "done",
                "finished_at": _now_iso(),
                "error": None,
            }
            if track_progress:
                done["progress"] = 100
                done["phase"] = "Complete"
            write_status(renders_dir, kind, done)
        except Exception as exc:
            current = read_status(renders_dir, kind)
            if current.get("status") == "cancelled":
                return
            failed: dict = {
                "status": "error",
                "finished_at": _now_iso(),
                "error": str(exc),
            }
            if track_progress:
                failed["phase"] = "Failed"
            write_status(renders_dir, kind, failed)
        finally:
            _render_slots.release()
            unregister_active_process(project_id, kind)
            lock.release()

    thread_name = f"{kind}-render-{project_id}"
    threading.Thread(target=_run, daemon=True, name=thread_name).start()
    return read_status(renders_dir, kind)
