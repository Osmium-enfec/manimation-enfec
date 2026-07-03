"""Manim render subprocess."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path

from app.render_progress import ManimProgressParser

MANIM_ROOT = Path(__file__).resolve().parents[3]

ProgressCallback = Callable[[int, str], None]


def _manim_python() -> str:
    """Resolve Manim interpreter: Docker /opt/venv, local .venv, or PATH."""
    for candidate in (
        os.environ.get("MANIM_PYTHON"),
        MANIM_ROOT / ".venv" / "bin" / "python",
        Path("/opt/venv/bin/python"),
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return str(path)
    return shutil.which("python") or sys.executable


def _cleanup_partial_movies(quality_dir: Path) -> None:
    """Delete Manim's per-render partial movie cache.

    Every project renders a scene named ``generated_scene``, so all renders pile
    into the same ``media/videos/generated_scene/<quality>/partial_movie_files``
    tree. With ``flush_cache=False`` this grows without bound and eventually
    fills the disk. The final combined MP4 has already been copied out by the
    time we call this, so the partial clips are safe to remove.
    """
    pmf = quality_dir / "partial_movie_files"
    if pmf.is_dir():
        shutil.rmtree(pmf, ignore_errors=True)


def detect_scene_class_from_file(scene_file: Path) -> str:
    try:
        source = scene_file.read_text()
    except OSError:
        return "GeneratedScene"
    match = re.search(
        r"class\s+(\w+)\s*\(\s*(?:MovingCameraScene|BeatScene|MovingBeatScene|Scene)",
        source,
    )
    return match.group(1) if match else "GeneratedScene"


def _emit_progress(
    parser: ManimProgressParser,
    progress_callback: ProgressCallback | None,
    line: str,
) -> None:
    if not progress_callback:
        return
    cleaned = line.replace("\r", "\n")
    for part in cleaned.split("\n"):
        part = part.strip()
        if not part:
            continue
        update = parser.feed(part)
        if update:
            progress_callback(*update)
            return
    update = parser.feed(line)
    if update:
        progress_callback(*update)


def _stream_process_output(
    proc: subprocess.Popen[str],
    parser: ManimProgressParser,
    progress_callback: ProgressCallback | None,
) -> str:
    # Keep only a bounded tail of the output. A verbose Manim render can emit a
    # large amount of stdout, and we only ever use the tail (for error context),
    # so accumulating the whole stream would waste memory on every render.
    tail: deque[str] = deque(maxlen=500)
    buffer = ""
    assert proc.stdout is not None

    while True:
        ch = proc.stdout.read(1)
        if not ch:
            if buffer.strip():
                tail.append(buffer)
                _emit_progress(parser, progress_callback, buffer)
            break
        buffer += ch
        if ch in "\r\n":
            tail.append(buffer)
            _emit_progress(parser, progress_callback, buffer)
            buffer = ""

    proc.wait()
    return "".join(tail)


def render_scene(
    scene_file: Path,
    scene_class: str = "GeneratedScene",
    quality: str = "-ql",
    output_mp4: Path | None = None,
    progress_callback: ProgressCallback | None = None,
    *,
    process_registry: tuple[str, str] | None = None,
) -> Path:
    scene_file = scene_file.resolve()
    if scene_class == "GeneratedScene":
        scene_class = detect_scene_class_from_file(scene_file)

    if output_mp4:
        output_mp4 = output_mp4.resolve()
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        staging_mp4 = output_mp4.with_name(output_mp4.stem + ".partial.mp4")
        if staging_mp4.exists():
            staging_mp4.unlink()
    else:
        staging_mp4 = None

    python = _manim_python()
    cmd = [
        python,
        "-m",
        "manim",
        "render",
        quality,
        str(scene_file),
        scene_class,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MANIM_ROOT / "animations") + os.pathsep + env.get("PYTHONPATH", "")

    parser = ManimProgressParser.from_scene(scene_file)
    if progress_callback:
        progress_callback(1, "Starting Manim")

    proc = subprocess.Popen(
        cmd,
        cwd=str(MANIM_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process_registry:
        from app.render_jobs import register_active_process, unregister_active_process

        project_id, kind = process_registry
        register_active_process(project_id, kind, proc)
    output = _stream_process_output(proc, parser, progress_callback)
    if process_registry:
        unregister_active_process(project_id, kind)
    if proc.returncode != 0:
        tail = output[-4000:]
        raise RuntimeError(f"Manim render failed:\n{tail}")

    if progress_callback:
        progress_callback(99, "Saving export")

    media = MANIM_ROOT / "media" / "videos"
    candidates = sorted(
        media.rglob(f"{scene_class}.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Rendered MP4 not found")

    mp4 = candidates[0]
    if output_mp4:
        target = staging_mp4 or output_mp4
        shutil.copy2(mp4, target)
        if staging_mp4 and staging_mp4 != output_mp4:
            staging_mp4.replace(output_mp4)
        _cleanup_partial_movies(mp4.parent)
        if progress_callback:
            progress_callback(100, "Complete")
        return output_mp4
    _cleanup_partial_movies(mp4.parent)
    if progress_callback:
        progress_callback(100, "Complete")
    return mp4
