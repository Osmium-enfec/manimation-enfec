"""Local filesystem project storage with snapshot history for revert."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(
    os.environ.get("MANIMATIONS_DATA_DIR", Path.home() / "manimations-studio")
)
DEFAULT_MAX_PROJECTS = max(1, int(os.environ.get("MANIMATIONS_MAX_PROJECTS", "10")))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    def __init__(self, data_dir: Path | None = None, *, max_projects: int | None = None):
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.max_projects = max(1, max_projects if max_projects is not None else DEFAULT_MAX_PROJECTS)
        self.prune_old_projects()

    def _project_dir(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def _project_created_at(self, project_dir: Path) -> str:
        meta_path = project_dir / "project.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                created_at = str(meta.get("created_at") or "").strip()
                if created_at:
                    return created_at
            except (json.JSONDecodeError, OSError):
                pass
        return datetime.fromtimestamp(project_dir.stat().st_mtime, tz=timezone.utc).isoformat()

    def _sorted_project_dirs_oldest_first(self) -> list[Path]:
        dirs = [
            d
            for d in self.projects_dir.iterdir()
            if d.is_dir() and (d / "project.json").exists()
        ]
        return sorted(dirs, key=self._project_created_at)

    def prune_old_projects(self, *, max_projects: int | None = None) -> list[str]:
        """Keep only the newest N projects; delete older ones from disk."""
        limit = max(1, max_projects if max_projects is not None else self.max_projects)
        project_dirs = self._sorted_project_dirs_oldest_first()
        deleted: list[str] = []
        while len(project_dirs) > limit:
            oldest = project_dirs.pop(0)
            project_id = oldest.name
            try:
                self.delete_project(project_id)
                deleted.append(project_id)
                logger.info("Pruned old project %s (limit=%s)", project_id, limit)
            except FileNotFoundError:
                continue
        if deleted:
            logger.info(
                "Project retention: removed %s old project(s); keeping newest %s",
                len(deleted),
                limit,
            )
        return deleted

    def list_projects(self) -> list[dict]:
        out = []
        for d in sorted(self.projects_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "project.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                preview = d / "renders" / "latest.mp4"
                out.append(
                    {
                        "id": d.name,
                        "name": meta.get("name", d.name),
                        "updated_at": meta.get("updated_at"),
                        "created_at": meta.get("created_at"),
                        "beat_count": len(meta.get("beats", [])),
                        "theme_id": meta.get("theme_id", "builtin_orange"),
                        "creation_mode": meta.get("creation_mode", "beat_studio"),
                        "has_preview": preview.exists(),
                    }
                )
        return out

    def delete_project(self, project_id: str) -> None:
        pdir = self._project_dir(project_id)
        if not pdir.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        shutil.rmtree(pdir)

    def create_project(
        self,
        name: str = "Untitled",
        *,
        theme_id: str = "builtin_orange",
        creation_mode: str = "beat_studio",
    ) -> dict:
        project_id = str(uuid.uuid4())[:8]
        pdir = self._project_dir(project_id)
        pdir.mkdir(parents=True)
        (pdir / "history").mkdir()
        (pdir / "renders").mkdir()
        (pdir / "media").mkdir(exist_ok=True)
        project: dict[str, Any] = {
            "id": project_id,
            "name": name,
            "creation_mode": creation_mode,
            "theme_id": theme_id,
            "style_pack": "course_clean",
            "use_camera": False,
            "beats": [],
            "chat": [],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        if creation_mode == "voice_motion":
            project["code_customized"] = True
            project["voice_motion"] = None
        if creation_mode == "excalidraw":
            project["code_customized"] = True
            project["excalidraw"] = None
        self.save_project(project, snapshot=False)
        self.prune_old_projects()
        return project

    def load_project(self, project_id: str) -> dict:
        path = self._project_dir(project_id) / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        project = json.loads(path.read_text())
        if not project.get("theme_id"):
            project["theme_id"] = "builtin_orange"
        if not project.get("creation_mode"):
            project["creation_mode"] = "beat_studio"
        return project

    def save_project(self, project: dict, *, snapshot: bool = True) -> dict:
        project_id = project["id"]
        project["updated_at"] = _now_iso()
        pdir = self._project_dir(project_id)
        pdir.mkdir(parents=True, exist_ok=True)

        if snapshot and (pdir / "project.json").exists():
            self.create_snapshot(project_id, label="auto-save")

        path = pdir / "project.json"
        path.write_text(json.dumps(project, indent=2))
        return project

    def create_snapshot(self, project_id: str, label: str = "snapshot") -> dict:
        project = self.load_project(project_id)
        snap_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        snap = {
            "id": snap_id,
            "label": label,
            "created_at": _now_iso(),
            "project": project,
        }
        snap_path = self._project_dir(project_id) / "history" / f"{snap_id}.json"
        snap_path.write_text(json.dumps(snap, indent=2))
        self._trim_history(project_id, keep=30)
        return {"id": snap_id, "label": label, "created_at": snap["created_at"]}

    def _trim_history(self, project_id: str, keep: int = 30) -> None:
        hist = self._project_dir(project_id) / "history"
        files = sorted(hist.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[keep:]:
            f.unlink(missing_ok=True)

    def list_snapshots(self, project_id: str) -> list[dict]:
        hist = self._project_dir(project_id) / "history"
        if not hist.exists():
            return []
        snaps = []
        for f in sorted(hist.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(f.read_text())
            snaps.append(
                {
                    "id": data["id"],
                    "label": data.get("label", "snapshot"),
                    "created_at": data.get("created_at"),
                }
            )
        return snaps

    def revert(self, project_id: str, snapshot_id: str) -> dict:
        snap_path = self._project_dir(project_id) / "history" / f"{snapshot_id}.json"
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
        snap = json.loads(snap_path.read_text())
        project = snap["project"]
        # Save current state before revert
        current = self.load_project(project_id)
        pre = {
            "id": f"pre_revert_{snapshot_id}",
            "label": "before-revert",
            "created_at": _now_iso(),
            "project": current,
        }
        (self._project_dir(project_id) / "history" / f"{pre['id']}.json").write_text(
            json.dumps(pre, indent=2)
        )
        project["updated_at"] = _now_iso()
        path = self._project_dir(project_id) / "project.json"
        path.write_text(json.dumps(project, indent=2))
        return project

    def render_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "renders" / "latest.mp4"

    def export_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "renders" / "export_1080p60.mp4"

    def scene_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "generated_scene.py"

    def write_scene(self, project_id: str, code: str) -> Path:
        path = self.scene_path(project_id)
        path.write_text(code)
        return path
