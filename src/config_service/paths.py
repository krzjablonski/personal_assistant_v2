"""User-owned mutable data paths and non-destructive migration from older checkouts."""

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import sys
import tempfile
from time import monotonic


def default_data_dir() -> Path:
    """Locate mutable user data without reading or creating anything on disk."""
    override = os.environ.get("PERSONAL_ASSISTANT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser().resolve() / "personal-assistant"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "personal-assistant"
    return Path.home() / ".local" / "share" / "personal-assistant"


_PERMISSION_WARNINGS: set[Path] = set()


def private_directory(path: Path) -> Path:
    """Create an application data directory with access limited to its owner.

    Only directories created here are restricted to 0700. An existing directory
    (for example a user-named --data-dir) keeps its permissions; if group or
    other users can access it, a one-time warning is printed instead.
    """
    try:
        path.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        if not path.is_dir():
            raise
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077 and path.absolute() not in _PERMISSION_WARNINGS:
            _PERMISSION_WARNINGS.add(path.absolute())
            print(
                f"Warning: private data directory {path} is accessible by other users "
                f"(mode {mode:o}); run 'chmod 700 {path}' to restrict it.",
                file=sys.stderr,
            )
        return path
    path.chmod(0o700)
    return path


def _copy_regular_file_nofollow(source: Path, destination: Path) -> None:
    """Copy a regular file without following a final-component symbolic link."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        if source.is_symlink():
            raise ValueError("Migration does not follow a symbolic link for the inbox-triage report") from error
        raise
    with os.fdopen(descriptor, "rb") as reader:
        if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
            raise ValueError("Migration report must be a regular file")
        with open(destination, "xb") as writer:
            shutil.copyfileobj(reader, writer)
    shutil.copystat(source, destination, follow_symlinks=False)


def migrate_legacy_data(source: Path, target: Path, *, report_path: Path | None = None, timeout_seconds: float = 30) -> None:
    """Copy known legacy data atomically into a new directory, preserving the source.

    SQLite backup includes committed WAL content. Existing destinations are never
    replaced. To roll back, select the original directory with --data-dir; both
    copies remain available and subsequent edits are not synchronized.
    """
    source, target = Path(source).resolve(), Path(target).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Migration requires a new, empty destination path")
    if not source.is_dir():
        raise ValueError("Legacy data directory does not exist")
    if target.resolve().is_relative_to(source):
        raise ValueError("Migration destination must be outside the legacy data directory")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    copied = []
    deadline = monotonic() + timeout_seconds

    def check_backup_deadline(_status, _remaining, _total):
        if monotonic() >= deadline:
            raise TimeoutError("Legacy database backup exceeded its deadline")

    with tempfile.TemporaryDirectory(prefix=f".{target.name}-migration-", dir=target.parent) as temp:
        staging = private_directory(Path(temp) / "data")
        for name in ("agent_config.db", "agent_memory.db", "approved_commands.json", "agent_runs", "artifacts"):
            original, destination = source / name, staging / name
            if not original.exists():
                continue
            if original.is_symlink() or (original.is_dir() and any(p.is_symlink() for p in original.rglob("*"))):
                raise ValueError("Migration does not follow symbolic links inside legacy data")
            if name.endswith(".db"):
                with closing(sqlite3.connect(original.as_uri() + "?mode=ro", uri=True)) as old:
                    with closing(sqlite3.connect(destination)) as new:
                        old.backup(new, pages=128, progress=check_backup_deadline, sleep=0.05)
                        if new.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise ValueError("Legacy database integrity check failed")
            elif original.is_dir():
                shutil.copytree(original, destination)
            else:
                shutil.copy2(original, destination)
            copied.append(name)
        if report_path is not None and report_path.is_file():
            if report_path.is_symlink():
                raise ValueError("Migration does not follow a symbolic link for the inbox-triage report")
            reports = private_directory(staging / "reports")
            _copy_regular_file_nofollow(report_path, reports / "inbox-triage.md")
            copied.append("reports/inbox-triage.md")
        # Rebase only artifact locations. Tool inputs and old log transcripts
        # remain verbatim historical evidence.
        for manifest in (staging / "artifacts").rglob("*.json"):
            try:
                data = json.loads(manifest.read_text())
                old_path = Path(data["path"]).resolve()
                relative = old_path.relative_to(source)
            except (ValueError, KeyError, TypeError):
                continue
            data["path"] = str(target / relative)
            manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        (staging / "migration.json").write_text(json.dumps({
            "version": 1, "source": str(source), "copied": copied,
            "rollback": "Select the original directory with --data-dir; neither copy is deleted.",
        }, indent=2))
        for path in staging.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        # Reserve the destination with an exclusive mkdir. rename may replace
        # our own empty reservation, never a directory another caller created.
        target.mkdir(mode=0o700)
        reservation = target.stat()
        try:
            os.rename(staging, target)
        except BaseException:
            if target.exists() and target.stat().st_ino == reservation.st_ino:
                try:
                    target.rmdir()  # Only removes our reservation if still empty.
                except OSError:
                    pass
            raise
