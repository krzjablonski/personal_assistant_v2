from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from importlib.resources import files
from typing import Any, Iterable

import yaml


_SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_DEFAULT_ROOT = Path(str(files("agent_skills").joinpath("bundled")))

VALID_SCRIPT_SAFETY = frozenset({"read-only", "local-mutation", "external-draft", "executive"})


@dataclass(frozen=True)
class SkillDiagnostic:
    level: str
    path: str
    message: str


@dataclass(frozen=True)
class ScriptSpec:
    """Per-script execution policy declared in ``metadata.scripts``.

    ``path`` is the script location relative to the skill root, ``safety`` is one
    of ``read-only``, ``local-mutation``, ``external-draft`` or ``executive`` and ``environment`` is
    the allowlist of environment-variable names the script may receive.

    ``requires_approval`` is an optional override for *only* the approval
    component of the safety mapping. When ``None`` (the default) the approval
    requirement is derived from ``safety``. When set explicitly it forces
    approval on/off regardless of ``safety``. Side-effect classification still
    determines retry and concurrency behavior.
    """

    path: str
    safety: str
    description: str = ""
    environment: tuple[str, ...] = ()
    requires_approval: bool | None = None


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    location: Path
    body: str
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scripts: dict[str, ScriptSpec] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """Return the directory containing this skill's instructions and bundled resources."""
        return self.location.parent

    def list_resources(self, max_files: int = 100) -> list[str]:
        """List bundled script, reference, and asset file paths relative to the skill root.

        Stop at max_files for positive limits, visiting resource directories in
        that order and sorting files within each directory.
        """
        resources: list[str] = []
        for dirname in ("scripts", "references", "assets"):
            base = self.root / dirname
            if not base.exists() or not base.is_dir():
                continue
            for path in sorted(p for p in base.rglob("*") if p.is_file()):
                resources.append(path.relative_to(self.root).as_posix())
                if len(resources) >= max_files:
                    return resources
        return resources


class SkillCatalog:
    """Discovers and renders Agent Skills from filesystem directories."""

    def __init__(
        self,
        skills: dict[str, SkillDefinition] | None = None,
        diagnostics: list[SkillDiagnostic] | None = None,
    ):
        """Create a catalog from supplied skill definitions and discovery diagnostics."""
        self._skills = dict(skills or {})
        self.diagnostics = list(diagnostics or [])

    @classmethod
    def discover(cls, roots: Iterable[Path | str] | None = None) -> "SkillCatalog":
        """Discover skill directories beneath supplied roots or the default bundled root."""
        catalog = cls()
        for root in roots or [default_skills_root()]:
            catalog._scan_root(Path(root))
        return catalog

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        """Return a copy of the skill-name mapping for callers to inspect."""
        return dict(self._skills)

    def get(self, name: str) -> SkillDefinition:
        """Return a named available skill, or raise ValueError when it is absent."""
        try:
            return self._skills[name]
        except KeyError as exc:
            raise ValueError(f"Skill '{name}' is not available.") from exc

    def names(self) -> tuple[str, ...]:
        """Return available skill names in sorted order for prompts and tool schemas."""
        return tuple(sorted(self._skills))

    def build_catalog_prompt(self) -> str:
        """Describe available skills and their required loading lifecycle for the agent prompt.

        Include names and descriptions without full instructions, or return an
        empty string when no skills are available.
        """
        if not self._skills:
            return ""

        lines = [
            "## Agent Skills",
            "",
            "Agent Skills are packages of expert instructions for specific tasks. The catalog below lists ONLY each "
            "skill's name and description — the actual instructions are NOT in your context yet. You cannot follow a "
            "skill until you load it.",
            "",
            "Mandatory lifecycle — follow it in order, every time:",
            "1. When a task matches a skill's description, FIRST call `load_skill_instructions` with that skill's name. "
            "This pulls the full SKILL.md into context.",
            "2. Only AFTER the instructions are in context, do what they say: read files they reference under "
            "`references/` or `assets/` with `read_skill_resource`, and run any bundled scripts they name with "
            "`run_skill_command`.",
            "3. `read_skill_resource` and `run_skill_command` FAIL for a skill whose instructions you have not loaded. "
            "Load first, then use.",
            "",
            "Never guess a script path or its arguments — the exact command is defined only inside the loaded "
            "instructions. Loading is cheap and idempotent, so when you are unsure whether a skill applies, load it and "
            "check.",
            "",
            "Some skills are instruction-only: they bundle no scripts, and their capabilities are exposed as "
            "directly-registered tools named in their instructions. Those tools work without loading the skill, but "
            "load its instructions before first use anyway — they tell you how to use the tools correctly.",
            "",
            "<available_skills>",
        ]
        for skill in sorted(self._skills.values(), key=lambda s: s.name):
            lines.extend(
                [
                    "  <skill>",
                    f"    <name>{html.escape(skill.name)}</name>",
                    f"    <description>{html.escape(skill.description)}</description>",
                    "  </skill>",
                ]
            )
        lines.append("</available_skills>")
        return "\n".join(lines)

    def _scan_root(self, root: Path) -> None:
        """Add valid immediate-child skills from a root and record discovery problems.

        Later definitions replace earlier skills with the same name and produce
        a shadowing warning.
        """
        if not root.exists():
            self.diagnostics.append(
                SkillDiagnostic("warning", str(root), "Skills root does not exist.")
            )
            return
        if not root.is_dir():
            self.diagnostics.append(
                SkillDiagnostic("error", str(root), "Skills root is not a directory.")
            )
            return

        for skill_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            skill = self._parse_skill(skill_file)
            if skill is None:
                continue
            if skill.name in self._skills:
                previous = self._skills[skill.name]
                self.diagnostics.append(
                    SkillDiagnostic(
                        "warning",
                        str(skill_file),
                        f"Skill '{skill.name}' shadows {previous.location}.",
                    )
                )
            self._skills[skill.name] = skill

    def _parse_skill(self, skill_file: Path) -> SkillDefinition | None:
        """Read a SKILL.md file into a definition, recording metadata and validation diagnostics.

        Return None for unreadable files or missing required frontmatter fields;
        retain definitions with nonfatal naming or metadata warnings.
        """
        try:
            text = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            self.diagnostics.append(
                SkillDiagnostic("error", str(skill_file), f"Failed to read: {exc}")
            )
            return None

        parsed = self._split_frontmatter(text, skill_file)
        if parsed is None:
            return None
        frontmatter, body = parsed

        name = str(frontmatter.get("name") or "").strip()
        description = str(frontmatter.get("description") or "").strip()
        if not name:
            self.diagnostics.append(
                SkillDiagnostic("error", str(skill_file), "Missing required `name`.")
            )
            return None
        if not description:
            self.diagnostics.append(
                SkillDiagnostic(
                    "error", str(skill_file), "Missing required `description`."
                )
            )
            return None

        if not _is_valid_skill_name(name):
            self.diagnostics.append(
                SkillDiagnostic(
                    "warning",
                    str(skill_file),
                    f"Skill name '{name}' does not follow Agent Skills naming rules.",
                )
            )
        if name != skill_file.parent.name:
            self.diagnostics.append(
                SkillDiagnostic(
                    "warning",
                    str(skill_file),
                    f"Skill name '{name}' does not match directory '{skill_file.parent.name}'.",
                )
            )
        if len(description) > 1024:
            self.diagnostics.append(
                SkillDiagnostic(
                    "warning",
                    str(skill_file),
                    "Skill description exceeds 1024 characters.",
                )
            )

        metadata = frontmatter.get("metadata") or {}
        if not isinstance(metadata, dict):
            self.diagnostics.append(
                SkillDiagnostic(
                    "warning", str(skill_file), "`metadata` is not a mapping; ignored."
                )
            )
            metadata = {}

        scripts = self._parse_script_specs(metadata, skill_file)

        return SkillDefinition(
            name=name,
            description=description,
            location=skill_file.resolve(),
            body=body.strip(),
            license=_optional_string(frontmatter.get("license")),
            metadata=metadata,
            scripts=scripts,
        )

    def _parse_script_specs(
        self, metadata: dict[str, Any], skill_file: Path
    ) -> dict[str, ScriptSpec]:
        """Parse ``metadata.scripts`` into typed :class:`ScriptSpec` entries.

        Malformed entries are dropped with a warning so an undeclared or invalid
        script is treated as most-restrictive by the runtime rather than crashing
        discovery.
        """
        raw = metadata.get("scripts")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            self.diagnostics.append(
                SkillDiagnostic(
                    "warning",
                    str(skill_file),
                    "`metadata.scripts` is not a mapping; ignored.",
                )
            )
            return {}

        specs: dict[str, ScriptSpec] = {}
        for path, entry in raw.items():
            path_str = str(path).strip()
            if not path_str:
                continue
            if not isinstance(entry, dict):
                self.diagnostics.append(
                    SkillDiagnostic(
                        "warning",
                        str(skill_file),
                        f"Script '{path_str}' metadata is not a mapping; ignored.",
                    )
                )
                continue

            safety = str(entry.get("safety") or "").strip()
            if safety not in VALID_SCRIPT_SAFETY:
                self.diagnostics.append(
                    SkillDiagnostic(
                        "warning",
                        str(skill_file),
                        f"Script '{path_str}' has invalid or missing safety "
                        f"'{safety}'; ignored.",
                    )
                )
                continue

            description = str(entry.get("description") or "").strip()
            environment = self._parse_script_environment(
                entry.get("environment"), path_str, skill_file
            )
            requires_approval = self._parse_requires_approval(
                entry.get("requires-approval"), path_str, skill_file
            )
            specs[path_str] = ScriptSpec(
                path=path_str,
                safety=safety,
                description=description,
                environment=environment,
                requires_approval=requires_approval,
            )
        return specs

    def _parse_requires_approval(
        self, value: Any, path_str: str, skill_file: Path
    ) -> bool | None:
        """Parse an optional ``requires-approval`` boolean override.

        Accepts a real boolean (from PyYAML) or a case-insensitive
        ``true``/``false`` string (for quoted metadata values). Anything else is
        malformed: it is ignored with a warning so the safety-derived default is
        used instead of crashing discovery.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        self.diagnostics.append(
            SkillDiagnostic(
                "warning",
                str(skill_file),
                f"Script '{path_str}' has an invalid requires-approval value "
                f"'{value}'; expected true or false. Using the safety default.",
            )
        )
        return None

    def _parse_script_environment(
        self, value: Any, path_str: str, skill_file: Path
    ) -> tuple[str, ...]:
        """Read a script's allowed environment-variable names, warning and skipping malformed values."""
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            self.diagnostics.append(
                SkillDiagnostic(
                    "warning",
                    str(skill_file),
                    f"Script '{path_str}' environment must be a list; ignored.",
                )
            )
            return ()
        names: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                self.diagnostics.append(
                    SkillDiagnostic(
                        "warning",
                        str(skill_file),
                        f"Script '{path_str}' has an invalid environment entry; "
                        "skipped.",
                    )
                )
                continue
            names.append(item.strip())
        return tuple(names)

    def _split_frontmatter(
        self, text: str, skill_file: Path
    ) -> tuple[dict[str, Any], str] | None:
        """Separate YAML metadata from skill instructions, recording errors for invalid frontmatter.

        Return the metadata mapping and body, or None when parsing fails.
        """
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            self.diagnostics.append(
                SkillDiagnostic(
                    "error",
                    str(skill_file),
                    "SKILL.md must start with YAML frontmatter.",
                )
            )
            return None

        closing_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_index = index
                break
        if closing_index is None:
            self.diagnostics.append(
                SkillDiagnostic(
                    "error",
                    str(skill_file),
                    "YAML frontmatter is missing a closing delimiter.",
                )
            )
            return None

        raw_frontmatter = "\n".join(lines[1:closing_index])
        body = "\n".join(lines[closing_index + 1 :])
        try:
            data = yaml.safe_load(raw_frontmatter) or {}
        except yaml.YAMLError as exc:
            self.diagnostics.append(
                SkillDiagnostic(
                    "error",
                    str(skill_file),
                    f"Could not parse YAML frontmatter: {exc}",
                )
            )
            return None

        if not isinstance(data, dict):
            self.diagnostics.append(
                SkillDiagnostic(
                    "error",
                    str(skill_file),
                    "YAML frontmatter must be a mapping.",
                )
            )
            return None
        return data, body


def default_skills_root() -> Path:
    """Return the bundled skills directory used when no discovery root is supplied."""
    return _DEFAULT_ROOT


def _is_valid_skill_name(name: str) -> bool:
    """Check the catalog's lowercase, hyphenated skill-name rules."""
    return bool(_SKILL_NAME_RE.fullmatch(name)) and "--" not in name


def _optional_string(value: Any) -> str | None:
    """Normalize optional frontmatter values to text while preserving missing values as None."""
    if value is None:
        return None
    return str(value)
