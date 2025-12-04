# core/compliance/templates/__init__.py
"""
Sentenial-X — Compliance Templates package initializer

This module provides a tiny, dependency-free template registry for
compliance-related text snippets used by the compliance tooling
(e.g. regulatory_vector_matcher, reporting, audit automation).

Features
- Register and retrieve templates (id -> Template)
- Render templates with safe `.format()` substitution (no KeyError)
- Optionally use Jinja2 if available (recommended for complex templates)
- Load templates from JSON files in a directory (simple format)
- A few built-in example templates to get started

Template JSON format (simple):
[
  {
    "id": "gdpr:data-retention",
    "title": "GDPR — Data retention notice",
    "text": "Personal data will be retained for {retention_period} months to satisfy legal obligations..."
  },
  ...
]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "Template",
    "TemplateRegistry",
    "register_template",
    "get_template",
    "render_template",
    "list_templates",
    "load_templates_from_dir",
]

# -------------------------
# Template dataclass
# -------------------------
@dataclass
class Template:
    id: str
    title: str
    text: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Registry
# -------------------------
class TemplateRegistry:
    def __init__(self) -> None:
        self._templates: Dict[str, Template] = {}

    def register(self, tpl: Template, *, overwrite: bool = False) -> None:
        if tpl.id in self._templates and not overwrite:
            raise KeyError(f"Template with id '{tpl.id}' already registered")
        self._templates[tpl.id] = tpl
        logger.debug("Registered template: %s", tpl.id)

    def get(self, tpl_id: str) -> Optional[Template]:
        return self._templates.get(tpl_id)

    def list_ids(self) -> List[str]:
        return list(self._templates.keys())

    def list_templates(self) -> List[Template]:
        return list(self._templates.values())

    def clear(self) -> None:
        self._templates.clear()

    def load_from_iterable(self, items: Iterable[dict], *, overwrite: bool = False) -> None:
        for obj in items:
            if not isinstance(obj, dict):
                logger.debug("Skipping non-dict template entry: %r", obj)
                continue
            tpl = Template(
                id=str(obj["id"]),
                title=str(obj.get("title", "")),
                text=str(obj.get("text", "")),
                tags=list(obj.get("tags", [])),
                metadata=dict(obj.get("metadata", {})),
            )
            self.register(tpl, overwrite=overwrite)


# module-level default registry
_default_registry = TemplateRegistry()


def register_template(tpl: Template, *, overwrite: bool = False) -> None:
    """Register a Template into the default registry."""
    _default_registry.register(tpl, overwrite=overwrite)


def get_template(tpl_id: str) -> Optional[Template]:
    """Retrieve a template by id from the default registry."""
    return _default_registry.get(tpl_id)


def list_templates() -> List[Template]:
    """Return all registered templates (default registry)."""
    return _default_registry.list_templates()


# -------------------------
# Rendering utilities
# -------------------------
class _SafeDict(dict):
    """dict that returns a placeholder for missing keys to avoid KeyError in format_map."""
    def __missing__(self, key):
        return "{" + key + "}"


def _render_with_str_format(text: str, ctx: Dict[str, Any]) -> str:
    """
    Render using Python's str.format_map safely.
    Missing keys are left as {key}.
    """
    try:
        return text.format_map(_SafeDict(**(ctx or {})))
    except Exception as e:
        # Fallback: attempt to be permissive and return partially formatted text
        logger.debug("str.format_map failed (%s); returning original text. ctx=%r", e, ctx)
        return text


# if jinja2 available, prefer it for rendering (more powerful)
try:
    import jinja2  # type: ignore

    def _render_with_jinja(text: str, ctx: Dict[str, Any]) -> str:
        tpl = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(text)
        return tpl.render(**(ctx or {}))

    _PREFERRED_RENDERER = _render_with_jinja
    logger.debug("Jinja2 available — using it for template rendering")
except Exception:
    _PREFERRED_RENDERER = _render_with_str_format
    logger.debug("Jinja2 not available — using safe str.format rendering")


def render_template(tpl_id: str, ctx: Optional[Dict[str, Any]] = None) -> str:
    """
    Render the template with the provided context.

    - If template not found, raises KeyError.
    - Uses Jinja2 when available; otherwise falls back to safe str.format_map.
    """
    tpl = get_template(tpl_id)
    if tpl is None:
        raise KeyError(f"Template not found: {tpl_id}")
    # use selected renderer
    try:
        return _PREFERRED_RENDERER(tpl.text, ctx or {})
    except Exception as e:
        logger.warning("Template rendering failed for %s: %s — falling back to safe format", tpl_id, e)
        return _render_with_str_format(tpl.text, ctx or {})


# -------------------------
# Loading from directory
# -------------------------
def load_templates_from_dir(path: str, *, pattern: str = "*.json", overwrite: bool = False) -> int:
    """
    Load template JSON files from a directory into the default registry.

    Each file may contain either a single template object or a list of template objects.
    Returns the number of templates loaded.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    count = 0
    for file in sorted(p.glob(pattern)):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            objs = data if isinstance(data, list) else [data]
            _default_registry.load_from_iterable(objs, overwrite=overwrite)
            count += len(objs)
            logger.debug("Loaded %d templates from %s", len(objs), file)
        except Exception as e:
            logger.warning("Failed to load templates from %s: %s", file, e)
    return count


# -------------------------
# Built-in example templates
# -------------------------
_builtin_templates = [
    Template(
        id="gdpr:data-retention",
        title="GDPR — Data retention notice",
        text=(
            "Personal data will be retained for {retention_period} months to satisfy applicable "
            "legal obligations and for legitimate business purposes. After the retention period, "
            "data will be securely deleted or anonymized."
        ),
        tags=["gdpr", "retention", "privacy"],
    ),
    Template(
        id="hipaa:access-control",
        title="HIPAA — Access control policy snippet",
        text=(
            "Access to electronic protected health information (ePHI) must be restricted to "
            "authorized personnel only. Controls include unique user IDs, role-based access, "
            "and audit logging. Review access permissions every {review_interval} days."
        ),
        tags=["hipaa", "security", "access-control"],
    ),
    Template(
        id="pci:cardholder-data",
        title="PCI — Cardholder data handling",
        text=(
            "Cardholder data must be minimized and protected. Sensitive authentication data "
            "must not be stored post-authorization. Use strong encryption for data at rest "
            "and in transit. Mask PAN when displayed."
        ),
        tags=["pci", "payment", "cardholder"],
    ),
]

# register built-ins on import
_default_registry.load_from_iterable([t.__dict__ for t in _builtin_templates], overwrite=False)
logger.debug("Registered %d builtin compliance templates", len(_builtin_templates))
