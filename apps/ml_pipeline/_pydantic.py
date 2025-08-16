# sentenial-x/apps/ml_pipeline/_pydantic.py
"""
Pydantic compatibility layer (v1 <-> v2)

Usage (example):
    from ._pydantic import (
        BaseModel, Field, ValidationError, TypeAdapter,
        field_validator, model_validator, ConfigDict,
        model_dump, model_validate, to_json, pydantic_version_major,
    )

Then write code against these shims and it will run on both Pydantic v1 and v2.
"""

from __future__ import annotations
from typing import Any, Callable, Iterable, Optional, Type, TypeVar, Union, get_args, get_origin

# --- Detect version ---------------------------------------------------------
try:
    import pydantic as _pyd
    from pydantic import BaseModel, Field, ValidationError  # type: ignore
    _PYDANTIC_VERSION = getattr(_pyd, "__version__", "2")
    pydantic_version_major = int(str(_PYDANTIC_VERSION).split(".", 1)[0])
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "Pydantic is required but could not be imported. Install pydantic>=1.10."
    ) from _e


# Public names exported by this module
__all__ = [
    "BaseModel",
    "Field",
    "ValidationError",
    "TypeAdapter",
    "field_validator",
    "model_validator",
    "ConfigDict",
    "model_dump",
    "model_validate",
    "to_json",
    "pydantic_version_major",
]


T = TypeVar("T")


# --- v2 API -----------------------------------------------------------------
if pydantic_version_major >= 2:
    # Native v2 imports
    from pydantic import TypeAdapter  # type: ignore
    from pydantic import RootModel as _RootModel  # noqa: F401  (if you need it elsewhere)
    from pydantic import ConfigDict  # type: ignore
    from pydantic import field_validator, model_validator  # type: ignore

    def model_dump(obj: BaseModel, **kwargs: Any) -> dict:
        """
        v2: BaseModel.model_dump(...)
        """
        return obj.model_dump(**kwargs)

    def model_validate(cls: Type[T], data: Any, **kwargs: Any) -> T:
        """
        v2: BaseModel.model_validate(...)
        """
        return cls.model_validate(data, **kwargs)  # type: ignore[attr-defined]

    def to_json(obj: BaseModel, **kwargs: Any) -> str:
        """
        v2: BaseModel.model_dump_json(...)
        """
        return obj.model_dump_json(**kwargs)


# --- v1 API (shims) ---------------------------------------------------------
else:
    # v1 equivalents / shims
    from pydantic.tools import parse_obj_as  # type: ignore
    from pydantic import validator as _v1_validator  # type: ignore
    from pydantic import root_validator as _v1_root_validator  # type: ignore

    # v1 has no ConfigDict; accept dict as a stand-in
    class ConfigDict(dict):  # type: ignore
        """
        Minimal stand-in so code can accept ConfigDict in signatures.
        Ignored by v1 models; use class Config for v1-specific options.
        """

    class TypeAdapter:  # type: ignore
        """
        Lightweight shim mimicking pydantic.v2 TypeAdapter.

        Example:
            adapter = TypeAdapter(list[int])
            data = adapter.validate_python(["1", "2"])  # -> [1, 2] (if compatible)
        """

        def __init__(self, typ: Any):
            self.typ = typ

        def validate_python(self, data: Any) -> Any:
            # parse_obj_as handles many typing constructs in v1
            return parse_obj_as(self.typ, data)

        def dump_python(self, value: Any, **_: Any) -> Any:
            # v1 has no strict dump adapter; return value as-is
            return value

        def json_schema(self, **_: Any) -> dict:
            # Basic fallback; v1 doesn't provide the exact v2 schema API here
            return {"type": "object"}

    def field_validator(*fields: str, mode: str = "after", **kwargs: Any) -> Callable:
        """
        v1 shim for v2's @field_validator.

        Notes:
            - v1 doesn't have 'mode' (before/after); we ignore it.
            - Pass allow_reuse=True by default to match v2 typical usage.
        """
        allow_reuse = kwargs.pop("allow_reuse", True)
        return _v1_validator(*fields, allow_reuse=allow_reuse, **kwargs)

    def model_validator(mode: str = "after", **kwargs: Any) -> Callable:
        """
        v1 shim for v2's @model_validator.

        Maps to @root_validator(pre=(mode=='before')).
        """
        pre = (mode == "before")
        allow_reuse = kwargs.pop("allow_reuse", True)

        def decorator(fn: Callable) -> Callable:
            return _v1_root_validator(pre=pre, allow_reuse=allow_reuse)(fn)

        return decorator

    def model_dump(obj: BaseModel, **kwargs: Any) -> dict:
        """
        v1: BaseModel.dict(...)
        """
        # Map common v2 kwargs to v1 equivalents
        v1_kwargs = {}
        if "exclude_none" in kwargs:
            v1_kwargs["exclude_none"] = kwargs["exclude_none"]
        if "by_alias" in kwargs:
            v1_kwargs["by_alias"] = kwargs["by_alias"]
        if "exclude_unset" in kwargs:
            v1_kwargs["exclude_unset"] = kwargs["exclude_unset"]
        if "exclude_defaults" in kwargs:
            v1_kwargs["exclude_defaults"] = kwargs["exclude_defaults"]
        return obj.dict(**v1_kwargs)

    def model_validate(cls: Type[T], data: Any, **_: Any) -> T:
        """
        v1: BaseModel.parse_obj(...)
        """
        return cls.parse_obj(data)  # type: ignore[attr-defined]

    def to_json(obj: BaseModel, **kwargs: Any) -> str:
        """
        v1: BaseModel.json(...)
        """
        v1_kwargs = {}
        if "by_alias" in kwargs:
            v1_kwargs["by_alias"] = kwargs["by_alias"]
        if "exclude_none" in kwargs:
            v1_kwargs["exclude_none"] = kwargs["exclude_none"]
        return obj.json(**v1_kwargs)