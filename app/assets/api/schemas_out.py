from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class Asset(BaseModel):
    """API view of an asset. Maps to DB ``AssetReference`` joined with its ``Asset`` blob;
    ``id`` here is the AssetReference id, not the content-addressed Asset id."""

    id: str
    name: str = Field(
        ...,
        deprecated=True,
        description="Reference label, often caller-provided or derived from the filename. Deprecated for storage path/display semantics; use `loader_path`, `logical_path`, and `display_name` when present.",
    )
    hash: str | None = None
    loader_path: str | None = Field(
        default=None,
        description="In-root loader path for filesystem-backed assets: the path relative to its storage root with the top-level model category dropped (e.g. `models/checkpoints/foo/bar.safetensors` -> `foo/bar.safetensors`). This is the value model loaders consume. `None` when the file is not within a recognized root or model category.",
    )
    logical_path: str | None = Field(
        default=None,
        description="Runtime storage locator for filesystem-backed assets, using Comfy storage namespaces such as `input/`, `output/`, `temp/`, or `models/` (e.g. `models/checkpoints/foo/bar.safetensors`). Not an absolute filesystem path, unique identity, or model loader path.",
    )
    display_name: str | None = Field(
        default=None,
        description="Human-facing label derived from `logical_path`, usually the path below the top-level storage namespace. Not unique.",
    )
    asset_hash: str | None = None
    size: int | None = None
    mime_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    preview_url: str | None = None
    preview_id: str | None = None  # references an asset_reference id, not an asset id
    user_metadata: dict[str, Any] = Field(default_factory=dict)
    is_immutable: bool = False
    metadata: dict[str, Any] | None = None
    job_id: str | None = None
    prompt_id: str | None = None  # deprecated: use job_id
    created_at: datetime
    updated_at: datetime
    last_access_time: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("created_at", "updated_at", "last_access_time")
    def _serialize_datetime(self, v: datetime | None, _info):
        return v.isoformat() if v else None


class AssetCreated(Asset):
    created_new: bool


class AssetsList(BaseModel):
    assets: list[Asset]
    total: int
    has_more: bool
    # Opaque cursor for the next page. Omitted when there are no more results.
    next_cursor: str | None = None


class TagUsage(BaseModel):
    name: str
    count: int


class TagsList(BaseModel):
    tags: list[TagUsage] = Field(default_factory=list)
    total: int
    has_more: bool


class TagsAdd(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    added: list[str] = Field(default_factory=list)
    already_present: list[str] = Field(default_factory=list)
    total_tags: list[str] = Field(default_factory=list)


class TagsRemove(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    removed: list[str] = Field(default_factory=list)
    not_present: list[str] = Field(default_factory=list)
    total_tags: list[str] = Field(default_factory=list)


class TagHistogram(BaseModel):
    tag_counts: dict[str, int]
