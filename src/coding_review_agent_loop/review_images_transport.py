"""Invocation-local ownership of sealed image inputs; no workflow authority."""
from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
import stat
import uuid

from .review_images_config import ImageLimits, delivery_limits
from .review_images_manifest import SnapshotSet, schema


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ImageInventoryEntry:
    path: str
    digest: str
    size: int
    fd: int = field(repr=False)


@dataclass(frozen=True)
class ImageBinding:
    head_sha: str
    config_digest: str
    snapshot_digest: str
    seat: str
    provider: str
    configured_model: str
    effort: str
    contract_id: str
    executable_digest: str
    acquisition_id: str
    attempt_id: str
    inventory: tuple[tuple[str, str], ...]
    # (logical id, screenshot inventory index, optional baseline index).
    references: tuple[tuple[str, int, int | None], ...]
    scope: str = "visible-defects"


@dataclass(frozen=True)
class ImageDeliveryProof:
    binding: ImageBinding
    inventory_digests: tuple[str, ...]
    session_id: str
    turn_id: str
    reported_models: tuple[str, ...]
    model_evidence_source: str

    def require(self, binding: ImageBinding) -> None:
        if (self.binding != binding or not self.session_id or not self.turn_id
                or self.inventory_digests != tuple(d for _, d in binding.inventory)
                or not self.reported_models
                or any(m != binding.configured_model for m in self.reported_models)):
            raise ValueError("incomplete image delivery or current-turn model evidence")


@dataclass(frozen=True)
class ImageInputs:
    """Own duplicated read-only descriptors until the complete acquisition exits."""
    binding: ImageBinding
    capability: object
    entries: tuple[ImageInventoryEntry, ...]
    manifest_json: bytes
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def read(self, index: int) -> bytes:
        if self._closed:
            raise ValueError("image acquisition is closed")
        entry = self.entries[index]
        data = os.pread(entry.fd, entry.size + 1, 0)
        if len(data) != entry.size or digest(data) != entry.digest:
            raise ValueError("sealed image changed or disappeared")
        return data

    def paths(self) -> tuple[str, ...]:
        # Native ingestion only. Never grant the agent shell /proc access.
        for index in range(len(self.entries)):
            self.read(index)
        return tuple(f"/proc/{os.getpid()}/fd/{e.fd}" for e in self.entries)

    def close(self) -> None:
        if not self._closed:
            object.__setattr__(self, "_closed", True)
            for entry in self.entries:
                try:
                    os.close(entry.fd)
                except OSError:
                    pass

    def __enter__(self):
        if self._closed:
            raise ValueError("image acquisition is closed")
        return self

    def __exit__(self, *_):
        self.close()


def acquire_images(snapshots: SnapshotSet, capability, *, head_sha: str,
                   config_digest: str, snapshot_digest: str,
                   limits: ImageLimits | None = None) -> ImageInputs:
    """Validate the capture API without opening any manifest-named path."""
    if (snapshots.head_sha != head_sha or snapshots.config_digest != config_digest
            or snapshots.digest != snapshot_digest):
        raise ValueError("snapshot identity mismatch")
    limits = delivery_limits(limits)
    manifest, paths = schema(snapshots.manifest_json, head_sha, limits)
    names = tuple(f.path for f in snapshots.files)
    if names != tuple(sorted(set(names))) or set(names) != set(paths):
        raise ValueError("snapshot inventory mismatch")
    canonical = json.dumps({"manifest": manifest, "config": config_digest,
                            "files": [(f.path, f.digest) for f in snapshots.files]},
                           sort_keys=True, separators=(",", ":")).encode()
    if digest(canonical) != snapshot_digest:
        raise ValueError("snapshot digest mismatch")
    owned = []
    try:
        for file in snapshots.files:
            if file._closed or not 0 < file.size <= limits.file_bytes:
                raise ValueError("closed or oversized snapshot")
            fd = os.dup(file.fd)
            owned.append(ImageInventoryEntry(file.path, file.digest, file.size, fd))
            info = os.fstat(fd)
            seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
            if (not stat.S_ISREG(info.st_mode) or info.st_size != file.size
                    or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
                    or fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals != seals):
                raise ValueError("snapshot is not sealed and read-only")
            data = os.pread(fd, file.size + 1, 0)
            if len(data) != file.size or digest(data) != file.digest:
                raise ValueError("snapshot content mismatch")
            # Capture validated PNGs; also check their bound dimensions here.
            from .review_images_png import validate_png
            if validate_png(data, dimension=limits.dimension, pixels=limits.pixels) != (file.width, file.height):
                raise ValueError("snapshot PNG dimensions mismatch")
            import struct
            if (data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24
                    or struct.unpack(">II", data[16:24]) != (file.width, file.height)
                    or not 0 < file.width <= limits.dimension or not 0 < file.height <= limits.dimension):
                raise ValueError("unsupported image dimensions/format")
        by_name = {file.path: file for file in snapshots.files}
        # Count every screenshot/baseline reference, exactly as capture publish
        # does. Shared baselines must not bypass a lower aggregate policy.
        total = sum(by_name[path].size for path in paths)
        pixels = sum(by_name[path].width * by_name[path].height for path in paths)
        if total > limits.total_bytes or pixels > limits.total_pixels or len(owned) > 2 * limits.entries:
            raise ValueError("image inventory exceeds contract limits")
        index = {name: i for i, name in enumerate(names)}
        refs = tuple((image["id"], index[image["path"]],
                      index[image["baseline_path"]] if "baseline_path" in image else None)
                     for image in manifest["images"])
        binding = ImageBinding(head_sha, config_digest, snapshot_digest, capability.seat,
                               capability.provider, capability.model, capability.effort,
                               capability.contract_id, capability.executable_digest,
                               str(uuid.uuid4()), str(uuid.uuid4()),
                               tuple((f.path, f.digest) for f in owned), refs)
        return ImageInputs(binding, capability, tuple(owned), snapshots.manifest_json)
    except BaseException:
        for entry in owned:
            os.close(entry.fd)
        raise
