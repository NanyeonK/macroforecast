from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlopen

from ..cache import atomic_copy_to_cache, atomic_write_bytes_to_cache, get_raw_file_path
from ..errors import RawDownloadError, RawParseError
from ..manager import build_raw_artifact_record, normalize_version_request
from ..manifest import append_raw_manifest_entry
from ..types import RawDatasetMetadata, RawLoadResult
from .shared_csv import parse_fred_csv

_CURRENT_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
_VINTAGE_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{vintage}.csv"


def _extract_vintage_from_zip(zip_path: Path, vintage: str, target: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [f"{vintage}.csv", f"{vintage}-md.csv"]
        match = None
        for name in zf.namelist():
            if any(name.endswith(candidate) for candidate in candidates):
                match = name
                break
        if match is None:
            raise RawDownloadError(f"vintage {vintage!r} not found in historical zip {zip_path}")
        with zf.open(match) as src:
            atomic_write_bytes_to_cache(src.read(), target)


def load_fred_md(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
    local_zip_source: str | Path | None = None,
) -> RawLoadResult:
    request = normalize_version_request("fred_md", vintage=vintage)
    target = get_raw_file_path(request, cache_root, suffix="csv")

    cache_hit = target.exists() and not force and local_source is None and local_zip_source is None
    source_url = _CURRENT_URL if request.mode == "current" else _VINTAGE_URL.format(vintage=request.vintage)

    if not cache_hit:
        try:
            if local_source is not None:
                atomic_copy_to_cache(Path(local_source), target)
                source_url = str(local_source)
            elif local_zip_source is not None:
                _extract_vintage_from_zip(Path(local_zip_source), request.vintage, target)
                source_url = str(local_zip_source)
            else:
                with urlopen(source_url) as src:
                    atomic_write_bytes_to_cache(src.read(), target)
        except Exception as exc:
            raise RawDownloadError(f"failed to obtain FRED-MD raw file for request={request}") from exc

    try:
        df, tcodes = parse_fred_csv(target)
    except Exception as exc:
        raise RawParseError(f"failed to parse FRED-MD CSV at {target}") from exc

    artifact = build_raw_artifact_record(
        request=request,
        source_url=source_url,
        local_path=target,
        file_format="csv",
        cache_hit=cache_hit,
    )
    metadata = RawDatasetMetadata(
        dataset="fred_md",
        source_family="fred-md",
        frequency="monthly",
        version_mode=request.mode,
        vintage=request.vintage,
        data_through=df.index[-1].strftime("%Y-%m") if len(df) else None,
        support_tier="stable",
    )
    result = RawLoadResult(data=df, dataset_metadata=metadata, artifact=artifact, transform_codes=tcodes)
    append_raw_manifest_entry(result, cache_root=cache_root)
    return result
