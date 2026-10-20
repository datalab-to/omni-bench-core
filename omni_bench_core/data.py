"""Datasets: how they are found, downloaded and read, and how rows are chosen.

A dataset is a folder, or `hf://datasets/<org>/<name>[@<revision>]`, downloaded whole on first use. It holds:

    manifest.parquet       one row per thing to predict: its id, its input files and its gold file
    <any folder>/<file>    the input files; several rows may share one
    <any folder>/<id>.json each row's gold, named after its row

- `Row`: what the core reads of a row (its `id`);
- `read_manifest`: a manifest's rows, checked;
- `download_dataset`, `dataset_id`, `commit` and `manifest_path`: where a dataset is and what it is; and
- `selection` and `select`: a choice of rows (`only`, `sample`, `limit`).
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from omni_bench_core import errors

log = logging.getLogger(__name__)

HF_PREFIX = "hf://datasets/"
MANIFEST = "manifest.parquet"
DOWNLOAD_ATTEMPTS = 5


class Row(Protocol):
    """What the core reads of a row: its `id`, which names the row's prediction, record and gold. The task's row
    type holds everything else; a scorer reads what it needs through `Scorer.row_fields`."""

    @property
    def id(self) -> str: ...


def hf_dataset(dataset: str | Path) -> tuple[str, str | None] | None:
    """The repo and revision of an `hf://` address, or None when `dataset` is a folder."""
    s = str(dataset)
    form = f"{HF_PREFIX}<org>/<name>[@<revision>]"
    if isinstance(dataset, Path) and s.startswith("hf:"):
        raise ValueError(f"the dataset {s!r} was made a Path, which folds the '//' out of an address; "
                         f"pass it as a str: {form}")
    if not s.startswith("hf://"):
        return None
    org, _, name = s.removeprefix(HF_PREFIX).partition("/")
    name, at, revision = name.partition("@")
    if not s.startswith(HF_PREFIX) or not org or not name or "/" in name or (at and not revision):
        raise ValueError(f"{s!r} does not name a HuggingFace dataset; write it as {form}")
    return f"{org}/{name}", revision or None


@functools.cache
def download_dataset(dataset: str | Path) -> Path:
    """The dataset as a folder on disk. An `hf://` dataset is downloaded whole, at one commit, into the
    HuggingFace cache, even when only a few rows are run; a later run finds it there."""
    if (at := hf_dataset(dataset)) is None:
        root = Path(dataset).expanduser()
        if not root.is_dir():
            raise ValueError(f"no dataset folder at {root}; pass a folder or {HF_PREFIX}<org>/<name>")
        return root
    repo, revision = at
    huggingface_hub = hub()
    from huggingface_hub import errors as hf_errors

    cached = _cached(repo, revision)
    # Note: a cached dataset still asks the Hub for its latest commit, but shows no progress bars for files
    # it already has.
    if cached: huggingface_hub.utils.disable_progress_bars()
    if huggingface_hub.get_token() is None and not cached:
        log.info("downloading %s, all of it, once; later runs use the copy in the HuggingFace cache. Tip: set "
                 "HF_TOKEN (or run `hf auth login`) to download faster and avoid the Hub's rate limits", dataset)

    # Note: the Hub drops some of a dataset's hundreds of requests, and a new attempt resumes where the last
    # stopped; a missing repo or missing access would fail the same way every time, so it fails at once.
    try:
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                return Path(huggingface_hub.snapshot_download(repo, repo_type="dataset", revision=revision,
                                                              max_workers=4))
            except (hf_errors.RepositoryNotFoundError, hf_errors.GatedRepoError,
                    hf_errors.RevisionNotFoundError) as exc:
                raise ValueError(f"can't read dataset {dataset}: {type(exc).__name__}; check the repo and "
                                 "revision, and that you're logged in with access (hf auth login)") from None
            except (hf_errors.LocalEntryNotFoundError, hf_errors.OfflineModeIsEnabled) as exc:
                how = ("HF_HUB_OFFLINE is set; unset it" if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")
                       else "the Hub can't be reached; connect")
                raise ValueError(f"can't read dataset {dataset}: it isn't cached and {how}, or pass a folder "
                                 f"({type(exc).__name__})") from None
            except Exception as exc:
                if attempt == DOWNLOAD_ATTEMPTS - 1: raise
                log.warning("downloading %s failed (%s: %s); trying again in %ds", dataset, type(exc).__name__,
                            exc, 30 * (attempt + 1))
            time.sleep(30 * (attempt + 1))
    finally:
        if cached: huggingface_hub.utils.enable_progress_bars()
    raise AssertionError("unreachable")


def hub() -> Any:
    """The `huggingface_hub` module, which an `hf://` dataset needs."""
    try:
        import huggingface_hub
    except ImportError as exc:
        exc.add_note("reading an hf:// dataset needs huggingface_hub: pip install huggingface_hub (a benchmark "
                     "package's [benchmark] extra includes it), or pass a dataset folder")
        raise
    return huggingface_hub


def _cached(repo: str, revision: str | None) -> bool:
    """Whether the dataset is in the HuggingFace cache, asking nothing of the Hub."""
    try:
        hub().snapshot_download(repo, repo_type="dataset", revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 -- any reason it isn't there means it isn't cached
        return False
    return True


def dataset_id(dataset: str | Path) -> str:
    """Which dataset this is, ignoring its revision: an `hf://` repo, or a folder's full path."""
    if (at := hf_dataset(dataset)) is not None:
        return f"{HF_PREFIX}{at[0]}"
    return str(Path(dataset).expanduser().resolve())


def commit(dataset: str | Path, root: Path) -> str | None:
    """The commit an `hf://` dataset was read at (its downloaded folder's name); None for a folder."""
    return root.name if hf_dataset(dataset) is not None else None


def manifest_path(manifest: str) -> str:
    """A manifest's path inside its dataset, written one way. Refuses a path outside the dataset."""
    m = PurePosixPath(manifest)
    if m.is_absolute() or ".." in m.parts:
        raise ValueError(f"manifest {manifest} must be a path inside the dataset, relative to it, "
                         "e.g. manifests/sample.parquet")
    return m.as_posix()


@functools.cache
def read_manifest(dataset: str | Path, manifest: str = MANIFEST, *, columns: tuple[str, ...],
                  id: str = "id", inputs: tuple[str, ...] = (), gold: str | None = "gt_path"
                  ) -> tuple[Path, tuple[dict[str, Any], ...]]:
    """The dataset's folder, and its manifest's rows, checked, in order. Each id must be a file name, unique
    ignoring case; each path must be relative and inside the dataset; and a gold file must be `<id>.json`.

    Args:
      columns: the columns every row must have, besides the ones below.
      id: the column holding each row's id.
      inputs: the columns holding paths to input files, made full paths.
      gold: the column holding the path to the row's gold file, made a full path.

    Returns:
      The folder, and each row as a dict of `id`, `inputs`, `gold` and `columns`, plus `extra`, every other
      column, unread.
    """
    import pyarrow.parquet as pq

    manifest = manifest_path(manifest)
    _on_hub(dataset, manifest)
    root = download_dataset(dataset)
    path = root / manifest
    need = tuple(dict.fromkeys((id, *inputs, *([gold] if gold else []), *columns)))
    if not path.exists():
        raise ValueError(f"no manifest at {path}; a manifest is a parquet file inside the dataset with "
                         f"one row per input, with columns {', '.join(need)}")
    table = pq.read_table(path)
    if absent := [c for c in need if c not in table.column_names]:
        raise ValueError(f"{path} has no column {', '.join(absent)}; its rows need {', '.join(need)}, "
                         f"and it has {', '.join(table.column_names) or 'none'}")
    rows = tuple(row(r, n, path, root, need, id=id, inputs=inputs, gold=gold)
                 for n, r in enumerate(table.to_pylist()))
    if not rows:
        raise ValueError(f"the manifest {path} names no rows")
    unique([r[id] for r in rows], path)
    return root, rows


def row(r: dict[str, Any], n: int, path: Path, root: Path, need: tuple[str, ...], *, id: str,
        inputs: tuple[str, ...], gold: str | None) -> dict[str, Any]:
    """Manifest row `n`, checked, with full paths and its other columns in `extra`. `path` is for messages."""
    i = r[id]
    if not isinstance(i, str) or not i or i in (".", "..") or Path(i).name != i:
        raise ValueError(f"{path}: row {n + 1} has {id} {i!r}, which is not a file name; it names the "
                         "row's prediction and record, so it is one path component")
    for c in inputs:
        if not _inside(r[c]):
            raise ValueError(f"{path}: row {i} has {c} {r[c]!r}; give a relative path to a file inside the "
                             "dataset")
        r[c] = root / r[c]
    if gold:
        if not _inside(r[gold], i):
            raise ValueError(f"{path}: row {i} has {gold} {r[gold]!r}; give a relative path inside the "
                             f"dataset named for its row, e.g. gold/{i}.json")
        r[gold] = root / r[gold]
    return {**{c: r[c] for c in need}, "extra": {k: v for k, v in r.items() if k not in need}}


def unique(ids: Sequence[str], path: Path) -> None:
    """Refuses an id that appears twice ignoring case, since on macOS and Windows the two would be one file."""
    by: dict[str, list[str]] = {}
    for i in ids: by.setdefault(i.casefold(), []).append(i)
    if twice := [xs for xs in by.values() if len(xs) > 1]:
        raise ValueError(f"{path}: ids more than once, ignoring case: "
                         + "; ".join(" and ".join(xs) for xs in twice[:3]))


def selection(limit: int = 0, sample: int = 0, only: Iterable[str] | None = None) -> dict[str, Any]:
    """A choice of rows, written one way (only what was given, names sorted), as a run's question records it.
    `{}` is every row."""
    out: dict[str, Any] = {}
    if (o := _names(only, "only")) is not None: out["only"] = sorted(o)
    if sample: out["sample"] = sample
    if limit: out["limit"] = limit
    return out


def select[R: Row](rows: Sequence[R], limit: int = 0, sample: int = 0,
                   only: Iterable[str] | None = None) -> tuple[R, ...]:
    """The rows to run, in their order, filtered in this order:

    1. `only`: the rows with these ids;
    2. `sample`: this many rows, evenly spaced over the manifest, so two systems get the same rows; and
    3. `limit`: the first this many rows.

    `sample` and `limit` can't both be given."""
    only = _names(only, "only")
    for name, n in (("limit", limit), ("sample", sample)):
        if isinstance(n, bool) or not isinstance(n, int): raise TypeError(f"{name} is a whole number, not {n!r}")
        if n < 0: raise ValueError(f"{name} is {n}; it counts rows, so give 0 (no limit) or more")
    if limit and sample:
        raise ValueError(f"give sample ({sample}) or limit ({limit}), not both: sample spreads its rows over "
                         "the manifest, limit takes the first ones")
    xs = tuple(rows)
    if only is not None:
        if bad := sorted(only - {r.id for r in xs}):
            raise ValueError(f"rows asked for that aren't in the dataset: {errors.counted(len(bad), 'row')}, "
                             f"e.g. {bad[:3]}")
        xs = tuple(r for r in xs if r.id in only)
    if sample and xs:
        k = min(sample, len(xs))
        xs = tuple(xs[i * len(xs) // k] for i in range(k))
    if not (xs := xs[:limit] if limit else xs):
        raise ValueError("no rows selected; check only, sample and limit")
    return xs


def _names(xs: Iterable[str] | None, what: str) -> frozenset[str] | None:
    if isinstance(xs, str):
        raise TypeError(f"{what} is a list of names, not the string {xs!r}; pass [{xs!r}]")
    return None if xs is None else frozenset(xs)


def _inside(path: Any, id: str | None = None) -> bool:
    """Whether `path` is a relative path to a file inside the dataset, named `<id>.json` when `id` is given."""
    if not isinstance(path, str) or not path.strip(): return False
    p = Path(path)
    return (not p.is_absolute() and ".." not in p.parts and bool(p.suffix)
            and (id is None or (p.stem == id and p.suffix == ".json")))


def _on_hub(dataset: str | Path, manifest: str) -> None:
    """Refuses an `hf://` dataset with no `manifest` before downloading it. When the Hub doesn't answer, the
    download goes ahead."""
    if (at := hf_dataset(dataset)) is None: return
    huggingface_hub = hub()

    try:
        # Note: the Hub shows a private dataset this machine can't read as missing, so the message says both.
        if not huggingface_hub.repo_exists(at[0], repo_type="dataset"):
            raise ValueError(f"the Hugging Face Hub shows no dataset {at[0]}: it doesn't exist, or it is private "
                             "and this machine has no token that can read it (set HF_TOKEN)")
        missing = not huggingface_hub.file_exists(at[0], manifest, repo_type="dataset", revision=at[1])
    except OSError:
        missing = False
    if missing:
        raise ValueError(f"there is no {manifest} in {dataset}; name another manifest, or another dataset")
