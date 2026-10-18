from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common.io import ensure_parent
from common.progress import tqdm


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_LIMIT = 9999
DEFAULT_FETCH_BATCH_SIZE = 200
RETRYABLE_READ_ERRORS = (
    IncompleteRead,
    RemoteDisconnected,
    ConnectionResetError,
    TimeoutError,
    ssl.SSLEOFError,
    ValueError,
)
PUBMED_CHECK_MODES = {"auto", "fast", "repair"}


def run(config: dict) -> Path:
    output_path = Path(config["pubmed_jsonl_path"])
    metadata_path = output_path.with_name("download-metadata.json")
    state_path = output_path.with_name("download-state.json")
    signature = download_signature(config)
    query_signature = download_query_signature(config)
    term_cache_signature = term_cache_query_signature(config)
    check_mode = str(config.get("pubmed_check_mode", "auto")).strip().lower()
    if check_mode not in PUBMED_CHECK_MODES:
        raise ValueError(
            f"Invalid pubmed_check_mode={check_mode!r}; "
            f"choose one of {sorted(PUBMED_CHECK_MODES)}."
        )
    checkpoint_every_batches = max(
        1, int(config.get("download_checkpoint_batches", 10) or 10)
    )

    previous_state = load_json_mapping(state_path)
    previous_term_stats = previous_state.get("terms", {})
    if not isinstance(previous_term_stats, dict):
        previous_term_stats = {}
    previous_query_matches = (
        previous_state.get("term_cache_query_signature") == term_cache_signature
    )

    api_key = config.get("ncbi_api_key", "")
    ssl_context = build_ssl_context(config)
    terms = list(dict.fromkeys(config["source_terms"] + config["target_terms"]))
    max_records_per_term = int(config.get("retmax_per_term", 0) or 0)
    batch_size = max(1, min(int(config.get("pubmed_batch_size", DEFAULT_FETCH_BATCH_SIZE)), 500))
    # The downloader always keeps only records with an abstract unless a
    # caller explicitly overrides this behavior programmatically. The case
    # configs no longer need a require_abstract parameter.
    require_abstract = bool(config.get("require_abstract", True))
    request_delay = 0.11 if api_key else 0.34
    ensure_parent(output_path)
    cache_dir = output_path.parent
    term_cache_paths = {
        term: term_cache_path(cache_dir, term, term_cache_signature)
        for term in terms
    }
    write_download_metadata(
        output_path.with_name("term-cache-manifest.json"),
        {
            "query_signature": term_cache_signature,
            "cache_directory": str(cache_dir),
            "pubmed_check_mode": check_mode,
            "terms": {term: path.name for term, path in term_cache_paths.items()},
        },
    )

    state = {
        "version": 3,
        "complete": False,
        "signature": signature,
        "query_signature": query_signature,
        "term_cache_query_signature": term_cache_signature,
        "pubmed_check_mode": check_mode,
        "terms": {},
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_download_metadata(state_path, state)

    # Older versions used a term-named JSONL file instead of
    # merged-output.jsonl. Treat any such file as a local baseline so that the
    # first run of the new cache architecture can reuse it without EFetch.
    legacy_baseline_paths = (
        [output_path]
        if output_path.exists()
        else find_legacy_jsonl_files(output_path)
    )
    legacy_record_index = build_jsonl_offset_index_for_paths(legacy_baseline_paths)
    if legacy_record_index:
        print(
            f"Indexed {len(legacy_record_index)} existing PMIDs from "
            f"{len(legacy_baseline_paths)} local JSONL file(s); "
            "they will be reused while creating term cache files."
        )

    # Records obtained earlier in this run can also be reused by later terms,
    # even before merged-output.jsonl is rebuilt.
    run_record_cache: dict[str, dict] = {}
    term_stats: dict[str, dict] = {}

    try:
        for term_index, term in enumerate(terms, start=1):
            cache_path = term_cache_paths[term]
            cache_existed = cache_path.exists()
            ensure_parent(cache_path)
            cache_path.touch(exist_ok=True)
            cache_pmids = load_existing_pmids(cache_path)
            cached_before = len(cache_pmids)
            print(
                f"PubMed term {term_index}/{len(terms)}: {term}\n"
                f"Term cache: {cache_path} ({cached_before} PMIDs already cached)"
            )

            prior_stats = previous_term_stats.get(term, {})
            if not isinstance(prior_stats, dict):
                prior_stats = {}
            try:
                prior_missing = int(prior_stats.get("missing_pmids", 0) or 0)
            except (TypeError, ValueError):
                prior_missing = 1
            prior_term_complete = (
                cache_existed
                and previous_query_matches
                and prior_stats.get("complete") is True
                and prior_missing == 0
                # Do not trust a completion record produced by the old
                # per-term 9999-record cap. Such a record has
                # expected_pmids < segment_hits and must be rechecked.
                and prior_stats.get("expected_pmids") == prior_stats.get("segment_hits")
            )
            skip_term = (
                check_mode == "fast" and cache_existed and cache_path.stat().st_size > 0
            ) or (check_mode == "auto" and prior_term_complete)
            if skip_term:
                skipped_stats = dict(prior_stats)
                skipped_stats.update(
                    {
                        "cache_file": str(cache_path),
                        "cached_pmids": len(cache_pmids),
                        "complete": True,
                        "skipped_check": True,
                        "check_mode": check_mode,
                        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                term_stats[term] = skipped_stats
                state["terms"] = term_stats
                state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_download_metadata(state_path, state)
                print(
                    f"Skipping PubMed check for {term}: retained {len(cache_pmids)} "
                    f"cached PMIDs (mode={check_mode})."
                )
                continue

            available_count, segments = plan_search_segments(
                term,
                config["start_date"],
                config["end_date"],
                api_key,
                ssl_context,
                request_delay,
            )
            segment_hit_count = sum(segment[2] for segment in segments)
            # ``retmax_per_term`` is retained for compatibility and is used as
            # the maximum number returned by a single PubMed date-segment
            # request. It must not cap the total number of records for a term:
            # date segmentation is what allows all matching PMIDs to be
            # downloaded when the term has more than 9999 results.
            download_count = segment_hit_count
            print(
                f"Found {available_count} candidate records in {len(segments)} date segments. "
                f"The segments contain {segment_hit_count} hits before PMID deduplication; "
                f"checking {download_count} PMIDs against the term cache."
            )

            term_processed = 0
            reused_count = 0
            fetched_count = 0
            skipped_cached_count = 0
            duplicate_candidate_count = 0
            unresolved_count = 0
            checkpoint_batch_count = 0
            candidate_seen_pmids: set[str] = set()

            with tqdm(
                total=download_count,
                desc=f"Download {term}",
                unit="article",
                unit_scale=True,
                dynamic_ncols=True,
            ) as progress:
                for segment_start, segment_end, segment_limit in build_segment_limits(
                    segments, download_count
                ):
                    raw_pmids = list(dict.fromkeys(
                        search_segment_pmids(
                            term,
                            segment_start,
                            segment_end,
                            api_key,
                            ssl_context,
                            request_delay,
                        )[:segment_limit]
                    ))
                    pmids = [
                        pmid for pmid in raw_pmids if pmid not in candidate_seen_pmids
                    ]
                    candidate_seen_pmids.update(raw_pmids)
                    duplicate_candidate_count += len(raw_pmids) - len(pmids)
                    for start in range(0, len(pmids), batch_size):
                        chunk = pmids[start : start + batch_size]
                        missing_pmids = [pmid for pmid in chunk if pmid not in cache_pmids]
                        skipped_cached_count += len(chunk) - len(missing_pmids)

                        rows_by_pmid: dict[str, dict] = {}
                        legacy_pmids = []
                        for pmid in missing_pmids:
                            cached_row = run_record_cache.get(pmid)
                            if cached_row is not None:
                                rows_by_pmid[pmid] = cached_row
                            elif pmid in legacy_record_index:
                                legacy_pmids.append(pmid)
                        if legacy_pmids:
                            rows_by_pmid.update(
                                read_records_from_index(
                                    legacy_record_index,
                                    legacy_pmids,
                                )
                            )

                        reused_rows = [
                            rows_by_pmid[pmid]
                            for pmid in missing_pmids
                            if pmid in rows_by_pmid
                        ]
                        if reused_rows:
                            append_jsonl_records(cache_path, reused_rows)
                            reused_count += len(reused_rows)
                            for row in reused_rows:
                                pmid = str(row.get("PMID", "")).strip()
                                if pmid:
                                    cache_pmids.add(pmid)
                                    run_record_cache.setdefault(pmid, row)

                        fetch_pmids = [
                            pmid for pmid in missing_pmids if pmid not in cache_pmids
                        ]
                        fetched_rows = []
                        if fetch_pmids:
                            fetched_rows = fetch_details(
                                fetch_pmids,
                                api_key,
                                ssl_context,
                                request_delay,
                            )
                            fetched_by_pmid = {
                                str(row.get("PMID", "")).strip(): row
                                for row in fetched_rows
                                if str(row.get("PMID", "")).strip()
                            }
                            fetched_rows = [
                                fetched_by_pmid[pmid]
                                for pmid in fetch_pmids
                                if pmid in fetched_by_pmid
                            ]
                            if fetched_rows:
                                append_jsonl_records(cache_path, fetched_rows)
                                fetched_count += len(fetched_rows)
                                for row in fetched_rows:
                                    pmid = str(row.get("PMID", "")).strip()
                                    cache_pmids.add(pmid)
                                    run_record_cache.setdefault(pmid, row)

                        unresolved_count += sum(
                            1 for pmid in missing_pmids if pmid not in cache_pmids
                        )
                        term_processed += len(chunk)
                        progress.update(len(chunk))
                        progress.set_postfix(
                            cached=len(cache_pmids),
                            reused=reused_count,
                            fetched=fetched_count,
                            missing=unresolved_count,
                        )

                        term_stats[term] = {
                            "cache_file": str(cache_path),
                            "cache_existed_at_start": cache_existed,
                            "available": available_count,
                            "segment_hits": segment_hit_count,
                            "expected_pmids": download_count,
                            "checked_pmids": term_processed,
                            "cached_pmids": len(cache_pmids),
                            "reused_from_existing": reused_count,
                            "fetched_from_pubmed": fetched_count,
                            "skipped_cached": skipped_cached_count,
                            "deduplicated_candidates": duplicate_candidate_count,
                            "missing_pmids": unresolved_count,
                            "check_mode": check_mode,
                            "complete": unresolved_count == 0
                            and term_processed >= download_count,
                        }
                        state["terms"] = term_stats
                        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                        checkpoint_batch_count += 1
                        if checkpoint_batch_count % checkpoint_every_batches == 0:
                            write_download_metadata(state_path, state)

                    duplicate_count = len(raw_pmids) - len(pmids)
                    if duplicate_count:
                        term_processed += duplicate_count
                        progress.update(duplicate_count)

            term_stats[term] = {
                "cache_file": str(cache_path),
                "cache_existed_at_start": cache_existed,
                "available": available_count,
                "segment_hits": segment_hit_count,
                "expected_pmids": download_count,
                "checked_pmids": term_processed,
                "cached_pmids": len(cache_pmids),
                "reused_from_existing": reused_count,
                "fetched_from_pubmed": fetched_count,
                "skipped_cached": skipped_cached_count,
                "deduplicated_candidates": duplicate_candidate_count,
                "missing_pmids": unresolved_count,
                "check_mode": check_mode,
                "complete": unresolved_count == 0 and term_processed >= download_count,
            }
            state["terms"] = term_stats
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_download_metadata(state_path, state)

            print(
                f"Term cache check finished: {term}; reused={reused_count}, "
                f"fetched={fetched_count}, already_cached={skipped_cached_count}, "
                f"missing={unresolved_count}"
            )

        all_terms_skipped = all(
            term_stats.get(term, {}).get("skipped_check") is True for term in terms
        )
        previous_saved_records = previous_state.get("saved_records")
        previous_missing_abstract = previous_state.get("missing_abstract_records", 0)
        can_reuse_merged_output = (
            output_path.exists()
            and previous_state.get("complete") is True
            and previous_state.get("signature") == signature
            and all_terms_skipped
            and isinstance(previous_saved_records, int)
        )
        if can_reuse_merged_output:
            try:
                missing_abstract_count = int(previous_missing_abstract or 0)
            except (TypeError, ValueError):
                missing_abstract_count = 0
            saved_count = previous_saved_records
            print(
                "No term cache changed; keeping the existing merged-output.jsonl "
                "without rescanning all term files."
            )
        else:
            print("Merging per-term JSONL files into merged-output.jsonl...")
            saved_count, missing_abstract_count = merge_term_caches(
                output_path,
                [term_cache_paths[term] for term in terms],
                require_abstract,
                legacy_baseline_paths,
            )
    except BaseException:
        state["complete"] = False
        state["terms"] = term_stats
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_download_metadata(state_path, state)
        except OSError:
            pass
        print(
            f"Per-term records already saved under {cache_dir}; "
            f"resume checkpoint: {state_path}"
        )
        raise

    state["complete"] = True
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["saved_records"] = saved_count
    state["missing_abstract_records"] = missing_abstract_count
    state["terms"] = term_stats
    write_download_metadata(state_path, state)
    write_download_metadata(
        metadata_path,
        {
            "complete": True,
            "signature": signature,
            "query_signature": query_signature,
            "term_cache_query_signature": term_cache_signature,
            "pubmed_check_mode": check_mode,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "saved_records": saved_count,
            "missing_abstract_records": missing_abstract_count,
            "terms": term_stats,
        },
    )
    print(f"Saved {saved_count} PubMed records to {output_path}")
    return output_path


def download_signature(config: dict) -> str:
    payload = {
        "version": 5,
        "source_terms": config["source_terms"],
        "target_terms": config["target_terms"],
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "retmax_per_term": int(config.get("retmax_per_term", 0) or 0),
        "require_abstract": bool(config.get("require_abstract", True)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def download_query_signature(config: dict) -> str:
    payload = {
        "version": 2,
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "retmax_per_term": int(config.get("retmax_per_term", 0) or 0),
        "require_abstract": bool(config.get("require_abstract", True)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def term_cache_query_signature(config: dict) -> str:
    """Signature for the PMID set represented by a per-term cache file."""
    payload = {
        "version": 1,
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "retmax_per_term": int(config.get("retmax_per_term", 0) or 0),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def term_cache_path(cache_dir: Path, term: str, query_signature: str) -> Path:
    """Return a human-readable, stable cache filename for one PubMed term."""
    del query_signature  # The current query is recorded in the state/manifest.
    filename = re.sub(r'[<>:"/\\|?*]', "_", term).strip()
    filename = filename or "term"
    return cache_dir / f"{filename}.jsonl"


def download_is_complete(output_path: Path, metadata_path: Path, signature: str) -> bool:
    if not output_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("complete") is True and metadata.get("signature") == signature


def write_download_metadata(path: Path, metadata: dict) -> None:
    ensure_parent(path)
    part_path = path.with_suffix(path.suffix + ".part")
    part_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part_path, path)


def load_json_mapping(path: Path) -> dict:
    """Load a JSON object for resume decisions; tolerate a missing/broken file."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_download_state(
    path: Path,
    signature: str,
    query_signature: str,
    terms: list[str],
    config: dict,
) -> dict | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict):
        return None

    if state.get("signature") == signature:
        return state

    if not reusable_checkpoint(state, query_signature, terms, config):
        return None

    print(
        "Detected a changed keyword list; reusing the existing PubMed checkpoint "
        "and retaining already downloaded PMIDs."
    )
    return state


def reusable_checkpoint(
    state: dict,
    query_signature: str,
    terms: list[str],
    config: dict,
) -> bool:
    """Return whether per-segment offsets can be reused after a config change."""
    if not isinstance(state.get("segment_offsets", {}), dict):
        return False

    saved_terms = set(state.get("terms", {}))
    if not saved_terms.issubset(set(terms)):
        return False

    stored_query_signature = state.get("query_signature")
    if stored_query_signature:
        return stored_query_signature == query_signature

    # State files written before incremental keyword updates did not store a
    # query signature. Migrate them when their segment keys still cover the
    # configured date window. This is the common case when only source_terms
    # was extended.
    return legacy_checkpoint_matches_date_window(state, config)


def legacy_checkpoint_matches_date_window(state: dict, config: dict) -> bool:
    segment_offsets = state.get("segment_offsets", {})
    if not segment_offsets:
        return True

    try:
        configured_start = parse_config_date(config["start_date"])
        configured_end = parse_config_date(config["end_date"])
    except (KeyError, ValueError):
        return False

    ranges = []
    for raw_key in segment_offsets:
        try:
            _term, start_text, end_text = str(raw_key).rsplit("|", 2)
            range_start = parse_config_date(start_text)
            range_end = parse_config_date(end_text)
        except (ValueError, TypeError):
            return False
        if range_start < configured_start or range_end > configured_end or range_start > range_end:
            return False
        ranges.append((range_start, range_end))

    return (
        min(range_start for range_start, _ in ranges) == configured_start
        and max(range_end for _, range_end in ranges) == configured_end
    )


def recover_or_archive_legacy_output(output_path: Path) -> None:
    legacy_part_path = output_path.with_suffix(output_path.suffix + ".part")
    if legacy_part_path.exists():
        if output_path.exists():
            archive_output(output_path)
        os.replace(legacy_part_path, output_path)
        print(f"Recovered legacy partial download to: {output_path}")
        return
    if output_path.exists():
        archive_output(output_path)


def archive_output(output_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_path = output_path.with_name(f"{output_path.name}.previous-{timestamp}")
    counter = 1
    while archive_path.exists():
        archive_path = output_path.with_name(f"{output_path.name}.previous-{timestamp}-{counter}")
        counter += 1
    os.replace(output_path, archive_path)
    print(f"Archived incompatible previous output to: {archive_path}")
    return archive_path


def load_existing_pmids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    seen_pmids = set()
    file_size = output_path.stat().st_size
    last_valid_offset = 0
    trailing_partial = False
    with open(output_path, "rb") as file:
        line_number = 0
        while True:
            line = file.readline()
            if not line:
                break
            line_number += 1
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if file.tell() == file_size:
                    trailing_partial = True
                    break
                raise RuntimeError(
                    f"Invalid JSONL record at {output_path}:{line_number}; refusing to append to a corrupted file."
                ) from exc
            pmid = str(row.get("PMID", "")).strip()
            if pmid:
                seen_pmids.add(pmid)
            last_valid_offset = file.tell()

    if trailing_partial:
        with open(output_path, "r+b") as file:
            file.truncate(last_valid_offset)
        print(f"Removed an incomplete trailing JSONL record from: {output_path}")
    return seen_pmids


def find_legacy_jsonl_files(output_path: Path) -> list[Path]:
    return sorted(
        path
        for path in output_path.parent.glob("*.jsonl")
        if path.name != output_path.name
        and not path.name.startswith("term-")
        and not path.name.endswith(".part")
    )


def build_jsonl_offset_index(
    output_path: Path,
) -> dict[str, tuple[Path, int, int]]:
    """Index a JSONL file without keeping all article payloads in memory."""
    if not output_path.exists():
        return {}

    index: dict[str, tuple[Path, int, int]] = {}
    file_size = output_path.stat().st_size
    last_valid_offset = 0
    trailing_partial = False
    with open(output_path, "rb") as file:
        line_number = 0
        while True:
            line_start = file.tell()
            line = file.readline()
            if not line:
                break
            line_number += 1
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if file.tell() == file_size:
                    trailing_partial = True
                    break
                raise RuntimeError(
                    f"Invalid JSONL record at {output_path}:{line_number}; "
                    "refusing to index a corrupted file."
                ) from exc
            pmid = str(row.get("PMID", "")).strip()
            if pmid:
                index.setdefault(pmid, (output_path, line_start, len(line)))
            last_valid_offset = file.tell()

    if trailing_partial:
        with open(output_path, "r+b") as file:
            file.truncate(last_valid_offset)
        print(f"Removed an incomplete trailing JSONL record from: {output_path}")
    return index


def build_jsonl_offset_index_for_paths(
    paths: list[Path],
) -> dict[str, tuple[Path, int, int]]:
    combined: dict[str, tuple[Path, int, int]] = {}
    for path in paths:
        for pmid, location in build_jsonl_offset_index(path).items():
            combined.setdefault(pmid, location)
    return combined


def read_records_from_index(
    index: dict[str, tuple[Path, int, int]],
    pmids: list[str],
) -> dict[str, dict]:
    if not pmids:
        return {}

    records: dict[str, dict] = {}
    open_files: dict[Path, object] = {}
    try:
        for pmid in pmids:
            location = index.get(pmid)
            if location is None:
                continue
            source_path, offset, length = location
            if source_path not in open_files:
                open_files[source_path] = open(source_path, "rb")
            file = open_files[source_path]
            file.seek(offset)
            line = file.read(length)
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Invalid indexed JSONL record for PMID {pmid} in {source_path}."
                ) from exc
            row_pmid = str(row.get("PMID", "")).strip()
            if row_pmid == pmid:
                records[pmid] = row
    finally:
        for file in open_files.values():
            file.close()
    return records


def append_jsonl_records(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_parent(output_path)
    with open(output_path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def iter_jsonl_records(output_path: Path):
    if not output_path.exists():
        return
    with open(output_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL record at {output_path}:{line_number}."
                ) from exc
            if isinstance(row, dict):
                yield row


def merge_term_caches(
    output_path: Path,
    cache_paths: list[Path],
    require_abstract: bool,
    baseline_paths: list[Path],
) -> tuple[int, int]:
    """Merge current term caches, preserving old merged records and deduplicating PMID."""
    ensure_parent(output_path)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    seen_pmids: set[str] = set()
    saved_count = 0
    missing_abstract_count = 0

    for baseline_path in baseline_paths:
        if baseline_path.exists():
            # Validate/truncate a possible interrupted trailing line before
            # reading the local baseline.
            load_existing_pmids(baseline_path)

    with open(part_path, "w", encoding="utf-8") as output_file:
        for baseline_path in baseline_paths:
            if not baseline_path.exists():
                continue
            for row in iter_jsonl_records(baseline_path):
                pmid = str(row.get("PMID", "")).strip()
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                if require_abstract and not str(row.get("Abstract", "")).strip():
                    missing_abstract_count += 1
                    continue
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved_count += 1

        for cache_path in cache_paths:
            if not cache_path.exists():
                raise RuntimeError(f"Missing per-term cache file: {cache_path}")
            load_existing_pmids(cache_path)
            for row in iter_jsonl_records(cache_path):
                pmid = str(row.get("PMID", "")).strip()
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                if require_abstract and not str(row.get("Abstract", "")).strip():
                    missing_abstract_count += 1
                    continue
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                saved_count += 1
        output_file.flush()
        os.fsync(output_file.fileno())

    os.replace(part_path, output_path)
    return saved_count, missing_abstract_count


def build_segment_limits(
    segments: list[tuple[date, date, int]],
    download_count: int,
) -> list[tuple[date, date, int]]:
    remaining = download_count
    limited_segments = []
    for segment_start, segment_end, segment_count in segments:
        if remaining <= 0:
            break
        limit = min(segment_count, remaining)
        if limit > 0:
            limited_segments.append((segment_start, segment_end, limit))
            remaining -= limit
    return limited_segments


def segment_key(term: str, start_date: date, end_date: date) -> str:
    return f"{term}|{format_pubmed_date(start_date)}|{format_pubmed_date(end_date)}"


def build_ssl_context(config: dict) -> ssl.SSLContext:
    ca_file = config.get("ssl_ca_file") or os.getenv("LBD_SSL_CA_FILE")
    context = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()

    # Some Python distributions use an OpenSSL CA bundle that does not include
    # a locally trusted proxy/VPN certificate. Supplement it with the native
    # Windows certificate stores while keeping hostname and chain verification.
    if os.name == "nt" and hasattr(ssl, "enum_certificates"):
        for store_name in ("ROOT", "CA"):
            try:
                certificates = ssl.enum_certificates(store_name)
            except OSError:
                continue
            for certificate, encoding, _trust in certificates:
                if encoding != "x509_asn":
                    continue
                try:
                    context.load_verify_locations(cadata=certificate)
                except ssl.SSLError:
                    continue
    return context


def read_url(request: str | Request, ssl_context: ssl.SSLContext | None = None) -> bytes:
    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(5):
        try:
            with urlopen(request, timeout=60, context=ssl_context) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in retryable_statuses or attempt == 4:
                raise
            delay = min(2**attempt, 16)
            print(f"NCBI returned HTTP {exc.code}; retrying in {delay} seconds...")
            time.sleep(delay)
        except RETRYABLE_READ_ERRORS as exc:
            if attempt == 4:
                raise
            delay = min(2**attempt, 16)
            received = len(exc.partial) if isinstance(exc, IncompleteRead) else 0
            detail = f" after receiving {received} bytes" if received else ""
            print(
                f"NCBI response was interrupted{detail} ({type(exc).__name__}); "
                f"retrying in {delay} seconds..."
            )
            time.sleep(delay)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise RuntimeError(
                    "PubMed SSL certificate verification failed. Install the proxy/VPN root certificate "
                    "in the Windows Trusted Root store, or set LBD_SSL_CA_FILE (or config ssl_ca_file) "
                    "to the trusted CA PEM file. SSL verification has not been disabled."
                ) from exc
            if attempt == 4:
                raise
            delay = min(2**attempt, 16)
            print(f"NCBI connection failed; retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError("NCBI request failed after retries")


def plan_search_segments(
    term: str,
    start_date: str,
    end_date: str,
    api_key: str,
    ssl_context: ssl.SSLContext | None,
    request_delay: float,
) -> tuple[int, list[tuple[date, date, int]]]:
    first_date = parse_config_date(start_date)
    last_date = parse_config_date(end_date)
    if first_date > last_date:
        raise ValueError(f"PubMed start_date {start_date!r} is after end_date {end_date!r}")

    initial_count = search_result_count(
        term,
        first_date,
        last_date,
        api_key,
        ssl_context,
        request_delay,
    )
    pending = [(first_date, last_date, initial_count)]
    segments = []
    split_count = 0

    while pending:
        segment_start, segment_end, count = pending.pop()
        if count <= PUBMED_SEARCH_LIMIT:
            segments.append((segment_start, segment_end, count))
            continue
        if segment_start == segment_end:
            raise RuntimeError(
                f"PubMed returned {count} records for one day ({format_pubmed_date(segment_start)}), "
                f"which exceeds the ESearch limit of {PUBMED_SEARCH_LIMIT}. Use NCBI EDirect for this query."
            )

        midpoint = segment_start + (segment_end - segment_start) // 2
        right_start = midpoint + timedelta(days=1)
        left_count = search_result_count(
            term,
            segment_start,
            midpoint,
            api_key,
            ssl_context,
            request_delay,
        )
        right_count = search_result_count(
            term,
            right_start,
            segment_end,
            api_key,
            ssl_context,
            request_delay,
        )
        split_count += 1
        if split_count % 5 == 0:
            print(
                f"Planning date segments: {len(segments)} ready, "
                f"{len(pending) + 2} pending ranges."
            )
        pending.append((right_start, segment_end, right_count))
        pending.append((segment_start, midpoint, left_count))

    return initial_count, segments


def search_result_count(
    term: str,
    start_date: date,
    end_date: date,
    api_key: str,
    ssl_context: ssl.SSLContext | None,
    request_delay: float,
) -> int:
    params = {
        "db": "pubmed",
        "retmode": "json",
        "rettype": "count",
        "retmax": 0,
        "term": build_pubmed_query(term, start_date, end_date),
    }
    if api_key:
        params["api_key"] = api_key
    payload = json.loads(read_url(f"{EUTILS}/esearch.fcgi?{urlencode(params)}", ssl_context).decode("utf-8"))
    time.sleep(request_delay)
    return int(payload.get("esearchresult", {}).get("count", 0))


def search_segment_pmids(
    term: str,
    start_date: date,
    end_date: date,
    api_key: str,
    ssl_context: ssl.SSLContext | None,
    request_delay: float,
) -> list[str]:
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": PUBMED_SEARCH_LIMIT,
        "term": build_pubmed_query(term, start_date, end_date),
    }
    if api_key:
        params["api_key"] = api_key
    payload = json.loads(read_url(f"{EUTILS}/esearch.fcgi?{urlencode(params)}", ssl_context).decode("utf-8"))
    time.sleep(request_delay)
    result = payload.get("esearchresult", {})
    count = int(result.get("count", 0))
    if count > PUBMED_SEARCH_LIMIT:
        raise RuntimeError(
            f"PubMed date segment {format_pubmed_date(start_date)} to {format_pubmed_date(end_date)} "
            f"grew to {count} records after planning. Run the download again to rebuild date segments."
        )
    pmids = [str(pmid) for pmid in result.get("idlist", [])]
    if len(pmids) < count:
        raise RuntimeError(
            f"PubMed returned {len(pmids)} PMIDs for a segment reporting {count} records: "
            f"{format_pubmed_date(start_date)} to {format_pubmed_date(end_date)}"
        )
    return pmids


def build_pubmed_query(term: str, start_date: date, end_date: date) -> str:
    return (
        f"({term}) AND "
        f'(\"{format_pubmed_date(start_date)}\"[PDAT] : \"{format_pubmed_date(end_date)}\"[PDAT])'
    )


def parse_config_date(value: str) -> date:
    for date_format in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid PubMed date {value!r}; expected YYYY/MM/DD or YYYY-MM-DD")


def format_pubmed_date(value: date) -> str:
    return f"{value.year:04d}/{value.month:02d}/{value.day:02d}"


def fetch_details(
    pmids: list[str],
    api_key: str,
    ssl_context: ssl.SSLContext | None = None,
    request_delay: float = 0.0,
) -> list[dict]:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "rettype": "abstract",
        "retmode": "xml",
        "id": ",".join(pmids),
    }
    if api_key:
        params["api_key"] = api_key
    request = Request(
        f"{EUTILS}/efetch.fcgi",
        data=urlencode(params).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "CE-LBD-PubMed-Downloader/1.0",
        },
        method="POST",
    )
    try:
        xml_text = read_url(request, ssl_context).decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and len(pmids) > 1:
            midpoint = len(pmids) // 2
            print(
                f"NCBI rejected a batch of {len(pmids)} PMIDs with HTTP 400; "
                f"retrying as {midpoint} + {len(pmids) - midpoint}."
            )
            return fetch_details(pmids[:midpoint], api_key, ssl_context, request_delay) + fetch_details(
                pmids[midpoint:], api_key, ssl_context, request_delay
            )
        raise RuntimeError(
            f"NCBI EFetch failed with HTTP {exc.code} for {len(pmids)} PMIDs: {error_body[:500]}"
        ) from exc
    except RETRYABLE_READ_ERRORS as exc:
        if len(pmids) > 1:
            midpoint = len(pmids) // 2
            print(
                f"NCBI repeatedly returned an incomplete response for {len(pmids)} PMIDs; "
                f"retrying as {midpoint} + {len(pmids) - midpoint}."
            )
            return fetch_details(pmids[:midpoint], api_key, ssl_context, request_delay) + fetch_details(
                pmids[midpoint:], api_key, ssl_context, request_delay
            )
        raise RuntimeError(
            f"NCBI repeatedly returned an incomplete response for PMID {pmids[0]}"
        ) from exc
    time.sleep(request_delay)
    try:
        return parse_pubmed_xml(xml_text)
    except ET.ParseError as exc:
        if len(pmids) > 1:
            midpoint = len(pmids) // 2
            print(
                f"NCBI returned incomplete XML for {len(pmids)} PMIDs; "
                f"retrying as {midpoint} + {len(pmids) - midpoint}."
            )
            return fetch_details(pmids[:midpoint], api_key, ssl_context, request_delay) + fetch_details(
                pmids[midpoint:], api_key, ssl_context, request_delay
            )
        raise RuntimeError(f"NCBI returned incomplete XML for PMID {pmids[0]}") from exc


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        pmid = text_or_empty(medline.find("PMID"))
        article_node = medline.find("Article")
        title = text_or_empty(article_node.find("ArticleTitle")) if article_node is not None else ""
        abstract_parts = []
        if article_node is not None:
            for abstract in article_node.findall(".//AbstractText"):
                abstract_parts.append("".join(abstract.itertext()))
        records.append({"PMID": pmid, "Title": title, "Abstract": " ".join(abstract_parts)})
    return records


def text_or_empty(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()
