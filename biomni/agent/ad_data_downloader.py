import glob
import json
import os
import re
import time
from ftplib import FTP
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import tqdm


_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BiomniAD/1.0; +https://github.com/bioai/biomni)",
    "Accept": "*/*",
}


class DownloadTooLargeError(Exception):
    pass

def _get_resource_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "know_how", "resource"))


def _safe_filename(name: str, fallback: str) -> str:
    candidate = (name or "").strip()
    if not candidate:
        candidate = fallback
    candidate = candidate.replace("/", "_").replace("\\", "_")
    candidate = re.sub(r"[^A-Za-z0-9._\-()\[\]{} ]+", "_", candidate)
    candidate = candidate[:220].strip()
    return candidate or fallback


def _is_zenodo_record_url(uri: str) -> bool:
    parsed = urlparse(uri)
    if parsed.netloc != "zenodo.org":
        return False
    parts = [p for p in parsed.path.split("/") if p]
    return len(parts) >= 2 and parts[0] == "records" and parts[1].isdigit()


def _resolve_zenodo_record_files(uri: str, timeout: int = 20) -> list[dict[str, Any]]:
    parsed = urlparse(uri)
    parts = [p for p in parsed.path.split("/") if p]
    record_id = parts[1]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url, headers=_HTTP_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    resolved: list[dict[str, Any]] = []
    for item in payload.get("files", []):
        links = item.get("links", {})
        download_url = links.get("self") or links.get("download")
        if not download_url:
            continue
        resolved.append(
            {
                "name": item.get("key") or f"zenodo_{record_id}",
                "uri": download_url,
                "size_bytes": item.get("size"),
            }
        )
    return resolved

def _probe_size(uri: str, timeout: int = 15) -> tuple[int | None, str]:
    parsed = urlparse(uri)
    protocol = parsed.scheme.lower()

    if protocol in ["http", "https"]:
        # 1) Try HEAD
        try:
            response = requests.head(uri, headers=_HTTP_HEADERS, allow_redirects=True, timeout=timeout)
            if response.status_code == 200 and response.headers.get("Content-Length"):
                return (int(response.headers["Content-Length"]), protocol)
            if response.status_code >= 400:
                return (-response.status_code, protocol)  # negative status = server error sentinel
        except Exception:
            pass

        # 2) Range request (bytes=0-0) — servers that support it return Content-Range with total size
        try:
            range_headers = {**_HTTP_HEADERS, "Range": "bytes=0-0"}
            response = requests.get(uri, headers=range_headers, allow_redirects=True, timeout=timeout, stream=True)
            response.close()
            if response.status_code >= 400:
                return (-response.status_code, protocol)
            if response.status_code in (200, 206):
                cr = response.headers.get("Content-Range", "")  # e.g. "bytes 0-0/12345678"
                if "/" in cr:
                    total = cr.split("/")[-1].strip()
                    if total.isdigit():
                        return (int(total), protocol)
                cl = response.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return (int(cl), protocol)
        except Exception:
            pass

        # 3) Full streaming GET headers as last resort
        try:
            response = requests.get(uri, headers=_HTTP_HEADERS, stream=True, allow_redirects=True, timeout=timeout)
            response.close()
            if response.status_code >= 400:
                return (-response.status_code, protocol)
            size = response.headers.get("Content-Length")
            return (int(size) if size else None, protocol)
        except Exception:
            return (None, protocol)

    elif protocol == "ftp":
        try:
            ftp = FTP(parsed.netloc)
            ftp.login()
            size = ftp.size(parsed.path)
            ftp.quit()
            return (size, protocol)
        except Exception:
            return (None, protocol)

    elif protocol == "s3":
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if not key or key.endswith("/"):
                return (None, protocol)
            response = s3.head_object(Bucket=bucket, Key=key)
            return (response.get("ContentLength"), protocol)
        except Exception:
            return (None, protocol)

    return (None, protocol or "unknown")


def _looks_like_landing_page(response: requests.Response, expected_name: str) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    expected_lower = expected_name.lower()

    likely_data_ext = (
        ".gz", ".zip", ".tar", ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".gds", ".pdf"
    )
    expects_data = expected_lower.endswith(likely_data_ext)
    if expects_data and ("text/html" in content_type or "application/xhtml+xml" in content_type):
        return True

    return False


def _download_http(uri: str, dest_path: str, desc: str, max_size_bytes: int, expected_name: str,
                   retries: int = 3) -> int:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries):
        try:
            response = requests.get(uri, headers=_HTTP_HEADERS, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()

            if _looks_like_landing_page(response, expected_name):
                raise ValueError("URL resolved to landing page (HTML), not direct file")

            total_size = int(response.headers.get("content-length", 0))
            if total_size and total_size > max_size_bytes:
                raise DownloadTooLargeError(f"remote size {total_size} exceeds cap {max_size_bytes}")

            downloaded = 0
            with open(dest_path, "wb") as f:
                with tqdm.tqdm(total=total_size or None, unit="B", unit_scale=True, desc=desc, ncols=80, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            downloaded += len(chunk)
                            if downloaded > max_size_bytes:
                                raise DownloadTooLargeError(f"stream exceeded cap {max_size_bytes}")
                            f.write(chunk)
                            pbar.update(len(chunk))
            return downloaded

        except (DownloadTooLargeError, ValueError):
            raise
        except requests.HTTPError as e:
            # Don't retry 4xx client errors
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise
            last_exc = e
        except Exception as e:
            last_exc = e

        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    raise last_exc


def _download_ftp(uri: str, dest_path: str, desc: str, max_size_bytes: int) -> int:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    parsed = urlparse(uri)
    ftp = FTP(parsed.netloc)
    ftp.login()

    total_size = ftp.size(parsed.path) or 0
    if total_size and total_size > max_size_bytes:
        ftp.quit()
        raise DownloadTooLargeError(f"remote size {total_size} exceeds cap {max_size_bytes}")

    downloaded = 0
    with open(dest_path, "wb") as f:
        with tqdm.tqdm(total=total_size or None, unit="B", unit_scale=True, desc=desc, ncols=80, leave=False) as pbar:
            def callback(data):
                nonlocal downloaded
                downloaded += len(data)
                if downloaded > max_size_bytes:
                    raise DownloadTooLargeError(f"stream exceeded cap {max_size_bytes}")
                f.write(data)
                pbar.update(len(data))

            ftp.retrbinary(f"RETR {parsed.path}", callback)
    ftp.quit()
    return downloaded


def _download_s3(uri: str, dest_path: str, desc: str, max_size_bytes: int) -> int:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key or key.endswith("/"):
        raise ValueError("S3 URI points to a bucket/prefix, not a concrete file")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    response = s3.head_object(Bucket=bucket, Key=key)
    total_size = response.get("ContentLength", 0)
    if total_size and total_size > max_size_bytes:
        raise DownloadTooLargeError(f"remote size {total_size} exceeds cap {max_size_bytes}")

    downloaded = 0

    def _cb(bytes_transferred: int):
        nonlocal downloaded
        downloaded += bytes_transferred
        if downloaded > max_size_bytes:
            raise DownloadTooLargeError(f"stream exceeded cap {max_size_bytes}")
        pbar.update(bytes_transferred)

    with tqdm.tqdm(total=total_size or None, unit="B", unit_scale=True, desc=desc, ncols=80, leave=False) as pbar:
        s3.download_file(bucket, key, dest_path, Callback=_cb)
    return total_size or downloaded


def _expand_dataset_files(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    ds_id = dataset.get("id", "unknown")
    for file_entry in dataset.get("files", []):
        uri = file_entry.get("uri")
        if not uri:
            continue

        if _is_zenodo_record_url(uri):
            try:
                for resolved in _resolve_zenodo_record_files(uri):
                    expanded.append(
                        {
                            "dataset_id": ds_id,
                            "name": resolved.get("name") or file_entry.get("name") or "zenodo_file",
                            "uri": resolved["uri"],
                            "size_bytes": resolved.get("size_bytes"),
                            "source_entry": file_entry,
                        }
                    )
            except Exception:
                expanded.append(
                    {
                        "dataset_id": ds_id,
                        "name": file_entry.get("name") or "zenodo_record",
                        "uri": uri,
                        "size_bytes": file_entry.get("size_bytes"),
                        "source_entry": file_entry,
                    }
                )
            continue

        expanded.append(
            {
                "dataset_id": ds_id,
                "name": file_entry.get("name") or os.path.basename(urlparse(uri).path) or "unknown",
                "uri": uri,
                "size_bytes": file_entry.get("size_bytes"),
                "source_entry": file_entry,
            }
        )
    return expanded

def download_ad_catalog_data(
    data_lake_dir: str,
    max_size_mb: int = 100,
    force: bool = False,
    dry_run: bool = False,
    catalog_patterns: list[str] | None = None,
) -> dict[str, Any]:
    max_size_bytes = max_size_mb * 1024 * 1024
    resource_dir = _get_resource_dir()
    # SinaiADRD and BiomniAD_Discovery first (Zenodo-based, fast), NIAGADS last
    catalog_patterns = catalog_patterns or ["SinaiADRD.json", "BiomniAD*.json", "NIAGADS*.json"]
    catalog_paths = []
    for pat in catalog_patterns:
        catalog_paths.extend(glob.glob(os.path.join(resource_dir, pat)))

    results: dict[str, Any] = {
        "downloaded": [],
        "skipped_too_large": [],
        "skipped_error": [],
        "already_present": [],
        "skipped_prefix_uri": [],
        "total_bytes": 0,
    }

    print(f"🔍 Scanning {len(catalog_paths)} BiomniAD catalogs for files < {max_size_mb}MB...")

    for catalog_path in catalog_paths:
        updated = False
        try:
            with open(catalog_path, "r") as f:
                data = json.load(f)

            datasets = data.get("datasets", [])
            for dataset in datasets:
                ds_id = dataset.get("id", "unknown")
                expanded_files = _expand_dataset_files(dataset)

                for item in expanded_files:
                    uri = item["uri"]
                    raw_name = item["name"]
                    source_entry = item.get("source_entry", {})
                    parsed = urlparse(uri)
                    fallback_name = os.path.basename(parsed.path) or f"{ds_id}_file"
                    name = _safe_filename(raw_name, fallback_name)

                    if parsed.scheme == "s3" and (not parsed.path.strip("/") or parsed.path.endswith("/")):
                        results["skipped_prefix_uri"].append(f"{ds_id}: {name} ({uri})")
                        continue

                    # Skip files already known to be permanently unavailable
                    if not force and source_entry.get("download_error"):
                        results["skipped_error"].append(f"{ds_id}: {name} | cached: {source_entry['download_error']}")
                        continue

                    # Compute dest_path early for fast existence check
                    dest_path = os.path.join(data_lake_dir, "biomniAD", ds_id, name)
                    cached_size = item.get("size_bytes")

                    # Fast path: skip network probe if file exists with matching cached size
                    if not force and cached_size is not None and os.path.exists(dest_path):
                        local_size = os.path.getsize(dest_path)
                        if local_size == cached_size:
                            results["already_present"].append(f"{ds_id}: {name}")
                            continue

                    # Skip too-large files using cached size (no network probe needed)
                    if cached_size is not None and cached_size > max_size_bytes:
                        results["skipped_too_large"].append(f"{ds_id}: {name} ({cached_size/(1024*1024):.1f}MB)")
                        continue

                    # Only probe the network when we don't have enough info
                    if cached_size is not None:
                        size, protocol = cached_size, parsed.scheme.lower()
                    else:
                        size, protocol = _probe_size(uri)
                        if size is not None and size < 0:
                            # Negative size = HTTP error status (e.g. -404)
                            http_status = -size
                            err_msg = f"HTTP {http_status} (probe)"
                            if 400 <= http_status < 500:
                                source_entry["download_error"] = err_msg
                                updated = True
                            results["skipped_error"].append(f"{ds_id}: {name} | {err_msg}")
                            continue
                        if size is not None and source_entry.get("size_bytes") != size:
                            source_entry["size_bytes"] = size
                            updated = True

                    if size is not None and size > max_size_bytes:
                        results["skipped_too_large"].append(f"{ds_id}: {name} ({size/(1024*1024):.1f}MB)")
                        continue

                    # Slower existence check for files without cached size
                    if not force and os.path.exists(dest_path):
                        local_size = os.path.getsize(dest_path)
                        if size is None or local_size == size:
                            results["already_present"].append(f"{ds_id}: {name}")
                            continue

                    if dry_run:
                        results["downloaded"].append(f"{ds_id}: {name} (DRY RUN)")
                        continue

                    temp_path = dest_path + ".part"
                    try:
                        desc = f"📥 {name[:30]}..."
                        downloaded_size = 0
                        if protocol in ["http", "https"]:
                            downloaded_size = _download_http(uri, temp_path, desc, max_size_bytes, name)
                        elif protocol == "ftp":
                            downloaded_size = _download_ftp(uri, temp_path, desc, max_size_bytes)
                        elif protocol == "s3":
                            downloaded_size = _download_s3(uri, temp_path, desc, max_size_bytes)
                        else:
                            raise ValueError(f"unsupported protocol: {protocol}")

                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        os.replace(temp_path, dest_path)

                        if size is None:
                            source_entry["size_bytes"] = downloaded_size
                            updated = True

                        results["downloaded"].append(f"{ds_id}: {name}")
                        results["total_bytes"] += downloaded_size
                    except DownloadTooLargeError:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        results["skipped_too_large"].append(f"{ds_id}: {name} (> {max_size_mb}MB stream cap)")
                    except requests.HTTPError as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        status = e.response.status_code if e.response is not None else 0
                        err_msg = f"HTTP {status}"
                        if 400 <= status < 500:
                            # Permanent client error — cache so we skip on next run
                            source_entry["download_error"] = err_msg
                            updated = True
                        results["skipped_error"].append(f"{ds_id}: {name} | {err_msg}")
                    except Exception as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        results["skipped_error"].append(f"{ds_id}: {name} | {type(e).__name__}: {e}")

            if updated and not dry_run:
                tmp_path = catalog_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, catalog_path)

        except Exception as e:
            print(f"❌ Error processing catalog {catalog_path}: {e}")

    return results
