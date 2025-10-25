# -*- coding: utf-8 -*-
import os
from typing import List

from azure.storage.blob import BlobServiceClient

def _norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("/")

def list_blobs_with_prefix(conn_str: str, container: str, prefix: str) -> List[str]:
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cont = bsc.get_container_client(container)
    prefix = _norm(prefix)
    return [b.name for b in cont.list_blobs(name_starts_with=prefix)]

def download_blobs(conn_str: str, container: str, blobs: List[str], local_dir: str) -> List[str]:
    """
    Download a list of blob paths into local_dir (flat). Safe for Windows filenames.
    - Sanitizes invalid characters (<>:\"/\\|?*)
    - Skips empty basenames (virtual directories)
    - Ensures parent dir exists
    - De-dups colliding names by appending a numeric suffix
    """
    os.makedirs(local_dir, exist_ok=True)

    def _sanitize_win(name: str) -> str:
        bad = '<>:"/\\|?*'
        name = name.replace(":", "_")
        for ch in bad:
            name = name.replace(ch, "_")
        name = name.rstrip(" .")
        reserved = {
            "CON","PRN","AUX","NUL",
            *(f"COM{i}" for i in range(1,10)),
            *(f"LPT{i}" for i in range(1,10)),
        }
        base, ext = os.path.splitext(name)
        if base.upper() in reserved:
            base = f"_{base}"
        MAXLEN = 180
        if len(base) > MAXLEN:
            base = base[:MAXLEN]
        return base + ext

    bsc = BlobServiceClient.from_connection_string(conn_str)
    cont = bsc.get_container_client(container)

    used = set()
    out_paths: List[str] = []

    for blob in blobs:
        blob = _norm(blob)
        base = os.path.basename(blob)
        if not base:
            continue

        fname = _sanitize_win(base)
        fn = fname
        i = 1
        while fn.lower() in used or os.path.exists(os.path.join(local_dir, fn)):
            stem, ext = os.path.splitext(fname)
            fn = f"{stem}_{i}{ext}"
            i += 1
        used.add(fn.lower())

        dst = os.path.join(local_dir, fn)
        try:
            with open(dst, "wb") as f:
                f.write(cont.download_blob(blob).readall())
        except OSError as e:
            raise OSError(f"Failed to write '{dst}' for blob '{blob}': {e}") from e

        out_paths.append(dst)

    return out_paths

def download_prefix_flat(conn_str: str, container: str, prefix: str, local_dir: str) -> List[str]:
    blobs = list_blobs_with_prefix(conn_str, container, prefix)
    return download_blobs(conn_str, container, blobs, local_dir)

def smart_download_single_blob(conn_str: str, container: str, blob_name: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cont = bsc.get_container_client(container)
    with open(local_path, "wb") as f:
        f.write(cont.download_blob(_norm(blob_name)).readall())
    return local_path
