from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse


DEFAULT_URL = "https://drive.google.com/drive/folders/17CrakXjvwqjY0Erh6c5YMiTZjfef2mu2?hl=ko"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
FOLDER_ID_PATTERN = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
FILE_ID_PATTERN = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")


@dataclass(frozen=True)
class DriveFolder:
    folder_id: str
    name: str
    resource_key: str | None = None


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    resource_key: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all files in a public Google Drive folder using gdown. "
            "This script avoids gdown.download_folder()'s 50-file listing limit."
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Google Drive folder URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store downloaded files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce gdown output.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already exist locally.",
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        help="User-Agent used when listing folder contents.",
    )
    return parser.parse_args()


def extract_folder_info(url: str) -> tuple[str, str | None]:
    match = FOLDER_ID_PATTERN.search(url)
    if match is None:
        raise ValueError(f"Could not extract a Google Drive folder id from: {url}")

    query = parse_qs(urlparse(url).query)
    resource_key = query.get("resourcekey", [None])[0]
    return match.group(1), resource_key


def normalize_name(name: str, fallback: str) -> str:
    normalized = " ".join(name.split()).strip()
    normalized = normalized.replace("/", "_").replace("\\", "_")
    return normalized or fallback


def build_embedded_folder_url(folder_id: str, resource_key: str | None) -> str:
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"
    if resource_key:
        url = f"https://drive.google.com/embeddedfolderview?id={folder_id}&resourcekey={resource_key}#list"
    return url


def fetch_folder_entries(
    *,
    session,
    folder_id: str,
    resource_key: str | None,
) -> tuple[list[DriveFolder], list[DriveFile]]:
    from bs4 import BeautifulSoup

    response = session.get(build_embedded_folder_url(folder_id, resource_key), timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    folders: dict[tuple[str, str | None], DriveFolder] = {}
    files: dict[tuple[str, str | None], DriveFile] = {}

    for link in soup.find_all("a", href=True):
        href = link["href"]
        title = normalize_name(link.get_text(" ", strip=True), fallback="")
        query = parse_qs(urlparse(href).query)
        child_resource_key = query.get("resourcekey", [None])[0]

        folder_match = FOLDER_ID_PATTERN.search(href)
        if folder_match is not None:
            child_id = folder_match.group(1)
            if child_id == folder_id:
                continue
            child_name = normalize_name(title, fallback=f"folder_{child_id}")
            key = (child_id, child_resource_key)
            folders[key] = DriveFolder(
                folder_id=child_id,
                name=child_name,
                resource_key=child_resource_key,
            )
            continue

        file_match = FILE_ID_PATTERN.search(href)
        if file_match is not None:
            child_id = file_match.group(1)
            child_name = normalize_name(title, fallback=f"file_{child_id}")
            key = (child_id, child_resource_key)
            files[key] = DriveFile(
                file_id=child_id,
                name=child_name,
                resource_key=child_resource_key,
            )

    if not folders and not files:
        raise RuntimeError(
            "Could not list folder contents from the embedded view. "
            "The folder may be private, access-limited, or Google may have changed the page format."
        )

    return list(folders.values()), list(files.values())


def walk_drive_tree(
    *,
    session,
    folder_id: str,
    resource_key: str | None,
    output_dir: Path,
    visited: set[tuple[str, str | None]],
) -> list[tuple[DriveFile, Path]]:
    visit_key = (folder_id, resource_key)
    if visit_key in visited:
        return []
    visited.add(visit_key)

    folders, files = fetch_folder_entries(
        session=session,
        folder_id=folder_id,
        resource_key=resource_key,
    )

    downloads = [(drive_file, output_dir) for drive_file in files]
    for folder in folders:
        child_dir = output_dir / folder.name
        downloads.extend(
            walk_drive_tree(
                session=session,
                folder_id=folder.folder_id,
                resource_key=folder.resource_key,
                output_dir=child_dir,
                visited=visited,
            )
        )
    return downloads


def file_url(file_id: str, resource_key: str | None) -> str:
    if resource_key:
        return f"https://drive.google.com/uc?id={file_id}&resourcekey={resource_key}"
    return f"https://drive.google.com/uc?id={file_id}"


def main() -> int:
    args = parse_args()

    try:
        import gdown
        import requests
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}", file=sys.stderr)
        print("Install it first with: pip install gdown", file=sys.stderr)
        return 1

    try:
        root_folder_id, root_resource_key = extract_folder_info(args.url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    try:
        downloads = walk_drive_tree(
            session=session,
            folder_id=root_folder_id,
            resource_key=root_resource_key,
            output_dir=output_dir,
            visited=set(),
        )
    except Exception as exc:
        print(f"Failed to enumerate Google Drive folder: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    if not downloads:
        print("No files were found in the Google Drive folder.", file=sys.stderr)
        return 1

    downloaded_count = 0
    for drive_file, target_dir in downloads:
        target_dir.mkdir(parents=True, exist_ok=True)
        output_hint = str(target_dir) + "/"

        result = gdown.download(
            url=file_url(drive_file.file_id, drive_file.resource_key),
            output=output_hint,
            quiet=args.quiet,
            fuzzy=True,
            resume=args.resume,
        )
        if result is None:
            print(f"Failed to download file id={drive_file.file_id}", file=sys.stderr)
            return 1
        downloaded_count += 1

    print(f"Processed {downloaded_count} files into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
