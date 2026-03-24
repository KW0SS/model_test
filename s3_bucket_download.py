from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_BUCKET_NAME = "kw0ss-raw-data-s3"
REGION_NAME = "ap-northeast-2"
WORKDIR = Path(r"C:\kwoss_C\model_test")
ENV_PATH = WORKDIR / ".env"


def load_dotenv_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all objects from an S3 bucket.")
    parser.add_argument("--access-key", help="AWS access key ID")
    parser.add_argument("--secret-key", help="AWS secret access key")
    parser.add_argument("--bucket", default=None, help="S3 bucket name")
    return parser.parse_args()


def prompt_credentials(args: argparse.Namespace) -> tuple[str, str]:
    dotenv_values = load_dotenv_file(ENV_PATH)
    access_key = (
        args.access_key
        or os.environ.get("AWS_ACCESS_KEY_ID", "")
        or dotenv_values.get("AWS_ACCESS_KEY_ID", "")
        or dotenv_values.get("S3_ACCESS_KEY", "")
    ).strip()
    secret_key = (
        args.secret_key
        or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        or dotenv_values.get("AWS_SECRET_ACCESS_KEY", "")
        or dotenv_values.get("S3_PRIVATE_KEY", "")
    ).strip()

    if access_key and secret_key:
        return access_key, secret_key

    if not os.isatty(0):
        raise ValueError(
            "AWS credentials are required. In non-interactive mode, set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY or pass --access-key and --secret-key."
        )

    access_key = input("AWS Access Key ID: ").strip()
    secret_key = getpass.getpass("AWS Secret Access Key: ").strip()

    if not access_key or not secret_key:
        raise ValueError("AWS credentials are required.")

    return access_key, secret_key


def build_s3_client(access_key: str, secret_key: str):
    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=REGION_NAME,
    )
    return session.client("s3")


def should_skip_download(local_path: Path, remote_size: int) -> bool:
    return local_path.exists() and local_path.is_file() and local_path.stat().st_size == remote_size


def ensure_parent_directory(local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)


def iter_bucket_objects(s3_client, bucket_name: str):
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
            yield obj


def get_bucket_name(args: argparse.Namespace | None = None) -> str:
    dotenv_values = load_dotenv_file(ENV_PATH)
    bucket_name = (
        (args.bucket if args else None)
        or os.environ.get("S3_BUCKET_NAME", "")
        or dotenv_values.get("S3_BUCKET_NAME", "")
        or DEFAULT_BUCKET_NAME
    ).strip()
    return bucket_name


def download_bucket() -> int:
    args = parse_args()
    access_key, secret_key = prompt_credentials(args)
    bucket_name = get_bucket_name(args)
    download_root = WORKDIR / "downloads" / bucket_name
    s3_client = build_s3_client(access_key, secret_key)

    download_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    total = 0

    try:
        for obj in iter_bucket_objects(s3_client, bucket_name):
            key = obj["Key"]
            size = obj["Size"]
            total += 1

            if key.endswith("/"):
                (download_root / key).mkdir(parents=True, exist_ok=True)
                continue

            local_path = download_root / Path(*Path(key).parts)
            ensure_parent_directory(local_path)

            if should_skip_download(local_path, size):
                skipped += 1
                print(f"[SKIP] {key}")
                continue

            try:
                s3_client.download_file(bucket_name, key, str(local_path))
                downloaded += 1
                print(f"[DOWNLOADED] {key}")
            except (BotoCoreError, ClientError, OSError) as exc:
                failed.append((key, str(exc)))
                print(f"[FAILED] {key}: {exc}")
    except (BotoCoreError, ClientError) as exc:
        print(f"Bucket listing failed: {exc}")
        return 1

    print("\nDownload summary")
    print(f"Bucket: {bucket_name}")
    print(f"Region: {REGION_NAME}")
    print(f"Destination: {download_root}")
    print(f"Total objects seen: {total}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed objects")
        for key, error_message in failed:
            print(f"- {key}: {error_message}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(download_bucket())
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.")
        raise SystemExit(130)
    except ValueError as exc:
        print(exc)
        raise SystemExit(1)
