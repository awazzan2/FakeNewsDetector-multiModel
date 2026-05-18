import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import tweepy
from dotenv import load_dotenv
from tqdm import tqdm


ID_PATTERN = re.compile(r"\b\d{6,20}\b")
CHECKPOINT_EVERY_TWEETS = 100
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 2


def load_api_client() -> tweepy.Client:
    # Always load .env from this script's directory and override any empty
    # variable that may already exist in the shell session.
    dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=True)
    bearer = os.getenv("TWITTER_BEARER_TOKEN", "").strip()
    if not bearer:
        # Fallback for .env files saved with UTF-8 BOM.
        bearer = os.getenv("\ufeffTWITTER_BEARER_TOKEN", "").strip()
    if not bearer:
        raise RuntimeError("TWITTER_BEARER_TOKEN is missing in .env")
    return tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)


def extract_ids_from_value(value, bucket: set[str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).lower()
            # Prioritize fields that usually represent tweet IDs.
            if key in {"tweet_id", "tweetid", "status_id", "id_str"}:
                if isinstance(v, (str, int)):
                    s = str(v).strip()
                    if ID_PATTERN.fullmatch(s):
                        bucket.add(s)
                continue
            extract_ids_from_value(v, bucket)
    elif isinstance(value, list):
        for item in value:
            extract_ids_from_value(item, bucket)
    elif isinstance(value, (str, int)):
        # Fallback for textual files that include IDs among other text.
        text = str(value)
        for token in ID_PATTERN.findall(text):
            bucket.add(token)


def collect_tweet_ids(dataset_root: Path) -> list[str]:
    files = [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".txt", ".csv"}]
    tweet_ids: set[str] = set()

    for file_path in tqdm(files, desc="Scanning FakeNewsNet files"):
        try:
            if file_path.suffix.lower() == ".json":
                data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
                extract_ids_from_value(data, tweet_ids)
            else:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                extract_ids_from_value(text, tweet_ids)
        except Exception:
            # Skip unreadable/bad files and continue.
            continue

    return sorted(tweet_ids)


def chunked(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def append_rows(csv_path: Path, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def load_existing_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["tweet_id"], dtype={"tweet_id": str})
        return set(df["tweet_id"].dropna().astype(str))
    except Exception:
        return set()


def write_report(
    out_root: Path,
    total_ids: int,
    processed_ids: int,
    hydrated_total: int,
    missing_total: int,
) -> None:
    report = {
        "total_ids_found": total_ids,
        "processed_ids": processed_ids,
        "remaining_ids": max(total_ids - processed_ids, 0),
        "hydrated_tweets": hydrated_total,
        "missing_or_unavailable": missing_total,
        "coverage_percent": round((hydrated_total / total_ids) * 100, 2) if total_ids else 0.0,
        "progress_percent": round((processed_ids / total_ids) * 100, 2) if total_ids else 0.0,
    }
    (out_root / "hydration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def fetch_with_retry(client: tweepy.Client, batch: list[str]):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.get_tweets(
                ids=batch,
                tweet_fields=["created_at", "author_id", "public_metrics", "lang", "conversation_id", "possibly_sensitive"],
            )
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"retries_exhausted:{type(exc).__name__}") from exc
            sleep_s = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"Request failed ({type(exc).__name__}), retrying in {sleep_s}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(sleep_s)


def hydrate_tweets(
    client: tweepy.Client,
    pending_ids: list[str],
    out_root: Path,
    clean_dir: Path,
    total_ids: int,
    already_hydrated: int,
    already_missing: int,
) -> tuple[int, int]:
    tweets_csv = clean_dir / "tweets.csv"
    missing_csv = clean_dir / "missing_tweets.csv"
    tweets_cols = [
        "tweet_id", "author_id", "created_at", "text", "lang", "conversation_id",
        "possibly_sensitive", "retweet_count", "reply_count", "like_count", "quote_count",
    ]
    missing_cols = ["tweet_id", "reason"]

    new_hydrated = 0
    new_missing = 0
    processed_in_run = 0
    since_last_checkpoint = 0
    tweets_buffer: list[dict] = []
    missing_buffer: list[dict] = []
    batch_size = 100

    for batch in tqdm(chunked(pending_ids, batch_size), total=(len(pending_ids) + batch_size - 1) // batch_size, desc="Hydrating tweets"):
        try:
            response = fetch_with_retry(client, batch)
            found = set()

            if response and response.data:
                for tw in response.data:
                    found.add(str(tw.id))
                    metrics = tw.public_metrics or {}
                    tweets_buffer.append(
                        {
                            "tweet_id": str(tw.id),
                            "author_id": str(tw.author_id) if tw.author_id else "",
                            "created_at": str(tw.created_at) if tw.created_at else "",
                            "text": (tw.text or "").replace("\n", " ").strip(),
                            "lang": tw.lang or "",
                            "conversation_id": str(tw.conversation_id) if tw.conversation_id else "",
                            "possibly_sensitive": bool(tw.possibly_sensitive) if tw.possibly_sensitive is not None else "",
                            "retweet_count": metrics.get("retweet_count", ""),
                            "reply_count": metrics.get("reply_count", ""),
                            "like_count": metrics.get("like_count", ""),
                            "quote_count": metrics.get("quote_count", ""),
                        }
                    )
                    new_hydrated += 1

            for tweet_id in batch:
                if tweet_id not in found:
                    missing_buffer.append({"tweet_id": tweet_id, "reason": "not_returned_deleted_private_or_unavailable"})
                    new_missing += 1
        except Exception as exc:
            # Only mark missing after all retries are exhausted.
            err = str(exc)
            for tweet_id in batch:
                missing_buffer.append({"tweet_id": tweet_id, "reason": err})
                new_missing += 1

        processed_in_run += len(batch)
        since_last_checkpoint += len(batch)

        if since_last_checkpoint >= CHECKPOINT_EVERY_TWEETS:
            append_rows(tweets_csv, tweets_buffer, tweets_cols)
            append_rows(missing_csv, missing_buffer, missing_cols)
            tweets_buffer.clear()
            missing_buffer.clear()
            since_last_checkpoint = 0
            write_report(
                out_root=out_root,
                total_ids=total_ids,
                processed_ids=already_hydrated + already_missing + processed_in_run,
                hydrated_total=already_hydrated + new_hydrated,
                missing_total=already_missing + new_missing,
            )

    append_rows(tweets_csv, tweets_buffer, tweets_cols)
    append_rows(missing_csv, missing_buffer, missing_cols)
    write_report(
        out_root=out_root,
        total_ids=total_ids,
        processed_ids=already_hydrated + already_missing + processed_in_run,
        hydrated_total=already_hydrated + new_hydrated,
        missing_total=already_missing + new_missing,
    )
    return new_hydrated, new_missing


def main() -> int:
    project_root = Path.cwd()
    dataset_root = project_root / "FakeNewsNet"
    out_root = project_root / "output"
    clean_dir = out_root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        print("Error: 'FakeNewsNet' folder not found in current directory.")
        print("Tip: run this script from your project folder that contains the FakeNewsNet clone.")
        return 1

    try:
        client = load_api_client()
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    tweet_ids = collect_tweet_ids(dataset_root)
    if not tweet_ids:
        print("No candidate tweet IDs found in FakeNewsNet files.")
        return 1

    all_ids_csv = clean_dir / "all_tweet_ids.csv"
    pd.DataFrame({"tweet_id": tweet_ids}).to_csv(all_ids_csv, index=False)

    tweets_csv = clean_dir / "tweets.csv"
    missing_csv = clean_dir / "missing_tweets.csv"
    done_hydrated_ids = load_existing_ids(tweets_csv)
    done_missing_ids = load_existing_ids(missing_csv)
    done_ids = done_hydrated_ids | done_missing_ids

    pending_ids = [tid for tid in tweet_ids if tid not in done_ids]
    print(f"Found {len(tweet_ids)} total IDs")
    print(f"Already processed: {len(done_ids)}")
    print(f"Remaining to process: {len(pending_ids)}")

    if not pending_ids:
        write_report(
            out_root=out_root,
            total_ids=len(tweet_ids),
            processed_ids=len(done_ids),
            hydrated_total=len(done_hydrated_ids),
            missing_total=len(done_missing_ids),
        )
        print("Nothing to do. Resume check found all IDs already processed.")
        return 0

    new_hydrated, new_missing = hydrate_tweets(
        client=client,
        pending_ids=pending_ids,
        out_root=out_root,
        clean_dir=clean_dir,
        total_ids=len(tweet_ids),
        already_hydrated=len(done_hydrated_ids),
        already_missing=len(done_missing_ids),
    )

    final_hydrated = len(done_hydrated_ids) + new_hydrated
    final_missing = len(done_missing_ids) + new_missing
    final_report = {
        "total_ids_found": len(tweet_ids),
        "processed_ids": final_hydrated + final_missing,
        "remaining_ids": len(tweet_ids) - (final_hydrated + final_missing),
        "hydrated_tweets": final_hydrated,
        "missing_or_unavailable": final_missing,
        "coverage_percent": round((final_hydrated / len(tweet_ids)) * 100, 2) if tweet_ids else 0.0,
        "progress_percent": round(((final_hydrated + final_missing) / len(tweet_ids)) * 100, 2) if tweet_ids else 0.0,
    }
    print(json.dumps(final_report, indent=2))
    print("\nSaved files:")
    print(f"- {all_ids_csv}")
    print(f"- {tweets_csv}")
    print(f"- {missing_csv}")
    print(f"- {out_root / 'hydration_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
