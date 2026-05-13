import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repo.")
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face repo id, e.g. Qwen/Qwen2-VL-7B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Target directory. If omitted, uses /home/eboccaletti/models/<model_name>",
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        default="/home/eboccaletti/.cache/huggingface",
        help="Hugging Face cache directory",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch, tag, or commit hash",
    )
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    Path(args.hf_home).mkdir(parents=True, exist_ok=True)

    if args.output_dir is None:
        model_name = args.repo_id.split("/")[-1]
        output_dir = Path("/home/eboccaletti/models") / model_name
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo ID:    {args.repo_id}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"HF_HOME:    {args.hf_home}", flush=True)
    if args.revision is not None:
        print(f"Revision:   {args.revision}", flush=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
    )

    print("\nDownload complete.", flush=True)
    print(f"Saved to: {output_dir}", flush=True)

    try:
        files = sorted(output_dir.iterdir())
        print("\nTop-level contents:", flush=True)
        for path in files[:20]:
            print(f"  {path.name}", flush=True)
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more", flush=True)
    except Exception as e:
        print(f"Could not list files: {e}", flush=True)


if __name__ == "__main__":
    main()