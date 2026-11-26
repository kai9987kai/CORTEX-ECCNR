"""Download, verify, and prepare the FlyWire FAFB v783 connectome cache."""
import argparse
import json
from pathlib import Path
from eccnr.connectome import prepare_flywire_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and verify FlyWire FAFB v783 connectome")
    parser.add_argument("--data-dir", type=Path, default=Path("data/flywire"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = prepare_flywire_cache(args.data_dir, force=args.force)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
