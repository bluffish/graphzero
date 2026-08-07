from __future__ import annotations

import argparse
import sys

from gz.checkpoints import DirectorySource
from gz.common.log import setup
from gz.evaluator.backends import StubBackend, TorchBackend
from gz.evaluator.server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--backend", choices=["stub", "torch"], default="stub")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-pointer")
    parser.add_argument("--device")
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--max-resident-model-generations", type=int, default=2)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    args = parser.parse_args(argv)
    log = setup("gz.evaluator")
    try:
        if args.backend == "torch":
            if args.checkpoint_dir is None:
                parser.error("--checkpoint-dir is required for --backend torch")
            backend = TorchBackend(
                DirectorySource(
                    args.checkpoint_dir,
                    pointer=args.checkpoint_pointer or "latest.json",
                ),
                device=args.device,
                compile_model=not args.no_compile,
                max_batch=args.max_batch,
                max_resident_model_generations=args.max_resident_model_generations,
                poll_interval=args.poll_interval,
            )
        else:
            backend = StubBackend()
        log.info("event=start socket=%s backend=%s", args.socket, args.backend)
        serve(args.socket, backend)
    except Exception as error:
        log.error("event=error error=%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
