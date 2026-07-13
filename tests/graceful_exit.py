"""One-off test runner: starts pipeline.main_loop.run() and raises SIGINT in
the calling (main) thread after a few seconds so we can verify the graceful
KeyboardInterrupt -> finally -> print + vram_file.close() path.
"""
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fire_sigint():
    time.sleep(15.0)
    print("\n[test_runner] raising SIGINT on main thread", flush=True)
    signal.raise_signal(signal.SIGINT)


def main():
    threading.Thread(target=_fire_sigint, daemon=True).start()
    print("[test_runner] starting main_loop; SIGINT will fire in 15s", flush=True)
    import pipeline.main_loop as m
    m.run()


if __name__ == "__main__":
    main()