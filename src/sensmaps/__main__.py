"""Entry point. `python -m sensmaps` or the `sensmaps` console-script."""
from __future__ import annotations

import argparse
import sys
import tkinter as tk

from sensmaps.gui import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sensmaps")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build the main window, tick Tk once, destroy, and exit 0.",
    )
    args = parser.parse_args(argv)

    root = tk.Tk()
    root.title("sensmaps")
    root.geometry("1100x750")
    window = MainWindow(master=root)

    if args.smoke_test:
        root.update()
        root.destroy()
        return 0

    # Persist session on close
    def _on_close() -> None:
        window.save_session()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
