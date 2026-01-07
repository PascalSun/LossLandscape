"""
Module entry point for ``python -m loss_landscape``.

This simply forwards to the Click CLI defined in ``loss_landscape.cli``.
"""

from .cli import cli


def main() -> None:
    cli()


if __name__ == "__main__":
    main()


