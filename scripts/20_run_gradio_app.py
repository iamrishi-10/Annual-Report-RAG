"""
Launch the Gradio demo UI.

This is the standard entry point for running the app, so it starts the same
way as every other pipeline stage (python scripts/<n>_<name>.py) instead of
running the module directly. All UI wiring, .env loading, and launching
lives in src/app/gradio_app.py - this script only calls it.
"""

from src.app.gradio_app import main


if __name__ == "__main__":
    main()
