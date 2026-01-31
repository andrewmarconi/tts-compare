#!/usr/bin/env python3
"""TTS Compare — Textual TUI for comparing open-source TTS models."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TextArea,
)

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "output"
LOG_FILE = Path(__file__).parent / "log.txt"

_RICH_TAG_RE = re.compile(r"\[/?[a-z][a-z0-9_ ]*\]", re.IGNORECASE)


def get_next_run_folder(base_dir: Path) -> Path:
    """Get the next numbered run folder (e.g., 001, 002, 003).

    Looks in base_dir for existing numbered folders and returns the next
    number in sequence, zero-padded to 3 digits.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            existing_runs.append(int(item.name))

    next_num = max(existing_runs, default=0) + 1
    return base_dir / f"{next_num:03d}"

DEFAULT_TEXT = (
    "Wait—would you believe what just happened? The café served amazing "
    "crème brûlée and fresh croissants while a gentle breeze carried the "
    "scent of spring flowers across the garden, and birds sang their morning "
    "songs with pure joy!"
)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models() -> list[dict]:
    """Find all model directories with manifest.json and app.py."""
    # Detect current platform
    if sys.platform.startswith("linux"):
        current_platform = "linux"
    elif sys.platform == "darwin":
        current_platform = "macos"
    else:
        current_platform = "other"

    models = []
    if not MODELS_DIR.is_dir():
        return models
    for entry in sorted(MODELS_DIR.iterdir()):
        manifest_path = entry / "manifest.json"
        app_path = entry / "app.py"
        if entry.is_dir() and manifest_path.exists() and app_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Skip if platforms specified and current platform not supported
            if "platforms" in manifest and current_platform not in manifest["platforms"]:
                continue

            manifest["dir_name"] = entry.name
            models.append(manifest)
    return models


# ---------------------------------------------------------------------------
# Screen 1: Global parameters
# ---------------------------------------------------------------------------

class ParamsScreen(Screen):
    BINDINGS = [Binding("escape", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Text to Vocalize", classes="field-label")
            yield TextArea(DEFAULT_TEXT, id="text-input")
            yield Label("Voice Description (for models that support it)", classes="field-label")
            yield Input(placeholder="e.g. Warm male voice, 30s, american accent", id="description-input", value="warm male voice, mid-50s")
            yield Label("Reference Audio path (for models that support it)", classes="field-label")
            yield Input(value="audiosample.wav", placeholder="/path/to/reference.wav", id="reference-input")
            yield Label("Output directory", classes="field-label")
            yield Input(str(OUTPUT_DIR), id="output-dir-input")
            yield Button("Next →", variant="primary", id="btn-next")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-next":
            app = self.app
            assert isinstance(app, TTSCompareApp)
            app.global_params = {
                "text": self.query_one("#text-input", TextArea).text,
                "description": self.query_one("#description-input", Input).value,
                "reference_audio_path": self.query_one("#reference-input", Input).value,
                "output_dir": self.query_one("#output-dir-input", Input).value,
            }
            app.push_screen(ModelSelectScreen())


# ---------------------------------------------------------------------------
# Screen 2: Model selection + per-model voice config
# ---------------------------------------------------------------------------

class ModelSelectScreen(Screen):
    BINDINGS = [Binding("escape", "pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        app = self.app
        assert isinstance(app, TTSCompareApp)
        models = app.models

        yield Header()
        with Horizontal(id="model-layout"):
            with Vertical(id="model-list"):
                yield Label("Select Models", classes="section-label")
                for m in models:
                    yield Checkbox(m["name"], id=f"chk-{m['dir_name']}", value=True)
                yield Button("← Back", id="btn-back")
                yield Button("Generate ▶", variant="primary", id="btn-generate")

            with VerticalScroll(id="model-config"):
                yield Label("Per-Model Voice Configuration", classes="section-label")
                for m in models:
                    presets = m.get("voice_presets", [])
                    if presets:
                        yield Label(f"{m['name']} — Voice:", classes="field-label")
                        options = [(p, p) for p in presets]
                        yield Select(options, value=presets[0], id=f"voice-{m['dir_name']}")

                    tags = m.get("emotion_tags", [])
                    if tags:
                        yield Label(
                            f"{m['name']} — Emotion tags: {', '.join(tags)}",
                            classes="info-label",
                        )

                    caps = []
                    if m.get("supports_description"):
                        caps.append("voice description")
                    if m.get("supports_reference_audio"):
                        caps.append("reference audio")
                    if presets:
                        caps.append("voice presets")
                    if caps:
                        yield Label(
                            f"{m['name']} uses: {', '.join(caps)}",
                            classes="caps-label",
                        )
                    yield Static("─" * 40, classes="separator")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-generate":
            app = self.app
            assert isinstance(app, TTSCompareApp)

            # Collect selected models and per-model voice presets
            selected = []
            voice_presets: dict[str, str] = {}
            for m in app.models:
                chk = self.query_one(f"#chk-{m['dir_name']}", Checkbox)
                if chk.value:
                    selected.append(m)
                    try:
                        sel = self.query_one(f"#voice-{m['dir_name']}", Select)
                        voice_presets[m["dir_name"]] = str(sel.value)
                    except Exception:
                        voice_presets[m["dir_name"]] = ""

            app.selected_models = selected
            app.voice_presets = voice_presets
            app.push_screen(ExecutionScreen())


# ---------------------------------------------------------------------------
# Screen 3: Execution log
# ---------------------------------------------------------------------------

class ExecutionScreen(Screen):
    BINDINGS = [Binding("escape", "pop_screen", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="exec-log", highlight=True, markup=True)
        yield Button("← Done", id="btn-done")
        yield Footer()

    def on_mount(self) -> None:
        self.run_models()

    @work(thread=True)
    def run_models(self) -> None:
        app = self.app
        assert isinstance(app, TTSCompareApp)
        log = self.query_one("#exec-log", RichLog)
        params = app.global_params

        # Create numbered run folder (e.g., output/001, output/002, etc.)
        base_output_dir = Path(params.get("output_dir", str(OUTPUT_DIR)))
        run_folder = get_next_run_folder(base_output_dir)
        run_folder.mkdir(parents=True, exist_ok=True)
        output_dir = str(run_folder)

        # Save log file in the run folder
        log_file_path = run_folder / "log.txt"
        logfile = open(log_file_path, "w")

        def emit(msg: str) -> None:
            """Write to both the TUI RichLog and the plain-text log file."""
            log.write(msg)
            plain = _RICH_TAG_RE.sub("", msg)
            logfile.write(plain + "\n")
            logfile.flush()

        emit(f"[bold]Output folder: {output_dir}[/bold]\n")

        results: list[tuple[str, bool, str]] = []
        benchmarks: list[dict] = []

        for m in app.selected_models:
            dir_name = m["dir_name"]
            display_name = m["name"]
            model_dir = MODELS_DIR / dir_name
            output_path = str(Path(output_dir).resolve() / f"{dir_name}.wav")

            emit(f"\n{'=' * 50}")
            emit(f"[bold cyan]  Running: {display_name}[/bold cyan]")
            emit(f"{'=' * 50}")

            # Resolve reference audio to absolute path (subprocesses run in model dirs)
            ref_path = params.get("reference_audio_path", "")
            if ref_path and not os.path.isabs(ref_path):
                ref_path = str(Path(ref_path).resolve())

            run_params = {
                "text": params["text"],
                "description": params.get("description", ""),
                "voice_preset": app.voice_presets.get(dir_name, ""),
                "reference_audio_path": ref_path,
                "output_path": output_path,
            }

            try:
                result = asyncio.run(
                    self._run_subprocess(model_dir, run_params, emit)
                )

                timing = result.get("timing", {})

                if result["returncode"] == 0:
                    out = result["stdout"].strip().splitlines()
                    out_path = out[-1] if out else ""
                    if out_path and os.path.isfile(out_path):
                        emit(f"  [bold green]✓ Output: {out_path}[/bold green]")
                        results.append((display_name, True, out_path))
                    else:
                        emit(f"  [bold green]✓ Completed[/bold green]")
                        results.append((display_name, True, ""))

                    if timing:
                        emit(
                            f"  [cyan]Load: {timing['model_load_ms']:.0f}ms | "
                            f"Inference: {timing['model_inference_ms']:.0f}ms[/cyan]"
                        )
                    benchmarks.append({
                        "model": display_name,
                        "model_load_ms": timing.get("model_load_ms"),
                        "model_inference_ms": timing.get("model_inference_ms"),
                        "settings": {
                            "text": run_params["text"],
                            "description": run_params["description"],
                            "voice_preset": run_params["voice_preset"],
                            "reference_audio_path": run_params["reference_audio_path"],
                        },
                    })
                else:
                    emit(f"  [bold red]✗ Failed (exit {result['returncode']})[/bold red]")
                    results.append((display_name, False, ""))

            except Exception as exc:
                safe = str(exc).replace("[", "\\[")
                emit(f"  [bold red]✗ Error: {safe}[/bold red]")
                results.append((display_name, False, ""))

            # Release GPU memory between models
            emit("  [dim]Clearing GPU cache...[/dim]")
            try:
                asyncio.run(self._run_gpu_cleanup(model_dir))
                time.sleep(2)
            except Exception:
                pass

        # Write benchmark results JSON
        results_path = Path(output_dir).resolve() / "results.json"
        with open(results_path, "w") as f:
            json.dump(benchmarks, f, indent=2)
        emit(f"\n[bold]Benchmark results written to {results_path}[/bold]")

        # Summary
        emit(f"\n{'=' * 50}")
        emit("[bold]  Summary[/bold]")
        emit(f"{'=' * 50}")
        for name, success, path in results:
            if success:
                emit(f"  [green]✓ {name}[/green]  {path}")
            else:
                emit(f"  [red]✗ {name}[/red]")

        logfile.close()

    async def _run_subprocess(
        self, model_dir: Path, params: dict, emit: callable
    ) -> dict:
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "python", "-u", "app.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(model_dir),
            env=env,
            limit=1024 * 1024,  # 1 MiB buffer limit (default is 64 KiB)
        )
        # Send input and close stdin
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(params).encode())
        proc.stdin.close()

        # Stream stderr lines in real-time, capture TIMING lines
        timing: dict = {}
        assert proc.stderr is not None
        async for raw_line in proc.stderr:
            line = raw_line.decode().rstrip()
            if not line:
                continue
            if line.startswith("TIMING:"):
                timing = json.loads(line[7:])
            else:
                safe_line = line.replace("[", "\\[")
                emit(f"  [dim]{safe_line}[/dim]")

        # Read stdout (contains the output path)
        assert proc.stdout is not None
        stdout = await proc.stdout.read()
        await proc.wait()

        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode(),
            "timing": timing,
        }

    async def _run_gpu_cleanup(self, model_dir: Path) -> None:
        """Run a small subprocess to free GPU memory."""
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        proc = await asyncio.create_subprocess_exec(
            "uv", "run", "python", "-c",
            "import gc; gc.collect(); import torch;"
            " torch.cuda.empty_cache() if torch.cuda.is_available() else None;"
            " torch.mps.empty_cache() if hasattr(torch, 'mps') and torch.backends.mps.is_available() else None",
            cwd=str(model_dir),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-done":
            self.app.pop_screen()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class TTSCompareApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #text-input {
        height: 6;
        margin: 0 1;
    }
    .field-label {
        margin: 1 1 0 1;
        color: $accent;
    }
    .section-label {
        margin: 1;
        text-style: bold;
        color: $accent;
    }
    .info-label {
        margin: 0 1;
        color: $text-muted;
    }
    .caps-label {
        margin: 0 1;
        color: $success;
    }
    .separator {
        margin: 0 1;
        color: $surface;
    }
    Input {
        margin: 0 1;
    }
    Select {
        margin: 0 1;
    }
    Button {
        margin: 1 1;
    }
    #model-layout {
        height: 1fr;
    }
    #model-list {
        width: 35;
        border-right: solid $surface;
        padding: 1;
    }
    #model-config {
        width: 1fr;
        padding: 1;
    }
    #exec-log {
        height: 1fr;
        margin: 1;
        border: solid $surface;
    }
    """

    TITLE = "TTS Compare"
    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.models = discover_models()
        self.global_params: dict = {}
        self.selected_models: list[dict] = []
        self.voice_presets: dict[str, str] = {}

    def on_mount(self) -> None:
        self.push_screen(ParamsScreen())


if __name__ == "__main__":
    TTSCompareApp().run()
