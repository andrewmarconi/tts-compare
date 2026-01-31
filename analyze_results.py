#!/usr/bin/env python3
"""Analyze TTS benchmark results and generate an interactive Plotly HTML dashboard."""

import glob
import json
import os
import re
import webbrowser

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(output_dir: str = "output") -> list[dict]:
    """Load all results.json files, adding run ID to each record."""
    records = []
    paths = sorted(glob.glob(os.path.join(output_dir, "*/results.json")))
    for path in paths:
        run_id = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            entry["run"] = run_id
            entry["total_ms"] = entry["model_load_ms"] + entry["model_inference_ms"]
            records.append(entry)
    return records


def build_dashboard(records: list[dict], output_path: str = "output/analysis.html"):
    if not records:
        print("No results found.")
        return

    runs = sorted(set(r["run"] for r in records))
    models = sorted(set(r["model"] for r in records))

    # Color palette
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    ]
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    fig = make_subplots(
        rows=5, cols=1,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
        subplot_titles=[
            "Model Load Time (High-Low across runs)",
            "Inference Time (High-Low across runs)",
            "Cross-Run Trend: Total Time",
            "Load vs Inference Breakdown (Stacked)",
            "",  # table has no subplot title
        ],
        specs=[[{"type": "box"}], [{"type": "box"}], [{"type": "scatter"}], [{"type": "bar"}], [{"type": "table"}]],
        vertical_spacing=0.06,
    )

    # Consistent model ordering: sort by average total time
    models_by_total = sorted(models, key=lambda m: sum(
        r["total_ms"] for r in records if r["model"] == m
    ) / max(1, sum(1 for r in records if r["model"] == m)))

    # ── 1 & 2. Box plots for load and inference ──
    for row, metric, title_color in [
        (1, "model_load_ms", "#636EFA"),
        (2, "model_inference_ms", "#EF553B"),
    ]:
        for model in models_by_total:
            vals = [r[metric] / 1000 for r in records if r["model"] == model]
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=model,
                    marker_color=title_color,
                    boxmean=True,
                    boxpoints="all",
                    pointpos=0,
                    jitter=0.3,
                    showlegend=False,
                ),
                row=row, col=1,
            )

    # ── 2. Cross-run trend lines ──
    for model in models:
        model_data = sorted(
            [r for r in records if r["model"] == model],
            key=lambda r: r["run"],
        )
        fig.add_trace(
            go.Scatter(
                name=model,
                x=[r["run"] for r in model_data],
                y=[r["total_ms"] / 1000 for r in model_data],
                mode="lines+markers",
                marker_color=model_colors[model],
                legendgroup="trends", legendgrouptitle_text="Models",
            ),
            row=3, col=1,
        )

    # ── 4. Stacked bar: load vs inference proportion ──
    for model in models:
        model_data = sorted(
            [r for r in records if r["model"] == model],
            key=lambda r: r["run"],
        )
        x_labels = [f"{model}<br>Run {r['run']}" for r in model_data]
        fig.add_trace(
            go.Bar(
                name=f"{model} Load",
                x=x_labels,
                y=[r["model_load_ms"] / 1000 for r in model_data],
                marker_color=model_colors[model],
                opacity=0.6,
                legendgroup="stacked", showlegend=False,
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Bar(
                name=f"{model} Inference",
                x=x_labels,
                y=[r["model_inference_ms"] / 1000 for r in model_data],
                marker_color=model_colors[model],
                opacity=1.0,
                legendgroup="stacked", showlegend=False,
            ),
            row=4, col=1,
        )

    fig.update_layout(barmode="stack", bargap=0.3)

    # ── 5. Summary table ──
    header_vals = ["Model", "Runs", "Avg Load (s)", "Avg Infer (s)", "Avg Total (s)",
                   "Min Total (s)", "Max Total (s)"]
    table_rows: dict[str, list] = {h: [] for h in header_vals}

    for model in models:
        model_data = [r for r in records if r["model"] == model]
        loads = [r["model_load_ms"] / 1000 for r in model_data]
        infers = [r["model_inference_ms"] / 1000 for r in model_data]
        totals = [r["total_ms"] / 1000 for r in model_data]

        table_rows["Model"].append(model)
        table_rows["Runs"].append(len(model_data))
        table_rows["Avg Load (s)"].append(f"{sum(loads)/len(loads):.1f}")
        table_rows["Avg Infer (s)"].append(f"{sum(infers)/len(infers):.1f}")
        table_rows["Avg Total (s)"].append(f"{sum(totals)/len(totals):.1f}")
        table_rows["Min Total (s)"].append(f"{min(totals):.1f}")
        table_rows["Max Total (s)"].append(f"{max(totals):.1f}")

    fig.add_trace(
        go.Table(
            header=dict(values=header_vals, fill_color="#2a2a2e", font=dict(color="white", size=13), align="center"),
            cells=dict(
                values=[table_rows[h] for h in header_vals],
                fill_color="#1e1e22",
                font=dict(color="#e0e0e0", size=12),
                align="center",
            ),
        ),
        row=5, col=1,
    )

    # ── Layout ──
    fig.update_layout(
        title="TTS Model Benchmark Results",
        template="plotly_dark",
        height=1900,
        showlegend=False,
    )
    fig.update_yaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Total Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Time (s)", row=4, col=1)
    fig.update_xaxes(title_text="Run", row=3, col=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"Dashboard written to {output_path}")
    return output_path


if __name__ == "__main__":
    records = load_results()
    path = build_dashboard(records)
    if path:
        abs_path = os.path.abspath(path)
        try:
            import subprocess
            subprocess.Popen(["xdg-open", abs_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            webbrowser.open(f"file://{abs_path}")
