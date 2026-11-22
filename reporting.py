"""Summarize alerts by the week when the monitor could see them."""

import json

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


MONITORS = {
    "confidence_ks": ("Confidence KS", "batch"),
    "positive_rate": ("Predicted-positive rate", "batch"),
    "embedding_ks": ("Embedding KS", "batch"),
    "accuracy": ("Review accuracy", "review"),
    "auroc": ("Review AUROC", "review"),
    "kappa": ("Review Cohen's kappa", "review"),
}


def alert_history(weeks, thresholds, drift_week):
    monitors = {}
    for key, (_, source) in MONITORS.items():
        bounds = thresholds[key]
        alerts = []
        for entry in weeks:
            observation = entry[source]
            if observation is None:
                continue
            value = observation["metrics"][key]
            if value is None:
                continue
            if value < bounds.get("low", -float("inf")) or value > bounds.get("high", float("inf")):
                alerts.append({"week": entry["week"],
                               "source_week": observation.get("source_week", entry["week"]),
                               "value": value})
        drift_alert = next((alert for alert in alerts
                            if alert["source_week"] >= drift_week), None)
        monitors[key] = {
            "alerts": alerts,
            "first_drift_alert": drift_alert,
            "detection_delay_weeks": drift_alert["week"] - drift_week if drift_alert else None,
        }
    return monitors


def format_value(value):
    return "NA" if value is None else f"{value:.3f}"


def format_bounds(bounds):
    return " or ".join(
        f"{'<' if direction == 'low' else '>'} {limit:.3f}"
        for direction, limit in bounds.items())


def format_summary(report):
    config = report["config"]
    reference = report["reference_metrics"]
    lines = [
        f"Reference (n={report['reference_cases']}) | "
        f"accuracy {format_value(reference['accuracy'])} | "
        f"AUROC {format_value(reference['auroc'])} | "
        f"kappa {format_value(reference['kappa'])} | "
        f"predicted-positive {format_value(reference['positive_rate'])}",
        f"Drift starts week {config['drift_week']} | "
        f"{config['batch_size']} cases/week | "
        f"{config['review_size']} reviews after {config['label_delay_weeks']} week",
    ]
    for key, (name, _) in MONITORS.items():
        result = report["monitors"][key]
        first = result["first_drift_alert"]
        early = [str(alert["week"]) for alert in result["alerts"]
                 if alert["source_week"] < config["drift_week"]]
        alert_text = (f"week {first['week']} (cases from week {first['source_week']}, "
                      f"value {format_value(first['value'])}, "
                      f"delay {result['detection_delay_weeks']}w)" if first
                      else "no alert on shifted cases")
        lines.append(f"{name} | threshold {format_bounds(report['thresholds'][key])} | "
                     f"{alert_text} | alerts on pre-drift cases: {', '.join(early) or 'none'}")
    return "\n".join(lines)


def markdown_report(report):
    lines = [
        "# Imaging Drift Watch run",
        "",
        f"Dataset: {report['dataset']} | seed: {report['seed']} | "
        f"epochs: {report['epochs']} | training cases: {report['training_cases']}",
        "",
        "```text",
        format_summary(report),
        "```",
        "",
        "Week is the time of observation. Review metrics refer to the listed case week. "
        "NA means a metric was unavailable. Strict threshold crossings trigger alerts. "
        "Alert decisions use full precision.",
        "",
        "| Week | Confidence KS | Predicted-positive rate | Embedding KS | Review case week "
        "| Accuracy | AUROC | Kappa | Alerts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if report["fallback_reason"]:
        lines[2] += f" | fallback: {report['fallback_reason']}"
    for entry in report["weeks"]:
        batch = (entry["batch"] or {}).get("metrics", {})
        review = entry["review"] or {}
        columns = [str(entry["week"])]
        columns += [format_value(batch.get(key)) for key in list(MONITORS)[:3]]
        columns += [str(review.get("source_week", "NA"))]
        columns += [format_value(review.get("metrics", {}).get(key))
                    for key in list(MONITORS)[3:]]
        fired = [name for key, (name, _) in MONITORS.items()
                 if any(alert["week"] == entry["week"] for alert in report["monitors"][key]["alerts"])]
        columns += [", ".join(fired) or "None"]
        lines.append("| " + " | ".join(columns) + " |")
    lines.extend(("", "![Monitoring timeline](timeline.png)", ""))
    return "\n".join(lines)


def plot_timeline(report, path):
    weeks = report["weeks"]
    drift = report["config"]["drift_week"]
    figure = Figure(figsize=(11, 8.2), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2, sharex=True)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.10,
                           hspace=0.48, wspace=0.23)
    figure.suptitle("Imaging Drift Watch", x=0.08, y=0.968, ha="left",
                    fontsize=23, fontweight="bold", color="#142536")
    figure.text(0.08, 0.916,
                f"{report['dataset']} | simulated scanner and prevalence shift at week {drift} "
                "| reviews arrive one week after inference",
                color="#40576d", fontsize=10)
    for ax, (key, (name, source)) in zip(axes.T.flat, MONITORS.items()):
        points = [(entry["week"], entry[source]["metrics"][key])
                  for entry in weeks if entry[source] is not None]
        x, y = zip(*points)
        color = "#087f8c" if source == "batch" else "#3451a3"
        ax.axvspan(drift, weeks[-1]["week"] + 0.2, color="#eaf0f5")
        ax.axvline(drift, color="#546c80", ls=":", lw=1.4, zorder=2)
        for limit in report["thresholds"][key].values():
            ax.axhline(limit, color="#92734f", ls="--", lw=1, zorder=2)
        ax.plot(x, [float("nan") if value is None else value for value in y],
                marker="o", markersize=4, color=color, linewidth=1.6)
        result = report["monitors"][key]
        for alert in result["alerts"]:
            ax.scatter(alert["week"], alert["value"], facecolors="none",
                       edgecolors="#b72c43", s=75, lw=1.5, zorder=4)
        first = result["first_drift_alert"]
        if first:
            ax.scatter(first["week"], first["value"], color="#b72c43", s=19, zorder=5)
        delay = result["detection_delay_weeks"]
        delay_text = f"{delay}w delay" if delay is not None else "No alert on shifted cases"
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.set_title(delay_text, loc="right", fontsize=9, color="#40576d")
        ax.set_xticks([entry["week"] for entry in weeks])
        ax.set_xlim(0.8, weeks[-1]["week"] + 0.2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#dce5eb", lw=0.7, zorder=0)
        ax.tick_params(labelsize=8, length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    figure.text(0.28, 0.874, "AT INFERENCE", ha="center", color="#087f8c", fontweight="bold")
    figure.text(0.77, 0.874, "AFTER LABEL ARRIVAL", ha="center", color="#3451a3", fontweight="bold")
    figure.supxlabel("Observation week", y=0.061, fontsize=10)
    figure.legend(handles=[
        Line2D([], [], color="#546c80", ls=":", label="Drift begins"),
        Line2D([], [], color="#92734f", ls="--", label="Alert threshold"),
        Line2D([], [], color="#b72c43", marker="o", markerfacecolor="none",
               ls="none", label="Alert"),
        Line2D([], [], color="#b72c43", marker="o", ls="none",
               label="First alert on shifted cases"),
    ], loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.savefig(path, dpi=150)


def write_reports(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    (out_dir / "report.md").write_text(markdown_report(report), encoding="ascii")
    plot_timeline(report, out_dir / "timeline.png")
