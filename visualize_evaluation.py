#!/usr/bin/env python3
"""
visualize_evaluation.py - Genera todas las figuras requeridas para el TFM
a partir de los resultados de evaluate_synthetic_corpus.py.

Figuras generadas:
  1. Matriz de confusion 3x3 (heatmap)
  2. F1/Precision/Recall por clase (barras agrupadas)
  3. Accuracy y F1 por familia documental
  4. Deteccion de firmas y casillas (P/R/F1)
  5. Deteccion de campos por tipo (heatmap)
  6. Tabla de ablacion (si disponible)
  7. Bootstrap CIs (forest plot)
  8. Distribucion de tiempos de procesamiento
  9. Errores por estado (valid/incomplete_1/incomplete_2/inconsistent_sem/inconsistent_mul)
  10. OCR confidence vs correctness

Uso:
  python visualize_evaluation.py --results-file data/results/synthetic/eval_reglas_*.json
                                 [--output-dir figures/]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# Paleta profesional
COLORS = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
    "bg": "#F8FAFC",
    "valid": "#10B981",
    "incomplete": "#F59E0B",
    "inconsistent": "#EF4444",
}

FAMILY_COLORS = {
    "CN-001-ES": "#2563EB",
    "DLR-001-ES": "#7C3AED",
    "MED-001-ES": "#10B981",
    "ADM-001-ES": "#F59E0B",
    "PREOP-001-ES": "#EF4444",
}

FAMILY_LABELS = {
    "CN-001-ES": "Notas Clinicas",
    "DLR-001-ES": "Inf. Diagnosticos",
    "MED-001-ES": "Lista Medicacion",
    "ADM-001-ES": "Form. Admision",
    "PREOP-001-ES": "Checklist Preop",
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })


def fig1_confusion_matrix(data, output_dir):
    """Matriz de confusion 3x3 con anotaciones."""
    cm = np.array(data["global_metrics"]["confusion_matrix"])
    classes = data["global_metrics"]["classes"]
    labels_es = {"valid": "Valido", "incomplete": "Incompleto", "inconsistent": "Inconsistente"}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # Anotaciones
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([labels_es.get(c, c) for c in classes], fontsize=11)
    ax.set_yticklabels([labels_es.get(c, c) for c in classes], fontsize=11)
    ax.set_xlabel("Prediccion del agente", fontsize=12)
    ax.set_ylabel("Veredicto esperado (ground truth)", fontsize=12)
    ax.set_title("Matriz de Confusion - Veredicto Global", fontsize=14, fontweight="bold")

    # Metricas en texto
    acc = data["global_metrics"]["accuracy"]
    kappa = data["global_metrics"]["kappa"]
    f1 = data["global_metrics"]["macro_f1"]
    ax.text(0.02, -0.15, f"Accuracy: {acc:.3f}  |  Macro F1: {f1:.3f}  |  Kappa: {kappa:.3f}",
            transform=ax.transAxes, fontsize=10, color="#444")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_confusion_matrix.png", bbox_inches="tight")
    plt.close()


def fig2_per_class_metrics(data, output_dir):
    """P/R/F1 por clase de veredicto."""
    per_class = data["global_metrics"]["per_class"]
    classes = list(per_class.keys())
    labels_es = {"valid": "Valido", "incomplete": "Incompleto", "inconsistent": "Inconsistente"}

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    p_vals = [per_class[c]["precision"] for c in classes]
    r_vals = [per_class[c]["recall"] for c in classes]
    f_vals = [per_class[c]["f1"] for c in classes]

    bars1 = ax.bar(x - width, p_vals, width, label="Precision", color=COLORS["primary"], alpha=0.85)
    bars2 = ax.bar(x, r_vals, width, label="Recall", color=COLORS["secondary"], alpha=0.85)
    bars3 = ax.bar(x + width, f_vals, width, label="F1-Score", color=COLORS["accent"], alpha=0.85)

    # Etiquetas sobre barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([labels_es.get(c, c) for c in classes])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title("Precision, Recall y F1 por Clase de Veredicto", fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_per_class_prf1.png", bbox_inches="tight")
    plt.close()


def fig3_per_family(data, output_dir):
    """Accuracy y F1 por familia documental."""
    per_family = data["per_family"]
    families = sorted(per_family.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(families))
    width = 0.35

    accs = [per_family[f]["accuracy"] for f in families]
    f1s = [per_family[f]["macro_f1"] for f in families]
    colors_fam = [FAMILY_COLORS.get(f, "#999") for f in families]

    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy",
                   color=[c + "CC" for c in colors_fam], edgecolor=colors_fam, linewidth=1.5)
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro F1",
                   color=colors_fam, alpha=0.7)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    labels = [FAMILY_LABELS.get(f, f) for f in families]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title("Rendimiento por Familia Documental", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_per_family.png", bbox_inches="tight")
    plt.close()


def fig4_signature_checkbox(data, output_dir):
    """Deteccion de firmas y casillas."""
    det = data["detection_metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (key, title) in zip(axes, [("signatures", "Deteccion de Firmas"),
                                        ("checkboxes", "Deteccion de Casillas")]):
        metrics = det[key]
        vals = [metrics["precision"], metrics["recall"], metrics["f1"]]
        labels = ["Precision", "Recall", "F1"]
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

        bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=11, fontweight="bold")

        ax.set_ylim(0, 1.2)
        ax.set_title(title, fontweight="bold")
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        ax.text(0.5, -0.12, f"TP={tp}  FP={fp}  FN={fn}",
                transform=ax.transAxes, ha="center", fontsize=9, color="#666")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_signatures_checkboxes.png", bbox_inches="tight")
    plt.close()


def fig5_field_detection(data, output_dir):
    """Heatmap de deteccion por campo."""
    fields = data["detection_metrics"].get("fields", {})
    if not fields:
        return

    field_names = sorted(fields.keys())
    metrics_names = ["precision", "recall", "f1"]
    matrix = np.array([[fields[f][m] for m in metrics_names] for f in field_names])

    fig, ax = plt.subplots(figsize=(6, max(4, len(field_names) * 0.5 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(len(field_names)):
        for j in range(len(metrics_names)):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color)

    ax.set_xticks(range(len(metrics_names)))
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_yticks(range(len(field_names)))
    ax.set_yticklabels(field_names, fontsize=9)
    ax.set_title("Deteccion por Campo Obligatorio", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "fig5_field_detection.png", bbox_inches="tight")
    plt.close()


def fig6_bootstrap_ci(data, output_dir):
    """Forest plot de intervalos de confianza bootstrap."""
    cis = data.get("confidence_intervals", {})
    if not cis:
        return

    metrics = []
    means = []
    lows = []
    highs = []

    labels_map = {
        "macro_f1": "Macro F1",
        "accuracy": "Accuracy",
        "kappa": "Kappa de Cohen",
    }

    for key in ["macro_f1", "accuracy", "kappa"]:
        if key in cis:
            ci = cis[key]
            metrics.append(labels_map.get(key, key))
            means.append(ci["mean"])
            lows.append(ci["ci_lower"])
            highs.append(ci["ci_upper"])

    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    y = np.arange(len(metrics))

    for i in range(len(metrics)):
        ax.plot([lows[i], highs[i]], [i, i], color=COLORS["primary"],
                linewidth=3, solid_capstyle="round")
        ax.plot(means[i], i, "o", color=COLORS["danger"], markersize=10, zorder=5)
        ax.text(highs[i] + 0.01, i,
                f"{means[i]:.3f} [{lows[i]:.3f}, {highs[i]:.3f}]",
                va="center", fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Valor")
    ax.set_title("Intervalos de Confianza Bootstrap (95%, n=1000)", fontweight="bold")
    ax.axvline(x=0.5, color="#CCC", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "fig6_bootstrap_ci.png", bbox_inches="tight")
    plt.close()


def fig7_timing(data, output_dir):
    """Distribucion de tiempos de procesamiento."""
    docs = data.get("documents", [])
    if not docs:
        return

    times = [d["elapsed_s"] for d in docs]
    families = sorted(set(d["family"] for d in docs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Histograma general
    ax1.hist(times, bins=20, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax1.axvline(np.mean(times), color=COLORS["danger"], linestyle="--",
                label=f"Media: {np.mean(times):.2f}s")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribucion de Tiempos", fontweight="bold")
    ax1.legend()

    # Boxplot por familia
    family_times = {f: [] for f in families}
    for d in docs:
        family_times[d["family"]].append(d["elapsed_s"])

    bp = ax2.boxplot(
        [family_times[f] for f in families],
        labels=[FAMILY_LABELS.get(f, f)[:12] for f in families],
        patch_artist=True,
    )
    for patch, f in zip(bp["boxes"], families):
        patch.set_facecolor(FAMILY_COLORS.get(f, "#999") + "80")
        patch.set_edgecolor(FAMILY_COLORS.get(f, "#999"))

    ax2.set_ylabel("Tiempo (s)")
    ax2.set_title("Tiempo por Familia", fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(output_dir / "fig7_timing.png", bbox_inches="tight")
    plt.close()


def fig8_error_analysis(data, output_dir):
    """Analisis de errores por estado del ground truth."""
    docs = data.get("documents", [])
    if not docs:
        return

    states = sorted(set(d["state"] for d in docs))
    state_correct = {s: 0 for s in states}
    state_total = {s: 0 for s in states}

    for d in docs:
        state_total[d["state"]] += 1
        if d["correct"]:
            state_correct[d["state"]] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(states))

    totals = [state_total[s] for s in states]
    corrects = [state_correct[s] for s in states]
    errors = [t - c for t, c in zip(totals, corrects)]

    ax.bar(x, corrects, label="Correctos", color=COLORS["valid"], alpha=0.8)
    ax.bar(x, errors, bottom=corrects, label="Errores", color=COLORS["danger"], alpha=0.8)

    for i in range(len(states)):
        rate = corrects[i] / totals[i] * 100 if totals[i] > 0 else 0
        ax.text(i, totals[i] + 0.3, f"{rate:.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=15, ha="right")
    ax.set_ylabel("Documentos")
    ax.set_title("Aciertos por Estado del Documento", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "fig8_error_by_state.png", bbox_inches="tight")
    plt.close()


def fig9_ocr_vs_correctness(data, output_dir):
    """OCR confidence vs correctness."""
    docs = data.get("documents", [])
    if not docs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    correct_conf = [d["ocr_confidence"] for d in docs if d["correct"]]
    wrong_conf = [d["ocr_confidence"] for d in docs if not d["correct"]]

    if correct_conf:
        ax.hist(correct_conf, bins=15, alpha=0.6, color=COLORS["valid"],
                label=f"Correctos (n={len(correct_conf)})", edgecolor="white")
    if wrong_conf:
        ax.hist(wrong_conf, bins=15, alpha=0.6, color=COLORS["danger"],
                label=f"Errores (n={len(wrong_conf)})", edgecolor="white")

    ax.set_xlabel("Confianza OCR (%)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Confianza OCR vs Correccion del Veredicto", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "fig9_ocr_vs_correctness.png", bbox_inches="tight")
    plt.close()


def fig10_summary_dashboard(data, output_dir):
    """Dashboard resumen con metricas clave."""
    gm = data["global_metrics"]
    ci = data.get("confidence_intervals", {})

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Dashboard de Evaluacion - PathwayGuard", fontsize=16, fontweight="bold", y=0.98)

    metrics_display = [
        ("Accuracy", gm["accuracy"], COLORS["primary"]),
        ("Macro F1", gm["macro_f1"], COLORS["secondary"]),
        ("Kappa Cohen", gm["kappa"], COLORS["accent"]),
        ("Sensibilidad", gm["sensitivity_valid"], COLORS["valid"]),
        ("Especificidad", gm["specificity_valid"], COLORS["danger"]),
        ("Weighted F1", gm["weighted_f1"], COLORS["neutral"]),
    ]

    for ax, (name, val, color) in zip(axes.flat, metrics_display):
        # Gauge-like display
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        # Background arc
        theta = np.linspace(0, np.pi, 100)
        r = 0.4
        ax.plot(0.5 + r * np.cos(theta), 0.3 + r * np.sin(theta),
                color="#E5E7EB", linewidth=12, solid_capstyle="round")

        # Value arc
        theta_val = np.linspace(0, np.pi * val, 100)
        ax.plot(0.5 + r * np.cos(theta_val), 0.3 + r * np.sin(theta_val),
                color=color, linewidth=12, solid_capstyle="round")

        ax.text(0.5, 0.35, f"{val:.3f}", ha="center", va="center",
                fontsize=24, fontweight="bold", color=color)
        ax.text(0.5, 0.15, name, ha="center", va="center",
                fontsize=12, color="#333")

        # CI if available
        key_map = {"Accuracy": "accuracy", "Macro F1": "macro_f1", "Kappa Cohen": "kappa"}
        if name in key_map and key_map[name] in ci:
            c = ci[key_map[name]]
            ax.text(0.5, 0.03, f"IC 95%: [{c['ci_lower']:.3f}, {c['ci_upper']:.3f}]",
                    ha="center", fontsize=8, color="#888")

        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "fig10_dashboard.png", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualizar resultados de evaluacion")
    parser.add_argument("--results-file", required=True, help="JSON de resultados")
    parser.add_argument("--output-dir", default="figures", help="Directorio de figuras")
    args = parser.parse_args()

    setup_style()

    with open(args.results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generators = [
        ("1. Matriz de confusion", fig1_confusion_matrix),
        ("2. P/R/F1 por clase", fig2_per_class_metrics),
        ("3. Por familia", fig3_per_family),
        ("4. Firmas y casillas", fig4_signature_checkbox),
        ("5. Deteccion campos", fig5_field_detection),
        ("6. Bootstrap CIs", fig6_bootstrap_ci),
        ("7. Tiempos", fig7_timing),
        ("8. Errores por estado", fig8_error_analysis),
        ("9. OCR vs correctness", fig9_ocr_vs_correctness),
        ("10. Dashboard", fig10_summary_dashboard),
    ]

    for name, fn in generators:
        try:
            fn(data, out)
            print(f"  OK: {name}")
        except Exception as e:
            print(f"  ERROR: {name} -> {e}")

    print(f"\n{len(generators)} figuras generadas en {out}/")


if __name__ == "__main__":
    main()
