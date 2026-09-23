"""Views of an explanation: terminal, SVG, and a self-contained HTML report.

Charts are hand-rolled SVG with no plotting dependency, so `pip install jev-why`
stays small and a diagram can be produced in a CI job or a container with
nothing extra installed.

Two rules from the house style drive the design. A chart never stands alone: a
heatmap ships beside a ranked table and a sentence naming the span that moved
the decision, because a coloured dot on its own leaves the reader guessing.
And colour never carries meaning by itself -- the sign is in the table, the
significance is in the text, and the colour is the fast path, not the only one.

Colour roles, per the data-viz method:
  attribution   diverging (signed): red pushed the answer up, blue pushed it
                down, neutral grey at zero. Never a rainbow, never a hue at
                the midpoint.
  two-series    categorical slots 1 and 2 (blue, orange), validated for
                colour-vision deficiency in both light and dark surfaces.
  reference     the perfect-calibration diagonal is chrome, not a series, so
                it is drawn recessive and dashed.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from jev_why.calibration import CalibrationReport
from jev_why.faithfulness import FaithfulnessReport
from jev_why.types import Attribution, Explanation, QuestionExplanation

# Validated against the light and dark chart surfaces; see the data-viz
# reference palette. Do not substitute by eye.
LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series1": "#2a78d6",
    "series2": "#eb6834",
    "neutral": "#f0efec",
    "positive": "#d03b3b",
    "negative": "#2a78d6",
}
DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series1": "#3987e5",
    "series2": "#d95926",
    "neutral": "#383835",
    "positive": "#e66767",
    "negative": "#3987e5",
}

# Single quotes inside the stack on purpose: this string is interpolated into
# double-quoted SVG and HTML attributes, where a nested double quote would
# terminate the attribute and produce malformed XML.
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _tokens() -> str:
    light = "\n".join(f"    --{k}: {v};" for k, v in LIGHT.items())
    dark = "\n".join(f"    --{k}: {v};" for k, v in DARK.items())
    return f"""
  .viz {{ color-scheme: light;
{light}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz {{ color-scheme: dark;
{dark}
    }}
  }}
  :root[data-theme="dark"] .viz {{ color-scheme: dark;
{dark}
  }}
"""


def _svg(width: int, height: int, body: str, *, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(title)}" class="viz" '
        f'style="max-width:100%;height:auto;font-family:{FONT}">'
        f"<style><![CDATA[{_tokens()}]]></style>"
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>'
        f"{body}</svg>"
    )


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = 11,
    fill: str = "var(--ink2)",
    anchor: str = "start",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(content)}</text>'
    )


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------


def _intensity(attribution: Attribution, peak: float) -> float:
    if peak <= 0 or not attribution.significant:
        return 0.0
    return min(1.0, abs(attribution.phi) / peak)


def headline(question: QuestionExplanation) -> str:
    """One sentence naming what moved the decision.

    This is the part that stops a heatmap from being decoration. It is printed
    above every rendering of an explanation, including the terminal one.
    """
    top = question.top(1)
    if not top:
        return (
            f"No span in this state moved {question.question!r} by more than the "
            f"measured noise floor ({question.noise_sigma:.4f}). The explanation "
            "is inconclusive."
        )
    best = top[0]
    direction = "raised" if best.phi > 0 else "lowered"
    redundant = abs(best.necessity) < abs(best.sufficiency) / 3
    tail = (
        " Other spans say the same thing, so removing it alone changes little." if redundant else ""
    )
    return (
        f"{best.span.label} {direction} {question.question!r} by "
        f"{abs(best.phi):.3f} ({question.link}): “{best.span.preview(70)}”.{tail}"
    )


def attribution_table(question: QuestionExplanation, *, k: int = 8) -> str:
    """A ranked table. Preferred over the chart when only one can be shown."""
    rows = question.top(k, include_insignificant=True)
    if not rows:
        return "no attributions"
    width = max(len(r.span.label) for r in rows)
    lines = [
        f"{'span'.ljust(width)}  {'effect':>8}  {'necess':>7}  {'suffic':>7}  text",
        f"{'-' * width}  {'-' * 8}  {'-' * 7}  {'-' * 7}  {'-' * 40}",
    ]
    for row in rows:
        flag = "" if row.significant else "  (below noise floor)"
        lines.append(
            f"{row.span.label.ljust(width)}  {row.phi:>+8.3f}  {row.necessity:>+7.3f}  "
            f"{row.sufficiency:>+7.3f}  {row.span.preview(40)}{flag}"
        )
    return "\n".join(lines)


def heatmap_text(explanation: Explanation, question: str, *, colour: bool = True) -> str:
    """The state with its spans tinted by signed effect, for a terminal."""
    q = explanation[question]
    by_index = {a.span.index: a for a in q.attributions}
    peak = max((abs(a.phi) for a in q.attributions), default=0.0)

    out: list[str] = []
    for span in explanation.spans:
        attribution = by_index.get(span.index)
        text = span.text
        if attribution is None or not colour:
            out.append(text)
            continue
        weight = _intensity(attribution, peak)
        if weight < 0.15:
            out.append(f"\033[2m{text}\033[0m" if colour else text)
        else:
            # 256-colour ramp: reds for positive, blues for negative. The sign
            # is also in the table, so colour is never the only channel.
            steps = [217, 210, 203, 196] if attribution.phi > 0 else [153, 111, 75, 39]
            code = steps[min(3, int(weight * 4))]
            out.append(f"\033[38;5;{code}m{text}\033[0m")
    return " ".join(out)


def heatmap_html(explanation: Explanation, question: str) -> str:
    q = explanation[question]
    by_index = {a.span.index: a for a in q.attributions}
    peak = max((abs(a.phi) for a in q.attributions), default=0.0)

    parts: list[str] = []
    for span in explanation.spans:
        attribution = by_index.get(span.index)
        escaped = html.escape(span.text)
        if attribution is None:
            parts.append(escaped)
            continue
        weight = _intensity(attribution, peak)
        pole = "var(--positive)" if attribution.phi > 0 else "var(--negative)"
        style = (
            f"background:color-mix(in srgb, {pole} {weight * 45:.0f}%, transparent);"
            "border-radius:3px;padding:1px 2px;"
        )
        if not attribution.significant:
            style += "opacity:.45;"
        parts.append(
            f'<mark style="{style}" title="{attribution.phi:+.4f} '
            f"(necessity {attribution.necessity:+.3f}, sufficiency "
            f'{attribution.sufficiency:+.3f})">{escaped}</mark>'
        )
    return " ".join(parts)


# --------------------------------------------------------------------------
# Reliability diagram
# --------------------------------------------------------------------------


def reliability_svg(
    report: CalibrationReport, *, width: int = 380, title: str = "Calibration"
) -> str:
    """Predicted against observed, with Wilson bars and the count histogram.

    The histogram panel is not decoration. Without it an ECE is uninterpretable:
    a bin holding nine observations and a bin holding nine thousand look alike
    on the curve and mean completely different things.
    """
    pad_l, pad_r = 46, 14
    side = width - pad_l - pad_r  # the plot is square: both axes are probabilities
    plot_top = 62
    plot_bottom = plot_top + side
    tick_y = plot_bottom + 15
    axis_title_y = plot_bottom + 31
    hist_label_y = plot_bottom + 52
    hist_top = plot_bottom + 58
    hist_h = 44
    footer_y = hist_top + hist_h + 20
    height = footer_y + 10

    def px(v: float) -> float:
        return pad_l + v * side

    def py(v: float) -> float:
        return plot_top + (1 - v) * side

    body = [
        _text(pad_l, 20, title, size=13, fill="var(--ink)", weight=600),
        _text(pad_l, 34, report.parts.reads_as(), size=11, fill="var(--muted)"),
    ]

    # A legend rather than a label on the line itself: an inline label sits on
    # whichever curve happens to pass through that corner.
    body.append(
        f'<line x1="{pad_l}" y1="46" x2="{pad_l + 16}" y2="46" stroke="var(--series1)" '
        f'stroke-width="2"/>'
    )
    body.append(_text(pad_l + 22, 49, "observed", size=10, fill="var(--ink2)"))
    body.append(
        f'<line x1="{pad_l + 86}" y1="46" x2="{pad_l + 102}" y2="46" stroke="var(--axis)" '
        f'stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    body.append(_text(pad_l + 108, 49, "perfect calibration", size=10, fill="var(--ink2)"))

    for i in range(6):
        v = i / 5
        body.append(
            f'<line x1="{px(0):.1f}" y1="{py(v):.1f}" x2="{px(1):.1f}" y2="{py(v):.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        body.append(
            _text(pad_l - 8, py(v) + 4, f"{v:.1f}", size=10, fill="var(--muted)", anchor="end")
        )
        body.append(_text(px(v), tick_y, f"{v:.1f}", size=10, fill="var(--muted)", anchor="middle"))

    # Chrome, not a series: perfect calibration is the reference the data is
    # read against, so it recedes.
    body.append(
        f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(1):.1f}" y2="{py(1):.1f}" '
        f'stroke="var(--axis)" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )

    bins = report.curve.non_empty
    for b in bins:
        x = px(b.mean_predicted)
        body.append(
            f'<line x1="{x:.1f}" y1="{py(b.ci_low):.1f}" x2="{x:.1f}" '
            f'y2="{py(b.ci_high):.1f}" stroke="var(--series1)" stroke-width="1.5" '
            f'opacity="0.5"/>'
        )
    points = [(b.mean_predicted, b.observed) for b in bins]
    if len(points) > 1:
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        body.append(
            f'<polyline points="{path}" fill="none" stroke="var(--series1)" '
            f'stroke-width="2" stroke-linejoin="round"/>'
        )
    for x, y in points:
        # 2px surface ring keeps overlapping markers separable.
        body.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="var(--series1)" '
            f'stroke="var(--surface)" stroke-width="2"/>'
        )

    body.append(
        _text(
            pad_l + side / 2,
            axis_title_y,
            "predicted probability",
            size=10,
            fill="var(--muted)",
            anchor="middle",
        )
    )
    body.append(
        _text(pad_l, hist_label_y, "where the predictions sit", size=9, fill="var(--muted)")
    )

    # Uniform bins, not the calibration bins. Those are equal-mass, so their
    # counts are identical by construction and a panel drawn from them is a
    # flat row of bars carrying no information at all.
    counts = report.distribution
    if counts:
        peak = max(counts) or 1
        slot = side / len(counts)
        for i, count in enumerate(counts):
            if not count:
                continue
            bar_h = max(1.0, (count / peak) * hist_h)
            body.append(
                f'<rect x="{pad_l + i * slot + 1:.1f}" '
                f'y="{hist_top + hist_h - bar_h:.1f}" '
                f'width="{max(1.5, slot - 2):.1f}" height="{bar_h:.1f}" rx="2" '
                f'fill="var(--series1)" opacity="0.38"/>'
            )

    body.append(
        _text(
            pad_l,
            footer_y,
            f"ECE {report.ece:.3f} \u00b7 AUROC {report.auroc:.3f} \u00b7 "
            f"slope {report.slope:.2f} \u00b7 n={report.n:,}",
            size=10,
            fill="var(--ink2)",
        )
    )
    return _svg(width, int(height), "".join(body), title=f"{title}: reliability diagram")


# --------------------------------------------------------------------------
# Faithfulness
# --------------------------------------------------------------------------


def faithfulness_svg(
    report: FaithfulnessReport, *, width: int = 480, title: str = "Faithfulness"
) -> str:
    """The attributed deletion curve against the random control.

    Two series, so a legend is present and both are directly labelled: identity
    is never carried by colour alone.
    """
    points = report.curve
    if not points:
        return _svg(width, 80, _text(20, 44, "no faithfulness data"), title=title)

    pad_l, pad_r = 48, 96
    plot_w = width - pad_l - pad_r
    plot_top, plot_h = 48, 176
    plot_bottom = plot_top + plot_h
    tick_y = plot_bottom + 15
    axis_title_y = plot_bottom + 31
    footer_y = plot_bottom + 50
    height = footer_y + 10

    top = max(max(p.comprehensiveness, p.random_comprehensiveness) for p in points)
    top = top if top > 0 else 1.0
    span = max(p.fraction for p in points) or 1.0

    def px(fraction: float) -> float:
        return pad_l + fraction / span * plot_w

    def py(value: float) -> float:
        return plot_bottom - (value / top) * plot_h

    body = [
        _text(pad_l, 20, title, size=13, fill="var(--ink)", weight=600),
        _text(pad_l, 34, report.verdict(), size=11, fill="var(--muted)"),
    ]
    for i in range(5):
        y = plot_bottom - i / 4 * plot_h
        body.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        body.append(
            _text(
                pad_l - 8, y + 4, f"{i / 4 * top:.2f}", size=10, fill="var(--muted)", anchor="end"
            )
        )

    band = (
        " ".join(f"{px(p.fraction):.1f},{py(p.random_comprehensiveness):.1f}" for p in points)
        + f" {px(points[-1].fraction):.1f},{py(0):.1f}"
        + f" {px(points[0].fraction):.1f},{py(0):.1f}"
    )
    body.append(f'<polygon points="{band}" fill="var(--series2)" opacity="0.13"/>')

    for key, colour, label in (
        ("random_comprehensiveness", "var(--series2)", "random spans"),
        ("comprehensiveness", "var(--series1)", "attributed"),
    ):
        path = " ".join(f"{px(p.fraction):.1f},{py(getattr(p, key)):.1f}" for p in points)
        body.append(
            f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for p in points:
            body.append(
                f'<circle cx="{px(p.fraction):.1f}" cy="{py(getattr(p, key)):.1f}" r="3.5" '
                f'fill="{colour}" stroke="var(--surface)" stroke-width="2"/>'
            )
        last = points[-1]
        body.append(
            _text(
                px(last.fraction) + 9,
                py(getattr(last, key)) + 4,
                label,
                size=10,
                fill="var(--ink2)",
            )
        )

    # Skip a tick label that would collide with the one before it.
    placed = -1e9
    for p in points:
        x = px(p.fraction)
        if x - placed < 30:
            continue
        body.append(
            _text(x, tick_y, f"{p.fraction:.0%}", size=10, fill="var(--muted)", anchor="middle")
        )
        placed = x
    body.append(
        _text(
            pad_l + plot_w / 2,
            axis_title_y,
            "share of spans deleted",
            size=10,
            fill="var(--muted)",
            anchor="middle",
        )
    )
    body.append(
        _text(
            pad_l,
            footer_y,
            f"lift {report.lift:+.3f} over random \u00b7 p={report.p_value:.3f} "
            f"\u00b7 {report.random_trials} trials",
            size=10,
            fill="var(--ink2)",
        )
    )
    return _svg(width, int(height), "".join(body), title=f"{title}: deletion curve")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportSection:
    question: str
    headline: str
    heatmap: str
    table: str
    faithfulness: str | None


def explanation_html(explanation: Explanation, *, title: str = "jev-why") -> str:
    """A self-contained report: no scripts, no external resources, every piece
    of the analysed text escaped.

    It is safe to open. It is not safe to share more widely than its input,
    because a report about a document necessarily contains that document.
    """
    sections: list[ReportSection] = []
    for name, question in explanation.questions.items():
        report = explanation.faithfulness.get(name)
        sections.append(
            ReportSection(
                question=name,
                headline=headline(question),
                heatmap=heatmap_html(explanation, name),
                table=attribution_table(question),
                faithfulness=faithfulness_svg(report, title=f"{name}: faithfulness")
                if report is not None
                else None,
            )
        )

    blocks = "".join(
        f"""
    <section>
      <h2>{html.escape(s.question)}</h2>
      <p class="headline">{html.escape(s.headline)}</p>
      <div class="doc">{s.heatmap}</div>
      <pre>{html.escape(s.table)}</pre>
      {s.faithfulness or ""}
    </section>"""
        for s in sections
    )

    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in explanation.warnings)
    warning_block = f"<ul class='warn'>{warnings}</ul>" if warnings else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{_tokens()}
  body {{ margin:0; background:var(--plane); color:var(--ink); font-family:{FONT};
         line-height:1.55; }}
  main {{ max-width:860px; margin:0 auto; padding:32px 16px 64px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:32px 0 8px; font-weight:600; }}
  .meta, .headline {{ color:var(--ink2); font-size:13px; margin:0 0 16px; }}
  .doc {{ background:var(--surface); border:1px solid var(--grid); border-radius:8px;
          padding:16px; font-size:14px; }}
  pre {{ background:var(--surface); border:1px solid var(--grid); border-radius:8px;
         padding:12px 16px; overflow-x:auto; font-size:12px; color:var(--ink2); }}
  .warn {{ color:var(--ink2); font-size:12px; padding-left:18px; }}
  section {{ margin-bottom:8px; }}
</style></head>
<body class="viz"><main>
  <h1>{html.escape(title)}</h1>
  <p class="meta">{explanation.spend.calls} calls, {explanation.spend.cache_hits} cached,
     ${explanation.spend.cost_usd:.4f}, model {html.escape(explanation.model_version)},
     mask {html.escape(explanation.mask_mode.value)}</p>
  {warning_block}
  {blocks}
</main></body></html>
"""


def explanation_text(explanation: Explanation, question: str, *, colour: bool = True) -> str:
    q = explanation[question]
    return "\n\n".join(
        [
            headline(q),
            heatmap_text(explanation, question, colour=colour),
            attribution_table(q),
            f"baseline {q.baseline:.4f} → fully redacted {q.empty:.4f} "
            f"| noise floor {q.noise_sigma:.4f} | efficiency gap {q.efficiency_gap:.0%} "
            f"| {q.estimator}",
        ]
    )


__all__ = [
    "attribution_table",
    "explanation_html",
    "explanation_text",
    "faithfulness_svg",
    "headline",
    "heatmap_html",
    "heatmap_text",
    "reliability_svg",
]
