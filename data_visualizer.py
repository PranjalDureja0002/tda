# Paste this into a Custom Code component's Code tab
# Data Visualizer Tool — LLM-powered intelligent chart generation
# Exposed as a Tool for Worker Node via tool_mode=True
#
# Unlike basic chart tools, this uses the LLM to:
#   1. Extract data from ANY format (JSON, markdown tables, text paragraphs)
#   2. Plan the optimal chart configuration (type, axes, sorting, grouping)
#   3. Generate contextual titles and labels
#   4. Apply smart transformations (top-N + "Other", pivoting, sorting)

from agentcore.custom import Node
import json
import base64
import io
import re


# ═══════════════════════════════════════════════════════════════════════════════
# CHART STYLE PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

STYLE_PRESETS = {
    "corporate": {
        "colors": ["#1e3a5f", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d", "#3b1f2b", "#44bba4", "#e94f37",
                    "#5b8c5a", "#8338ec", "#ff006e", "#3a86ff", "#fb5607", "#ffbe0b"],
        "bg_color": "#ffffff",
        "text_color": "#333333",
        "grid_color": "#e0e0e0",
        "title_size": 16,
        "label_size": 11,
        "tick_size": 10,
    },
    "modern": {
        "colors": ["#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316", "#eab308", "#22c55e", "#06b6d4",
                    "#a855f7", "#14b8a6", "#f59e0b", "#ef4444", "#3b82f6", "#10b981"],
        "bg_color": "#0f172a",
        "text_color": "#e2e8f0",
        "grid_color": "#1e293b",
        "title_size": 16,
        "label_size": 11,
        "tick_size": 10,
    },
    "executive": {
        "colors": ["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd", "#c4a35a", "#b08968", "#7f5539",
                    "#606c38", "#283618", "#dda15e", "#bc6c25", "#264653", "#2a9d8f"],
        "bg_color": "#fafaf9",
        "text_color": "#1c1917",
        "grid_color": "#e7e5e4",
        "title_size": 16,
        "label_size": 11,
        "tick_size": 10,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# LLM CHART PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

CHART_PLANNER_PROMPT = """You are a data visualization expert. Analyze the data and user request, then output a chart configuration as JSON.

**User Request:** {user_request}

**Data:**
{data_preview}

**Available chart types:** bar, bar_horizontal, line, pie, scatter, stacked_bar, table

**Respond with ONLY this JSON structure:**
{{
    "chart_type": "bar",
    "title": "Descriptive business title for the chart",
    "x_column": 0,
    "y_columns": [1],
    "x_label": "Human-readable X axis label",
    "y_label": "Human-readable Y axis label",
    "sort_by_value": true,
    "sort_descending": true,
    "top_n": null,
    "group_others": false,
    "annotations": ["optional insight to show on chart"]
}}

**Decision rules:**
- Pie: ONLY for 2-8 categories showing proportions/shares. Set group_others=true if >6 items.
- Bar: For comparisons, rankings, top-N. Use bar_horizontal if labels are long or >10 items.
- Line: For time series, trends, monthly/quarterly data. x_column must be the date/time column.
- Stacked bar: For composition across categories (e.g., spend by region broken down by material type).
- Scatter: For correlations between two numeric columns.
- Table: Fallback if data has too many columns or doesn't suit any chart.
- sort_by_value=true for rankings/comparisons, false for time series.
- top_n: Set to 10-15 if there are many rows and user asked for "top" or data would be cluttered.
- title: Make it a proper business title (e.g., "Top 10 Suppliers by Total Spend (EUR)" not "SUPPLIER_NAME vs TOTAL_SPEND_EUR").
- x_label/y_label: Human readable (e.g., "Total Spend (EUR)" not "TOTAL_SPEND_EUR").
- annotations: One brief data insight (e.g., "Top 3 suppliers account for 45% of total").

Return ONLY the JSON, no explanations."""


DATA_EXTRACTOR_PROMPT = """Extract structured data from the following content. The user wants to visualize this data.

**Content:**
{content}

**User Request:** {user_request}

**Return ONLY this JSON:**
{{
    "columns": ["Column1", "Column2"],
    "rows": [["val1", 123], ["val2", 456]]
}}

Rules:
- Extract column headers and all data rows
- Convert numeric values to numbers (not strings)
- If the content is a markdown table, parse it
- If the content is text with numbers, extract the key data points
- If you can't extract meaningful data, return {{"columns": [], "rows": []}}

Return ONLY the JSON."""


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _try_parse_data_json(text):
    """Try to extract structured data from <data_json> tags or raw JSON."""
    # Try <data_json> tags
    tag_match = re.search(r'<data_json>(.*?)</data_json>', text, re.DOTALL)
    if tag_match:
        try:
            data = json.loads(tag_match.group(1).strip())
            if data.get("columns") and data.get("rows"):
                return data, text[tag_match.end():].strip()
        except json.JSONDecodeError:
            pass

    # Try raw JSON object with columns/rows
    json_match = re.search(r'\{[^{}]*"columns"\s*:\s*\[.*?\].*?"rows"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if data.get("columns") and data.get("rows"):
                return data, text[json_match.end():].strip()
        except json.JSONDecodeError:
            pass

    return None, text


def _try_parse_markdown_table(text):
    """Try to extract data from a markdown table."""
    lines = text.strip().split("\n")
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:  # need header + separator + at least 1 data row
        return None

    # Parse header
    header = table_lines[0]
    columns = [c.strip() for c in header.split("|") if c.strip()]
    if not columns:
        return None

    # Skip separator row(s)
    data_start = 1
    for i in range(1, len(table_lines)):
        if re.match(r'^[\s|:-]+$', table_lines[i]):
            data_start = i + 1
        else:
            break

    # Parse data rows
    rows = []
    for line in table_lines[data_start:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) != len(columns):
            continue
        parsed_row = []
        for cell in cells:
            # Try to parse as number
            clean = cell.replace(",", "").replace("—", "").replace("NULL", "").strip()
            if not clean or clean == "—":
                parsed_row.append(None)
            else:
                try:
                    parsed_row.append(float(clean) if "." in clean else int(clean))
                except ValueError:
                    parsed_row.append(cell)
        rows.append(parsed_row)

    if rows:
        return {"columns": columns, "rows": rows}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHART RENDERING (matplotlib)
# ═══════════════════════════════════════════════════════════════════════════════

def _chart_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def _render_chart(plan, columns, rows, style_name):
    """Render a chart based on the LLM's plan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["corporate"])
    colors = style["colors"]
    chart_type = plan.get("chart_type", "bar")
    title = plan.get("title", "Data Visualization")
    x_col = plan.get("x_column", 0)
    y_cols = plan.get("y_columns", [1])
    x_label = plan.get("x_label", "")
    y_label = plan.get("y_label", "")
    sort_by_value = plan.get("sort_by_value", False)
    sort_desc = plan.get("sort_descending", True)
    top_n = plan.get("top_n")
    group_others = plan.get("group_others", False)
    annotations = plan.get("annotations", [])

    def _safe_float(v):
        if v is None:
            return 0
        try:
            return float(str(v).replace(",", ""))
        except (ValueError, TypeError):
            return 0

    # Prepare data with sorting and top-N
    work_rows = list(rows)

    if sort_by_value and y_cols:
        y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
        work_rows.sort(key=lambda r: _safe_float(r[y_idx] if y_idx < len(r) else 0), reverse=sort_desc)

    if top_n and len(work_rows) > top_n:
        top_rows = work_rows[:top_n]
        rest_rows = work_rows[top_n:]
        if group_others and y_cols:
            # Aggregate the rest into "Other"
            other_vals = []
            for y_idx in y_cols:
                total = sum(_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in rest_rows)
                other_vals.append(total)
            other_row = [None] * len(columns)
            other_row[x_col] = f"Other ({len(rest_rows)})"
            for i, y_idx in enumerate(y_cols):
                if y_idx < len(other_row):
                    other_row[y_idx] = other_vals[i]
            top_rows.append(other_row)
        work_rows = top_rows

    # Extract labels and values
    labels = [str(r[x_col])[:30] if x_col < len(r) else "?" for r in work_rows]

    # Create figure
    fig_width = max(10, min(16, len(labels) * 0.8))
    fig_height = 6 if chart_type != "bar_horizontal" else max(6, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(style["bg_color"])
    ax.set_facecolor(style["bg_color"])
    ax.tick_params(colors=style["text_color"], labelsize=style["tick_size"])
    ax.xaxis.label.set_color(style["text_color"])
    ax.yaxis.label.set_color(style["text_color"])
    ax.title.set_color(style["text_color"])
    for spine in ax.spines.values():
        spine.set_color(style["grid_color"])
    ax.grid(True, alpha=0.3, color=style["grid_color"], linestyle="--")

    # ── Render by chart type ──

    if chart_type in ("bar", "stacked_bar"):
        if len(y_cols) > 1:
            # Grouped or stacked bar
            num_groups = len(work_rows)
            if chart_type == "stacked_bar":
                bottom = [0] * num_groups
                for i, y_idx in enumerate(y_cols):
                    vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
                    col_name = columns[y_idx] if y_idx < len(columns) else f"Series {i+1}"
                    ax.bar(labels, vals, bottom=bottom, label=col_name,
                           color=colors[i % len(colors)], edgecolor="none", width=0.7)
                    bottom = [b + v for b, v in zip(bottom, vals)]
            else:
                bar_width = 0.8 / len(y_cols)
                x_pos = list(range(num_groups))
                for i, y_idx in enumerate(y_cols):
                    vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
                    col_name = columns[y_idx] if y_idx < len(columns) else f"Series {i+1}"
                    offset = (i - (len(y_cols) - 1) / 2) * bar_width
                    positions = [x + offset for x in x_pos]
                    ax.bar(positions, vals, bar_width * 0.9, label=col_name,
                           color=colors[i % len(colors)], edgecolor="none")
                ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["tick_size"])
            ax.legend(fontsize=9, facecolor=style["bg_color"], edgecolor=style["grid_color"],
                      labelcolor=style["text_color"])
        else:
            y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
            vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
            bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
            bars = ax.bar(labels, vals, color=bar_colors, edgecolor="none", width=0.7)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["tick_size"])
            # Value labels on bars
            max_val = max(vals) if vals else 1
            for bar, val in zip(bars, vals):
                if val > 0:
                    text = f"{val:,.0f}" if abs(val) >= 100 else f"{val:,.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_val * 0.01,
                            text, ha="center", va="bottom",
                            fontsize=8, color=style["text_color"], fontweight="bold")

    elif chart_type == "bar_horizontal":
        y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
        vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
        bar_colors = [colors[i % len(colors)] for i in range(len(vals))]
        bars = ax.barh(labels, vals, color=bar_colors, edgecolor="none", height=0.65)
        ax.invert_yaxis()
        max_val = max(vals) if vals else 1
        for bar, val in zip(bars, vals):
            if val > 0:
                text = f"{val:,.0f}" if abs(val) >= 100 else f"{val:,.2f}"
                ax.text(bar.get_width() + max_val * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f" {text}", ha="left", va="center",
                        fontsize=9, color=style["text_color"], fontweight="bold")

    elif chart_type == "line":
        for i, y_idx in enumerate(y_cols):
            vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
            col_name = columns[y_idx] if y_idx < len(columns) else f"Series {i+1}"
            line, = ax.plot(labels, vals, marker="o", markersize=5, linewidth=2.5,
                            color=colors[i % len(colors)], label=col_name)
            if len(y_cols) == 1:
                ax.fill_between(range(len(vals)), vals, alpha=0.1, color=colors[i % len(colors)])
                # Value labels on points
                for j, (lbl, val) in enumerate(zip(labels, vals)):
                    if val > 0:
                        ax.annotate(f"{val:,.0f}" if abs(val) >= 100 else f"{val:,.2f}",
                                    (j, val), textcoords="offset points", xytext=(0, 8),
                                    ha="center", fontsize=7, color=style["text_color"])
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=style["tick_size"])
        if len(y_cols) > 1:
            ax.legend(fontsize=9, facecolor=style["bg_color"], edgecolor=style["grid_color"],
                      labelcolor=style["text_color"])

    elif chart_type == "pie":
        y_idx = y_cols[0] if y_cols[0] < len(columns) else 1
        vals = [_safe_float(r[y_idx] if y_idx < len(r) else 0) for r in work_rows]
        # Filter zero/negative
        filtered = [(l, v) for l, v in zip(labels, vals) if v > 0]
        if filtered:
            pie_labels, pie_vals = zip(*filtered)
            pie_colors = [colors[i % len(colors)] for i in range(len(pie_vals))]
            total = sum(pie_vals)
            # Custom labels with values
            display_labels = [f"{l}\n({v/total*100:.1f}%)" for l, v in zip(pie_labels, pie_vals)]
            wedges, texts, autotexts = ax.pie(
                pie_vals, labels=display_labels, colors=pie_colors,
                autopct=lambda p: f"{p*total/100:,.0f}" if p > 3 else "",
                startangle=140, pctdistance=0.75,
                textprops={"fontsize": style["tick_size"], "color": style["text_color"]},
                wedgeprops={"edgecolor": style["bg_color"], "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color("white")
                at.set_fontweight("bold")
            ax.axis("equal")

    elif chart_type == "scatter":
        if len(y_cols) >= 1 and len(columns) >= 3:
            y1 = y_cols[0] if y_cols[0] < len(columns) else 1
            y2 = y_cols[1] if len(y_cols) > 1 and y_cols[1] < len(columns) else (y1 + 1 if y1 + 1 < len(columns) else y1)
            x_vals = [_safe_float(r[y1] if y1 < len(r) else 0) for r in work_rows]
            y_vals = [_safe_float(r[y2] if y2 < len(r) else 0) for r in work_rows]
            ax.scatter(x_vals, y_vals, c=colors[0], alpha=0.7, edgecolors=colors[1], s=80, linewidths=1)
            if x_label:
                ax.set_xlabel(x_label, fontsize=style["label_size"])
            if y_label:
                ax.set_ylabel(y_label, fontsize=style["label_size"])

    # ── Labels and title ──
    if chart_type not in ("pie", "scatter"):
        if x_label:
            ax.set_xlabel(x_label, fontsize=style["label_size"], color=style["text_color"], labelpad=10)
        if y_label:
            ax.set_ylabel(y_label, fontsize=style["label_size"], color=style["text_color"], labelpad=10)

    ax.set_title(title, fontsize=style["title_size"], fontweight="bold",
                 color=style["text_color"], pad=20)

    if chart_type not in ("pie",):
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}" if abs(x) >= 1 else f"{x:.2f}"))

    # ── Annotation (insight text at bottom) ──
    if annotations:
        fig.text(0.5, -0.02, annotations[0][:120],
                 ha="center", fontsize=9, color=style["text_color"],
                 style="italic", alpha=0.7)

    plt.tight_layout()
    b64 = _chart_to_base64(fig)
    plt.close(fig)
    return b64


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

class CodeEditorNode(Node):
    display_name = "Data Visualizer"
    description = (
        "ONLY use this tool when the user EXPLICITLY asks to plot, chart, graph, or visualize data. "
        "Do NOT call this for normal data queries. Requires data from a previous query result. "
        "Pass the data_json and chart instructions as input."
    )
    icon = "bar-chart-3"
    name = "DataVisualizer"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Data + Instructions",
            info="Data (JSON, markdown table, or text) + optional chart instructions.",
            tool_mode=True,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="LLM for intelligent chart planning and data extraction.",
            required=True,
        ),
        DropdownInput(
            name="chart_style",
            display_name="Chart Style",
            options=["corporate", "modern", "executive"],
            value="corporate",
            info="Visual style preset.",
        ),
    ]

    outputs = [
        Output(display_name="Chart", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:
        raw_input = self.input_value or ""
        if not raw_input.strip():
            return Message(text="No data provided. Call Talk to Data first to get data, then ask me to chart it.")

        # ── Step 1: Extract structured data ──
        data = None
        instructions = raw_input

        # Try parsing data_json tags or JSON
        parsed, remaining = _try_parse_data_json(raw_input)
        if parsed:
            data = parsed
            instructions = remaining or "visualize this data"

        # Try parsing markdown table
        if not data:
            parsed = _try_parse_markdown_table(raw_input)
            if parsed:
                data = parsed
                # Instructions = anything that's not the table
                non_table_lines = [l for l in raw_input.split("\n") if not l.strip().startswith("|")]
                instructions = "\n".join(non_table_lines).strip() or "visualize this data"

        # Last resort: ask LLM to extract data from text
        if not data:
            data = self._llm_extract_data(raw_input, instructions)

        if not data or not data.get("columns") or not data.get("rows"):
            return Message(text="Could not extract chartable data from the input. Please provide a data table or data_json.")

        columns = data["columns"]
        rows = data["rows"]

        # ── Step 2: LLM Chart Planning ──
        plan = self._llm_plan_chart(columns, rows, instructions)

        # ── Step 3: Render Chart ──
        try:
            chart_type = plan.get("chart_type", "bar")

            # Fallback to table if LLM says so
            if chart_type == "table":
                return self._render_as_table(columns, rows, plan.get("title", "Data"))

            b64_image = _render_chart(plan, columns, rows, self.chart_style)

            parts = []
            parts.append(f"![{plan.get('title', 'Chart')}](data:image/png;base64,{b64_image})")
            parts.append(f"\n*{chart_type.replace('_', ' ').title()} chart — {len(rows)} data points*")

            # Add insight annotation if available
            annotations = plan.get("annotations", [])
            if annotations:
                parts.append(f"\n> {annotations[0]}")

            # Echo back data_json so follow-up chart modifications have data available
            # (e.g. "change x axis to supplier name" needs the data from this response)
            data_json_obj = {"columns": columns, "rows": rows}
            parts.append(f"\n<data_json>{json.dumps(data_json_obj)}</data_json>")

            self.status = f"{chart_type} | {len(rows)} rows"
            return Message(text="\n".join(parts))

        except Exception as e:
            self.status = f"Chart error: {e}"
            return self._render_as_table(columns, rows, plan.get("title", "Data"), error=str(e))

    # ─────────────────────────────────────────────────────────────────────
    # LLM HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _llm_plan_chart(self, columns, rows, user_request):
        """Ask the LLM to plan the optimal chart configuration."""
        # Build a compact data preview (first 10 rows)
        preview_rows = rows[:10]
        preview = f"Columns: {columns}\n"
        preview += f"Total rows: {len(rows)}\n"
        preview += f"Sample data (first {min(10, len(rows))} rows):\n"
        for row in preview_rows:
            preview += f"  {row}\n"

        # Detect numeric columns for context
        numeric_cols = []
        for i, col in enumerate(columns):
            if rows and isinstance(rows[0][i] if i < len(rows[0]) else None, (int, float)):
                numeric_cols.append(f"{col} (index {i})")
        if numeric_cols:
            preview += f"Numeric columns: {', '.join(numeric_cols)}\n"

        prompt = CHART_PLANNER_PROMPT.format(
            user_request=user_request,
            data_preview=preview,
        )

        # Default fallback plan
        fallback = {
            "chart_type": "bar",
            "title": "Data Visualization",
            "x_column": 0,
            "y_columns": [1] if len(columns) > 1 else [0],
            "x_label": columns[0] if columns else "",
            "y_label": columns[1] if len(columns) > 1 else "",
            "sort_by_value": True,
            "sort_descending": True,
            "top_n": None,
            "group_others": False,
            "annotations": [],
        }

        try:
            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            plan = json.loads(text.strip())
            # Validate critical fields
            if "chart_type" not in plan:
                plan["chart_type"] = fallback["chart_type"]
            if "x_column" not in plan:
                plan["x_column"] = fallback["x_column"]
            if "y_columns" not in plan:
                plan["y_columns"] = fallback["y_columns"]
            return plan
        except Exception:
            return fallback

    def _llm_extract_data(self, content, user_request):
        """Ask the LLM to extract structured data from unstructured content."""
        # Truncate very long content
        content_preview = content[:3000] if len(content) > 3000 else content

        prompt = DATA_EXTRACTOR_PROMPT.format(
            content=content_preview,
            user_request=user_request,
        )

        try:
            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
            if data.get("columns") and data.get("rows"):
                return data
        except Exception:
            pass
        return None

    def _render_as_table(self, columns, rows, title="Data", error=None):
        """Fallback: render as markdown table."""
        parts = []
        if error:
            parts.append(f"*Chart generation failed ({error}). Showing data table:*\n")
        parts.append(f"**{title}**\n")
        header = "| " + " | ".join(str(c) for c in columns) + " |"
        sep = "| " + " | ".join("---" for _ in columns) + " |"
        row_lines = []
        for row in rows[:50]:
            cells = []
            for v in row:
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:,.2f}")
                elif isinstance(v, int) and abs(v) >= 1000:
                    cells.append(f"{v:,}")
                else:
                    cells.append(str(v)[:50])
            row_lines.append("| " + " | ".join(cells) + " |")
        parts.append("\n".join([header, sep] + row_lines))
        if len(rows) > 50:
            parts.append(f"\n*Showing 50 of {len(rows)} rows*")
        self.status = f"Table: {len(rows)} rows"
        return Message(text="\n".join(parts))
