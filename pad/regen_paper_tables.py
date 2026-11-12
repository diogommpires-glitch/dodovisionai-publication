"""pad/regen_paper_tables.py -- regenerate reports/paper_tables.tex
from the freshest .md reports, preserving table labels for cross-refs.

Reads:
    reports/bootstrap_ci.md       -> Table 1 (operating points + 95% CI)
    reports/cross_dataset_matrix.md -> Table 2 (cross-dataset ACER/AUC)
    reports/ablation_matrix.md    -> Table 3 (drop-one-branch)
    reports/apcer_by_class.md     -> Table 4 (per-PAI APCER)

Writes:
    reports/paper_tables.tex      (\\input{}-ready, IJCB template-compatible)

Per POST_QUEUE_PLAN phase C step 13 + the v5 verification gate's
'paper LaTeX tables' freshness check.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
OUT = REPORTS / "paper_tables.tex"

# ----------------------------------------------------------------------
# Markdown table parsing
# ----------------------------------------------------------------------

def _md_table_rows(md_path: Path, header_pattern: str) -> List[List[str]]:
    """Return list of cell-arrays for the rows of the FIRST markdown table
    whose header line matches header_pattern (regex substring)."""
    if not md_path.exists():
        raise FileNotFoundError(md_path)
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    rows: List[List[str]] = []
    in_table = False
    saw_separator = False
    for line in lines:
        if not in_table:
            if re.search(header_pattern, line, flags=re.IGNORECASE):
                in_table = True
            continue
        if not saw_separator:
            if re.match(r"^\s*\|[\s:|\-]+\|\s*$", line):
                saw_separator = True
            continue
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _strip_md(s: str) -> str:
    """Remove markdown emphasis (`bold`, `code`) for LaTeX paste.
    NOTE: do NOT strip _underscores_ — our reports never use _italic_
    style and stripping breaks identifier-like strings such as
    meta_train_records_celeba_v5.
    """
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    return s.strip()


def _escape_tex(s: str) -> str:
    s = s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    return s


# ----------------------------------------------------------------------
# Table 1 — bootstrap CI
# ----------------------------------------------------------------------

TABLE1_HEAD = r"""\begin{table}[t]
  \centering
  \small
  \caption{%
    Per-dataset operating points at the min-ACER optimum, with 95\,\%
    bootstrap confidence intervals ($n_\text{bootstrap}=1000$,
    $\alpha=0.05$). CelebDF reported on the post-temporal MC3-18
    branch (video-level, $n{=}1889$). Numbers regenerated from
    \texttt{reports/bootstrap_ci.md} on \DATE.%
  }
  \label{tab:bootstrap_ci}
  \begin{tabular}{@{}l r c l l l@{}}
    \toprule
    Dataset & $n$ & $\tau^\star$ & APCER (95\,\% CI) & BPCER (95\,\% CI) & ACER (95\,\% CI) \\
    \midrule
"""

TABLE1_FOOT = r"""    \bottomrule
  \end{tabular}
\end{table}
"""

DATASET_PRETTY = {
    "meta_train_records_celeba_v5":            "CelebA-Spoof v5",
    "meta_train_records_replay_v5":            "Replay-Attack v5",
    "deepfake_temporal_v2_records_celebdf":    "Celeb-DF (temporal v2)",
    "audio_asvspoof_records_v3":               "ASVspoof 2019 LA",
    "celeba":     "CelebA-Spoof",
    "replay":     "Replay-Attack",
    "celebdf":    "Celeb-DF",
    "asvspoof":   "ASVspoof 2019 LA",
}


def _pretty(raw: str) -> str:
    """Pretty-print a dataset key, preserving underscores in unmatched
    keys but escaping for LaTeX."""
    return DATASET_PRETTY.get(raw, raw)


def emit_table1() -> str:
    rows = _md_table_rows(
        REPORTS / "bootstrap_ci.md",
        r"Dataset.*\|.*n.*\|.*threshold.*\|.*APCER",
    )
    body = []
    for r in rows:
        if len(r) < 6:
            continue
        ds_raw = _strip_md(r[0])
        ds = DATASET_PRETTY.get(ds_raw, ds_raw)
        n = r[1]
        tau = r[2]
        apcer = _strip_md(r[3])
        bpcer = _strip_md(r[4])
        acer = _strip_md(r[5])
        body.append(f"    {_escape_tex(ds)} & {n} & {tau} & {apcer} & {bpcer} & {acer} \\\\")
    return TABLE1_HEAD + "\n".join(body) + "\n" + TABLE1_FOOT


# ----------------------------------------------------------------------
# Table 2 — cross-dataset matrix
# ----------------------------------------------------------------------

TABLE2_HEAD = r"""\begin{table}[t]
  \centering
  \small
  \caption{%
    Cross-dataset generalisation. Each cell is
    \textbf{ACER (AUC)} at the per-cell optimum threshold; rows are
    the meta-fuser training source, columns the evaluation source.
    Diagonals are same-source. Regenerated from
    \texttt{reports/cross_dataset_matrix.md} on \DATE.%
  }
  \label{tab:cross_dataset}
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{@{}l cccc@{}}
    \toprule
    train $\backslash$ test & CelebA-S & Replay & Celeb-DF & ASVspoof \\
    \midrule
"""

TABLE2_FOOT = r"""    \bottomrule
  \end{tabular}
\end{table}
"""


def emit_table2() -> str:
    rows = _md_table_rows(
        REPORTS / "cross_dataset_matrix.md",
        r"train.*test.*celeba.*replay.*celebdf.*asvspoof",
    )
    body = []
    for r in rows:
        if len(r) < 5:
            continue
        ds = _pretty(_strip_md(r[0]))
        cells = [_strip_md(c) for c in r[1:5]]
        body.append(f"    {_escape_tex(ds)} & " + " & ".join(cells) + " \\\\")
    return TABLE2_HEAD + "\n".join(body) + "\n" + TABLE2_FOOT


# ----------------------------------------------------------------------
# Table 3 — ablation
# ----------------------------------------------------------------------

TABLE3_HEAD = r"""\begin{table}[t]
  \centering
  \small
  \caption{%
    Drop-one-branch ablation at fixed $\tau=0.5$. Cells are
    fused-fusion ACER with the named branch masked out; $\Delta$ is
    the absolute change vs.\ full fusion (positive means the branch
    is useful). Regenerated from \texttt{reports/ablation_matrix.md}
    on \DATE.%
  }
  \label{tab:ablation}
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}l r r r r r r@{}}
    \toprule
    Dataset & $n$ & full & $-$moire ($\Delta$) & $-$patch\_depth ($\Delta$) & $-$deepfake ($\Delta$) & $-$audio ($\Delta$) \\
    \midrule
"""

TABLE3_FOOT = r"""    \bottomrule
  \end{tabular}
\end{table}
"""


def emit_table3() -> str:
    rows = _md_table_rows(
        REPORTS / "ablation_matrix.md",
        r"Dataset.*\|.*n.*\|.*full.*ACER.*moire",
    )
    body = []
    for r in rows:
        if len(r) < 7:
            continue
        ds = _pretty(_strip_md(r[0]))
        body.append(
            f"    {_escape_tex(ds)} & {r[1]} & {_strip_md(r[2])} & "
            f"{_strip_md(r[3])} & {_strip_md(r[4])} & {_strip_md(r[5])} & {_strip_md(r[6])} \\\\"
        )
    return TABLE3_HEAD + "\n".join(body) + "\n" + TABLE3_FOOT


# ----------------------------------------------------------------------
# Table 4 — APCER per PAI
# ----------------------------------------------------------------------

TABLE4_HEAD = r"""\begin{table}[t]
  \centering
  \small
  \caption{%
    APCER per Presentation Attack Instrument (PAI), required by
    ISO/IEC\,30107-3~\S7.1. Decision threshold $\tau^\star$ is the
    per-dataset min-ACER optimum; BPCER is dataset-wide. Regenerated
    from \texttt{reports/apcer_by_class.md} on \DATE.%
  }
  \label{tab:apcer_pai}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}l l r r r r@{}}
    \toprule
    Dataset      & PAI class                       & $n$  & APCER $@\,\tau^\star$ & BPCER (ds) & $\tau^\star$ \\
    \midrule
"""

TABLE4_FOOT = r"""    \bottomrule
  \end{tabular}
\end{table}
"""


def emit_table4() -> str:
    rows = _md_table_rows(
        REPORTS / "apcer_by_class.md",
        r"Dataset.*\|.*PAI.*class.*\|.*n.*\|.*APCER",
    )
    body = []
    for r in rows:
        if len(r) < 6:
            continue
        ds = _pretty(_strip_md(r[0]))
        pai = _escape_tex(_strip_md(r[1]))
        body.append(
            f"    {_escape_tex(ds)} & {pai} & {r[2]} & {_strip_md(r[3])} & "
            f"{_strip_md(r[4])} & {_strip_md(r[5])} \\\\"
        )
    return TABLE4_HEAD + "\n".join(body) + "\n" + TABLE4_FOOT


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

PREAMBLE = r"""% =====================================================================
%  DodoVisionAI -- IJCB 2026 paper tables  (regenerated)
%  Source markdowns are read at regen time from reports/*.md.
%  Run `python -m pad.regen_paper_tables` to refresh.
%
%  Assumes the IJCB template loads:
%    \usepackage{booktabs}
%
%  Usage: \input{paper_tables.tex}
%  Reference via:
%     \ref{tab:bootstrap_ci}, \ref{tab:cross_dataset},
%     \ref{tab:ablation},     \ref{tab:apcer_pai}
% =====================================================================

"""


def main() -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    parts = [PREAMBLE.replace("\\DATE", today)]
    try:
        parts.append("\n% --- Table 1 (bootstrap CIs) ---\n")
        parts.append(emit_table1().replace("\\DATE", today))
        parts.append("\n\n% --- Table 2 (cross-dataset matrix) ---\n")
        parts.append(emit_table2().replace("\\DATE", today))
        parts.append("\n\n% --- Table 3 (ablation) ---\n")
        parts.append(emit_table3().replace("\\DATE", today))
        parts.append("\n\n% --- Table 4 (APCER per PAI) ---\n")
        parts.append(emit_table4().replace("\\DATE", today))
    except FileNotFoundError as e:
        print(f"[regen-paper] missing source: {e}", file=sys.stderr)
        return 1
    OUT.write_text("".join(parts), encoding="utf-8")
    sz = OUT.stat().st_size
    print(f"[regen-paper] wrote {OUT} ({sz:,} bytes)")
    print(f"[regen-paper] tables: bootstrap_ci, cross_dataset, ablation, apcer_pai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
