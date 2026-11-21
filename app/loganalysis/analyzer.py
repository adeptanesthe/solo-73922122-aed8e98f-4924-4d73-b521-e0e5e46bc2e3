"""Rule-based log anomaly extraction.

Each rule maps a regex to (severity, fault category, human label). Matched
lines become log clues with 1-based line numbers. The analyzer also detects
*positive* signals (recovery / healthy markers) used later to flag
contradictions (e.g. errors followed by a clean recovery).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    severity: float  # 0..1
    category: str
    label: str


def _rx(pat: str) -> re.Pattern[str]:
    return re.compile(pat, re.IGNORECASE)


RULES: list[Rule] = [
    Rule("panic", _rx(r"\b(panic|fatal|critical|segfault|core dumped)\b"), 0.95, "application", "panic/fatal"),
    Rule("oom", _rx(r"\b(oom|out of memory|oomkilled|cannot allocate memory)\b"), 0.9, "resource", "out-of-memory"),
    Rule("disk", _rx(r"\b(no space left|disk full|read-only file system)\b"), 0.9, "resource", "disk exhaustion"),
    Rule("timeout", _rx(r"\b(timeout|timed out|deadline exceeded|read timed out)\b"), 0.8, "dependency", "timeout"),
    Rule("conn", _rx(r"\b(connection refused|connection reset|no route to host|unreachable|dns|name resolution)\b"), 0.8, "network", "connectivity failure"),
    Rule("auth", _rx(r"\b(unauthorized|forbidden|permission denied|invalid token|authentication failed|bad credentials)\b|\b(401|403)\b"), 0.8, "auth", "authn/authz failure"),
    Rule("db", _rx(r"\b(deadlock|too many connections|relation .* does not exist|database .* (down|unavailable)|sqlstate)\b"), 0.8, "database", "database error"),
    Rule("http5xx", _rx(r"\b(5\d{2})\b(?=.*\b(status|responded|returned|http)\b)|\b(status|responded|returned)\s+(5\d{2})\b|\bhttp[/ ]\d\.\d.?\s+(5\d{2})\b"), 0.75, "application", "HTTP 5xx"),
    Rule("circuit", _rx(r"\b(circuit[ =_-]?open|circuit breaker|backoff|rate ?limit|429)\b"), 0.7, "dependency", "dependency protection triggered"),
    Rule("error", _rx(r"\b(error|exception|traceback|failed|failure|cannot|unable to)\b"), 0.7, "application", "generic error"),
    Rule("warn", _rx(r"\b(warn|warning|deprecated|retrying|retries)\b"), 0.45, "application", "warning"),
]

POSITIVE_PATTERN = _rx(
    r"\b(recovered|healthy|started|listening|ready|ok|success|succeeded|up)\b|\b(200|201|204)\b"
)


@dataclass
class LogClue:
    line_no: int  # 1-based
    text: str
    severity: float
    category: str
    rule: str
    label: str
    matched: list[str] = field(default_factory=list)


@dataclass
class LogAnalysis:
    total_lines: int
    clues: list[LogClue]
    positive_lines: list[int]  # line numbers with healthy/recovery markers
    recovery_after_error: bool  # positive signal appears after the last error clue

    @property
    def max_severity(self) -> float:
        return max((c.severity for c in self.clues), default=0.0)


def analyze_logs(text: str, max_clues: int = 40) -> LogAnalysis:
    lines = text.splitlines()
    clues: list[LogClue] = []
    positive_lines: list[int] = []
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        best: Rule | None = None
        matched: list[str] = []
        for rule in RULES:
            m = rule.pattern.search(line)
            if m:
                matched.append(rule.label)
                if best is None or rule.severity > best.severity:
                    best = rule
        if best is not None:
            clues.append(
                LogClue(
                    line_no=i,
                    text=line.strip()[:300],
                    severity=best.severity,
                    category=best.category,
                    rule=best.name,
                    label=best.label,
                    matched=matched,
                )
            )
        if POSITIVE_PATTERN.search(line):
            positive_lines.append(i)

    # keep the most severe clues if there are too many
    if len(clues) > max_clues:
        clues = sorted(clues, key=lambda c: c.severity, reverse=True)[:max_clues]
        clues.sort(key=lambda c: c.line_no)

    last_error = max((c.line_no for c in clues if c.severity >= 0.7), default=-1)
    recovery = last_error > 0 and any(p > last_error for p in positive_lines)
    return LogAnalysis(
        total_lines=len(lines),
        clues=clues,
        positive_lines=positive_lines,
        recovery_after_error=recovery,
    )
