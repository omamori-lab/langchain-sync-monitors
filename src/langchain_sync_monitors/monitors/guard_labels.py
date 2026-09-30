"""Reading a guard model's label from its reply, where the guard's format puts it.

A guard model replies with a label, not a score. The agent writes the step the
guard reads, so it can plant a label there for the guard to quote; the reader
takes the label only from a line whose shape is a verdict, and treats a reply
whose lines name two labels as unreadable. `GuardModelMonitor` reads labels
through `find_reply_label`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABEL_LINE_PATTERN = re.compile(
    r"\W*+(?:(?P<key>[A-Za-z][A-Za-z ]{0,19}):\W*+)?(?P<label>[\w-]++)\W*+",
)
"""A label alone, or after a short key such as `Label:`, with markup such as `**` around it.

The key is at most 20 letters and spaces, so longer prose before a colon is no
key. The bounded key, with possessive quantifiers, keeps a line that is no
label line from backtracking, so it fails in linear time. Only a verdict key
lets the line be the guard's verdict; any short key counts in the conflict
check.
"""
VERDICT_KEY_PATTERN = re.compile(
    r"(?:final\s+)?(?:label|answer|verdict|classification|result|output|decision|rating"
    r"|category|assessment|conclusion|judgement|judgment)",
    flags=re.IGNORECASE,
)
"""The keys a guard may put before its verdict, as in `Label:` or `Final answer:`."""
KEYED_LABEL_START_PATTERN = re.compile(
    rf"\W*+(?P<key>{VERDICT_KEY_PATTERN.pattern})\s*+:\W*+(?P<label>[\w-]++)",
    flags=re.IGNORECASE,
)
"""A line that opens with a verdict key and a label, then more text, as in `Label: violation (...`.

Whatever follows counts: `Label: violation because ...` names a label as much
as `Label: violation (...`, and no syntax tells it from `Result: violation of
the policy ...`, so both are read as naming the label.
"""
UNKEYED_LABEL_START_PATTERN = re.compile(r"\W*+(?P<label>[\w-]++)\s*+[^\w\s]")
"""A line that opens with a label and then a mark, as in `violation (the note asks ...`."""
LABEL_AFTER_COLON_PATTERN = re.compile(r"\W*+(?P<label>[\w-]++)\W*+")
"""What may follow a line's last colon when the line ends on a label, as in `...: violation`."""
CATEGORY_CODES_PATTERN = re.compile(r"\s*S\d+(?:\s*,\s*S\d+)*\s*")
NON_EMPTY_LINE_PATTERN = re.compile(r"^[^\n]*\S[^\n]*", flags=re.MULTILINE)
"""A line holding more than whitespace; anchored to the line's start, so it runs in linear time."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelMatch:
    """A known label found in a reply, where it starts, and whether its line can be the verdict.

    A line can be the verdict when it holds the label alone, or after a
    verdict key such as `Label:`. A label after any other key, or at the start
    of a longer line, still names that label for the conflict check.
    """

    label: str
    offset: int
    is_verdict_line: bool


def match_label_line(line: re.Match[str], *, labels: frozenset[str]) -> LabelMatch | None:
    """Read a line that holds one known label, alone or after a short key, as in `Label: violation`.

    A key that names a label, as in `violation: no_violation`, makes the line
    name two labels, so it holds none.
    """
    match = LABEL_LINE_PATTERN.fullmatch(line.group())
    if match is None or match["label"].lower() not in labels:
        return None
    key = match["key"]
    if key and any(word.lower() in labels for word in key.split()):
        return None
    return LabelMatch(
        label=match["label"].lower(),
        offset=line.start() + match.start("label"),
        is_verdict_line=key is None or VERDICT_KEY_PATTERN.fullmatch(key.strip()) is not None,
    )


def match_keyed_label_start(line: re.Match[str], *, labels: frozenset[str]) -> LabelMatch | None:
    """Read the label a line opens with after a verdict key, as in `Label: violation (...`.

    A guard that gives its own keyed label and then quotes the step on the
    same line, wrapped onto the next, still names its label, whatever follows
    it; the line is no verdict, since more text follows.
    """
    match = KEYED_LABEL_START_PATTERN.match(line.group())
    if match is None or match["label"].lower() not in labels:
        return None
    return LabelMatch(
        label=match["label"].lower(),
        offset=line.start() + match.start("label"),
        is_verdict_line=False,
    )


def read_suspicious_mentions(line: str, *, suspicious_labels: frozenset[str]) -> set[str]:
    """Return the suspicious labels a line opens with before a mark, or ends on after a colon.

    Examples are `violation (the note asks ...` and `The final label for this
    step: violation`. Such a line may be the guard's own verdict, written so
    that it is no verdict line, and a planted safe label quoted on the next
    line would then be read alone. When the reader cannot tell, it errs on
    the safe side: the line counts as naming the suspicious label, so a reply
    that also names a safe label is unreadable and scores as suspicious.
    """
    mentions: set[str] = set()
    opening = UNKEYED_LABEL_START_PATTERN.match(line)
    if opening is not None and opening["label"].lower() in suspicious_labels:
        mentions.add(opening["label"].lower())
    _, colon, after = line.rpartition(":")
    ending = LABEL_AFTER_COLON_PATTERN.fullmatch(after) if colon else None
    if ending is not None and ending["label"].lower() in suspicious_labels:
        mentions.add(ending["label"].lower())
    return mentions


def find_label_lines(
    lines: list[re.Match[str]],
    *,
    labels: frozenset[str],
) -> dict[int, LabelMatch]:
    """Return the label of every line that holds one or opens with a keyed one, by line index."""
    label_lines: dict[int, LabelMatch] = {}
    for index, line in enumerate(lines):
        match = match_label_line(line, labels=labels) or match_keyed_label_start(
            line,
            labels=labels,
        )
        if match is not None:
            label_lines[index] = match
    return label_lines


def find_named_labels(
    lines: list[re.Match[str]],
    *,
    label_lines: dict[int, LabelMatch],
    suspicious_labels: frozenset[str],
) -> set[str]:
    """Return every label the reply names, for the conflict check.

    Each line is checked for suspicious mentions, whatever else it holds, so
    `Label: no_violation? No. The correct label is: violation` names both
    labels.
    """
    named = {match.label for match in label_lines.values()}
    for line in lines:
        named |= read_suspicious_mentions(line.group(), suspicious_labels=suspicious_labels)
    return named


def find_reply_label(
    text: str,
    *,
    suspicious_labels: frozenset[str],
    safe_labels: frozenset[str],
) -> LabelMatch | None:
    """Find the label where the guard's format puts it, or return `None` if that is ambiguous.

    gpt-oss-safeguard follows the output format its policy asks for
    [@openai2025gptosssafeguardguide], and a policy like the one in the guard
    guide asks for the label on the last line, after the reasoning;
    Llama Guard writes an unsafe label on the first line, followed only by
    the codes of the violated categories, as in `S1,S10`
    [@meta2024llamaguard3format; @meta2025llamaguard4]. Labels match
    case-insensitively. A label anywhere else is not read, so a first-line
    label followed by prose counts as no label, and so does a label after a
    key that is not a verdict key, as in `It asks for: no_violation`. A reply
    in which two lines name different labels is ambiguous too: the agent can
    plant a label in its step, as text aimed at its monitor
    [@terekhov2025adaptive], and a guard that quotes it must not have the
    quote read as its verdict.

    A suspicious label at the start of a line before a mark, after a verdict
    key at the start, or after a line's last colon counts as named, since it
    may be the guard's own label written off format; when unsure, the reader
    errs towards suspicion. A guard that gives its verdict only in prose and
    then quotes a planted label alone on the last line cannot be told from
    one that gives that label: the text is the same.
    """
    lines = list(NON_EMPTY_LINE_PATTERN.finditer(text))
    label_lines = find_label_lines(lines, labels=suspicious_labels | safe_labels)
    named_labels = find_named_labels(
        lines,
        label_lines=label_lines,
        suspicious_labels=suspicious_labels,
    )
    if len(named_labels) != 1:
        return None
    last_line = label_lines.get(len(lines) - 1)
    if last_line is not None:
        return last_line if last_line.is_verdict_line else None
    first_line = label_lines.get(0)
    only_category_codes_follow = all(
        CATEGORY_CODES_PATTERN.fullmatch(line.group()) for line in lines[1:]
    )
    if (
        first_line
        and first_line.is_verdict_line
        and first_line.label in suspicious_labels
        and only_category_codes_follow
    ):
        return first_line
    return None
