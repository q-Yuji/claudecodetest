# BLUEPRINT 5: Friction journal — one-week "ugh, this again" capture experiment

**BUILDER:** Claude Haiku, working alone, cold start, cannot ask questions.
(Two tiny files, no tricky decisions left — Haiku is enough.)

## GOAL

A zero-thought way for the user to log a moment of repeated friction from
anywhere on his PC (one hotkey → type a phrase → done), plus a review
command that turns a week of entries into a ranked shortlist. Purpose: this
is the agreed product-discovery experiment — log "ugh, this again" moments
for one week across non-trading screen time, then run the demand/wedge/
distribution filter on the list to pick the next build-for-self product.

## CONTEXT THE BUILDER NEEDS

- Files to read first: `scripts/log_trade_hotkey.ahk` — copy its structure
  (global hotkey, header comment explaining Startup-folder install). It
  focuses VS Code; THIS script must NOT do that — it shows an AutoHotkey
  `InputBox` instead so capture works without leaving the current app.
- Entries live in `journal/frictions.md` (the `journal/` folder is the
  durable, tracked record per CLAUDE.md).
- Entry format (exact):
  `- 2026-07-07 19:42 | [Google Chrome — Gmail] | copying the same trade numbers between three tabs again`
  where the bracket part is the active window title at hotkey press
  (AutoHotkey `WinGetActiveTitle`), truncated to 60 chars.

## CONSTRAINTS

- Must stay inside: `scripts/friction_hotkey.ahk` (new),
  `tools/friction_review.py` (new, create `tools/` folder),
  `journal/frictions.md` (new, seeded with a header).
- Must not change: anything else. Do not touch the F11 or (if it exists)
  F9 scripts.
- Stack: AutoHotkey v1 syntax; Python 3.14 stdlib.
- Non-negotiables: capture must take < 5 seconds and never steal focus
  from the InputBox; the log file is append-only.

## STEP-BY-STEP PLAN

1. Create `journal/frictions.md` with header:
   `# Friction journal — "ugh, this again" log` and one line explaining
   the experiment (one week, then review) and the entry format example
   above.
2. Create `scripts/friction_hotkey.ahk`: hotkey `F8`; on press: capture
   active window title; show `InputBox` "What's the friction? (one line)";
   if non-empty, append the formatted line (local time HH:mm) to
   `C:\Users\lucap\.vscode-shared\claudecodetest\journal\frictions.md`
   with UTF-8 encoding (`FileEncoding, UTF-8`); silent otherwise. Header
   comment documents the Startup-folder shortcut install, same wording
   style as log_trade_hotkey.ahk.
3. Create `tools/friction_review.py`:
   `python tools/friction_review.py [--since YYYY-MM-DD]`
   parses frictions.md lines, groups case-insensitively by fuzzy key
   (first: exact app-title bucket; second: entries whose descriptions share
   3+ common words ≥4 chars), and prints a ranked table: count, app,
   representative description, dates seen. Bottom line prints the filter
   reminder verbatim: `Filter: my daily friction AND a reachable crowd
   shares it → demand / wedge / distribution.`
4. Append one line to `journal/frictions.md` header: "Review with:
   `python tools/friction_review.py`".
5. Test: append 4 fake entries (2 similar), run the review, confirm the
   similar pair groups together and ranks first; then REMOVE the fake
   entries, leaving only the header.

## EXACT INPUTS TO USE

- Kickoff prompt: "Open blueprints/friction-journal.md and build it exactly
  as written. Three files only."
- Hotkey: `F8`. Entry format: exactly as shown in CONTEXT.

## DEFINITION OF DONE

- [ ] F8 from any app pops an input box; entering text appends one
      correctly formatted line; Esc/empty appends nothing.
- [ ] `python tools/friction_review.py` on the 4-fake-entry fixture groups
      the similar pair and exits 0.
- [ ] `journal/frictions.md` ends the build containing ONLY the header
      lines (fixtures removed).
- [ ] AutoHotkey script header documents Startup install.

## IF SOMETHING IS UNCLEAR

Smallest safe assumption, `ASSUMPTION:` tag, keep going. Do not build a GUI,
a database, or a web app — this is a one-week text-file experiment.
