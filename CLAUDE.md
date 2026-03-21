# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Jarvis is a static front-end dashboard. No build step, no dependencies, no package manager. Open `index.html` directly in a browser to run it.

```bash
open index.html
```

## Architecture

Three files, each with a distinct responsibility:

- **`index.html`** — all markup and layout. Stat cards and activity chart bars are hardcoded here; task list items are also seeded here but can be added dynamically via JS.
- **`style.css`** — all visual design. Theming is done entirely through CSS custom properties defined in `:root` (colors, sidebar width). The activity bar heights are driven by the `--h` CSS variable set inline on each `.bar` element.
- **`app.js`** — all interactivity: live clock/greeting (updates every second via `setInterval`), sidebar nav active-state toggling, and task list management (add task, toggle done state).

## Git workflow

Commit work to git regularly throughout a session — after each meaningful change, not just at the end. Push to GitHub (`git push`) so there is always an up-to-date remote backup. Use clean, descriptive commit messages that explain *why* the change was made. A Stop hook in `~/.claude/settings.json` auto-pushes on session end, but don't rely on that alone — commit and push at logical checkpoints.

## Design conventions

- Dark theme only. Color palette lives in `:root` in `style.css` — always use those variables, never hardcode colors.
- Layout is sidebar (fixed, 200px) + `.main` (flexbox column, `margin-left: var(--sidebar-w)`).
- New dashboard sections/panels should use the `.card` class.
- The `.hidden` utility class (`display: none`) is used to show/hide the add-task form.
- Stat change indicators use `.positive`, `.negative`, `.neutral` modifier classes on `.stat-change`.
