# Tools

## Blind Human Play

Use `play_game.py` to play any ARC-AGI-3 game through the public frame/action
interface and save first-contact human data.

```bash
.venv/bin/python tools/play_game.py --game ls20
```

For games not already downloaded locally, use normal mode with `ARC_API_KEY`
loaded from `.env`:

```bash
set -a
source .env
set +a
.venv/bin/python tools/play_game.py --game ar25 --mode normal
```

Each run writes a timestamped folder under `human-play/<game>/` containing:

- `trace.jsonl` - admissible frame/action trace readable by `experiments.harness.viewer`
- `notes.md` - blind-play notes template plus live `n <note>` entries
- `summary.json` - compact run metadata
- `scorecard.json` - ARC scorecard snapshot
- `raw-recordings/` - ARC toolkit recording with full public frames

Commands during play:

- `1`..`7` - public ARC actions
- `6 <x> <y>` or `click <x> <y>` - coordinate action when available
- `r` or `reset` - reset through the local wrapper
- `c` or `coords` - toggle display-pixel coordinate rulers
- `i`, `inspect`, `tiles`, or `palette` - list visible same-color components with click coordinates
- `n <text>` - append a timestamped note
- `s` - print status
- `h` - print help
- `q` - quit and save

Coordinate actions use the public 64x64 display frame. The compact terminal
renderer compresses two vertical pixels into one text row, so coordinate mode
labels rows as `00/01`, `02/03`, and so on.

For games with clickable color tiles, `inspect` is usually easier than counting
against the ruler. It prints rows like `color=4 ... -> click 47 6`; paste the
`click x y` command to send `ACTION6` at that visible component.
