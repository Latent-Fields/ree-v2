# Planning Artifacts

## Weekly Handoff Usage

Generate the deterministic weekly handoff markdown:

```bash
python3 scripts/generate_weekly_handoff.py --output evidence/planning/weekly_handoff/latest.md
```

Validate required sections/columns before sharing:

```bash
python3 scripts/validate_weekly_handoff.py --input evidence/planning/weekly_handoff/latest.md
```

Optional deterministic timestamp/date overrides for replayable outputs:

```bash
python3 scripts/generate_weekly_handoff.py \
  --output evidence/planning/weekly_handoff/latest.md \
  --week-of-utc 2026-02-09 \
  --generated-utc 2026-02-14T16:30:00Z
```
