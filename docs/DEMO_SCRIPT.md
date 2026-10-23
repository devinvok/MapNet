# Two-minute demo script

Goal: show, in one take, the three things no single existing tool combines:
**private, live, cheap**. Optionally a fourth: agents learning from each other.

## Setup (before recording)

```bash
ollama pull qwen2.5:3b
cp -r examples/company /tmp/demo-company          # a scratch copy you can change
rm -f .mapnet-notes.json
```

Two terminal windows side by side: left runs `ask.py`, right is for dropping files.

## Take

**0:00 — Offline.** Turn Wi-Fi off on camera.

```bash
python ask.py /tmp/demo-company
```

Ask: *"By when must we give notice to avoid auto-renewing the Tessaro Logistics contract?"*
Point at: the answer (January 30, 2027), the source file, "opened 3 of 18 files".
Say: *"A 3-billion-parameter model on this laptop. No internet. It didn't read every file; the map sent it to the right three."*

**0:30 — Live update.** Ask: *"How much does the Pro plan cost per robot?"* (89 USD).
In the right window:

```bash
cp examples/drop-in/pricing.md /tmp/demo-company/sales/pricing.md
cp examples/drop-in/2026-10-23-checkout-outage.md /tmp/demo-company/engineering/
```

The left window prints `[map] changed: sales/pricing.md (7 ms)`. Ask the price
again (now mentions 95 USD from January 2027), then: *"Is there an ongoing
checkout incident, and who's the incident commander?"*
Say: *"No re-indexing, no retraining. The file existed for two seconds."*

**1:00 — Permissions.** Ask: *"On what day are salaries paid?"* → no payroll answer: the employee view can't reach it.
Type `/role hr` and ask again → the 10th of each month.
Say: *"Payroll has an .access file. For everyone else, that folder is never even opened."*

**1:20 — Cost.** Cut to a pre-recorded run of:

```bash
ANTHROPIC_API_KEY=... python compare.py --cloud --price-in <in> --price-out <out>
```

Show the results table: correct answers and cost per 1,000 questions for the
local model versus the cloud model. (Use the cloud model's current list prices.)

**1:40 — Agents (optional).** Run `python agents_demo.py`. Point at "files opened
per task: 10.0 → 3.5 → 1.0" and at the planted false note being deleted.

**1:55 — Close.** *"Small model, private, always current. It knows where to look."*

## Tips

- Run the whole take once before recording: the first Ollama call loads the model
  and is slower.
- Keep the terminal font large; the dim status line is the proof, make sure it's readable.
- Don't claim the numbers generalize: say "on this sample company".
