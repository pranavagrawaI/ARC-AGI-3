# Ready: current solver + shallow MTP

Upload q6_mtp.ipynb with kernel-metadata.json sources, attached to
arc-prize-2026-arc-agi-3, RTX PRO 6000, internet off, session limit4800s.
This is the actual official offline launcher and ls20 environment with the frozen
prediction-feedback solver, not a synthetic engine prompt. Ten actions maximum.

Run the separate actor-mtp replay first for engine attribution. This notebook is
the integration pilot: actions, tool recovery, request accounting and shutdown.
Neither notebook has been launched by this preparation.

Local deterministic rebuild: `python3 build.py --check`.
The builder verifies the frozen donor notebook, every embedded solver file, the
join archive and the modified engine runtime. It does not edit the running donor.
Read EXPERIMENT-CARD.md for caps, evidence gates and limitations.

Download the whole output directory. INTEGRATION-IDENTITY.json identifies the
candidate; nooa-join-public3 contains launcher/model/accounting/lifecycle logs.
Use the donor assessor for host evidence only, then review MTP telemetry and actual
transitions separately. No action count alone proves solver quality or speed.
