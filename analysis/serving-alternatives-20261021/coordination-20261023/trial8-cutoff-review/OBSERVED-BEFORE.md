# Trial8 cutoff persistence witness

Observed by independent reviewer on 2026-10-23 before the owner repair. This
preserves the code and original observed output; the saved script was not rerun
after preservation. Packet files were not edited by the reviewer.

## Actual replay witness

```bash
python3 analysis/serving-alternatives-20261021/coordination-20261023/trial8-cutoff-review/reproduce.py replay
```

The existing local HTTP fixture sends complete 228-byte responses immediately.
Only writes to body.bin are replaced: delay 0.3 seconds, then raise OSError.
The replay has 0.05 seconds warmup plus 0.05 seconds measurement. Thus the final
response bytes are read before cutoff, but persistence fails after cutoff.

Observed result:

```json
{"run_ok": true, "failed": false, "response_events": 0, "pending": 2}
```

Both HTTP receipts recorded status 200, Content-Length 228, received_bytes 228,
complete false, and `OSError('synthetic persistence failure after final body read')`.
One receipt had connection_started 334293.212296109, started 334293.212577945,
ended 334293.514218193; the other had connection_started 334293.212246101,
started 334293.212456985, ended 334293.513647496. Both had complete request
write receipts, 193 header bytes and 81 body bytes, ending before 334293.213.

Mechanism in the reviewed replay.py: final body read sets body_completed at
line123, but output.write raises before complete=True at line125. The exception
at line128 skips fail because stop is already set. Line130 replaces the earlier
body clock with exception time because complete is false.

## Reducer false-positive witness

```bash
python3 analysis/serving-alternatives-20261021/coordination-20261023/trial8-cutoff-review/reproduce.py reducer
```

This uses the existing analytical qualifying pair. It gives one pending control
receipt the same full-byte, failed-persistence shape and an empty body.bin.
Only package provenance is stubbed, exactly as in the existing positive fixture;
the arm, load, accounting, and ratio reducers execute normally.

Observed output:

```json
{"verdict": "QUALITY_INVESTIGATION_ONLY", "ratio": 2.0, "issues": [], "control_valid": true, "control_issues": []}
```

The pending loop checks lost completions only when complete is true and does not
reject this persistence error. A received completion before cutoff can therefore
be censored, changing the numerator without invalidating comparison. The required
outcome is fail-closed: retain completion timing independently from persistence
success, reject persistence failures even after cutoff, and withhold the ratio.

No GPU, external network, launch, or broader tests were used. The replay witness
uses a localhost HTTP server.
