Local transport proposal only. See EXPERIMENT-CARD.md, transport-manifest.json, qualification.log, and freeze.json. The only notebook is unassigned and must refuse before extraction. There is no launch.py or slot confirmation in this proposal. The integration patch is for future review; do not apply it to the frozen failed launcher.

The first qualification log is retained: a test asserted the generic word "credential" was absent and incorrectly matched the explanatory body_policy string. Narrowing that assertion to the actual cookie fixture fixed the test; capture behavior was unchanged.
