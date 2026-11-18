# R10a repair-ready bytes for independent re-review

Both defects reproduced BEFORE repair: 2 reviewer probes passed as defect reproductions (`BEFORE.stdout`, `before-probes/`). Reviewer-owned evidence is untouched.

After repair: 2 changed-path acceptance regressions passed (`AFTER.stdout`, `after-probes/`); complete required mapped-user wrapper suite 16 passed in95.67s (`SUITE.stdout`, `full-suite/`). Source diff and embedded notebook/inventory regenerated. Budget, game, card, semantic authority and Actor machinery unchanged.

- Inventory SHA256: b2325fb93db5dfa16254a7d3d6c06b82a8b628bb61f4374276be6fe3e76e8aef
- Notebook SHA256: 87afeeb324d33bafe4a7992abf1f5f17ba9ef83631d716b3c5287a061fa770a3
- Packaged card SHA256: 6425dc576370b0f68ea884ff5ea11cbf6989aa80bc08beaf0c86d3c8e4c72bc9
- Source diff SHA256: 89fe18a9b5fd3e0a7cdecf25d7f59af17138f9661810fe874f009bc5054708cd

`frozen_client.py`: pending presence uses the exact action/step key, so admitted JSON null is ordinary data. Duplicate query and mismatched action/step assimilation still refuse; an unexecuted prediction can be cancelled only at its original step. Regression constructs/admits/seals/freezes a null-state candidate, rejects an unexecuted submit, refuses duplicate/mismatched operations, cancels its exact pending binding, executes CLICK, assimilates null and performs the next DOWN.

`audit.py` and `gameplay_play.py`: authoritative freeze requires digest-valid session-root FROZEN plus parent CONSTRUCTION and RESULT statuses CANDIDATE_READY/frozen. Arm audit considers only session roots with IPC, never arbitrary descendants. Session summary uses the same predicate; child artifacts are separately listed and retained for exact collection. Regression runs the real persist job through successful child write, then hard-kills it before parent promotion: no root freeze, audit freeze absent, session frozen false, child evidence retained.

Package bytes are now stable for review. Subsequent work in this task is gate tooling/evidence only. No package approval authored, no manifest freeze, no authority consumption or remote call. Independent re-review remains required.
