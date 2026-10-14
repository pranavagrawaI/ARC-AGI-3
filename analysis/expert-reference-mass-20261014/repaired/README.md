# Unlaunched repair

The routing math is unchanged. Each gameplay arm now runs in a fresh Python subprocess and rejects an already-initialized gate before the first game. This separates all harness globals, rather than resetting only the known clock. Parent containment owns both the model server and the gameplay process group. GPU test stdout/stderr and a graph-execution marker are saved explicitly.

Prepared after the frozen v1 run exposed process-global gate reuse and missing successful-test stderr. CPU process-isolation fixture passes; notebook schema/embedded-source compilation and incumbent gates pass. Full Kaggle runtime remains untested. No launch script or new GPU authority is attached to this repair. A new sealed card and separately authorized budget are needed for any rerun.
