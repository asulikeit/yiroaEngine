#!/usr/bin/env bash
# Runs the loom models (spec 19.2: "Every commit"). Loom replaces the atomics of jamsil_vfs::sync,
# so the models are built with `--cfg loom` into their own target directory, and only the loom test
# targets are run: other tests would panic on loom atomics outside a model.
#   Model 5: the frame's dirty flag against the checkpoint's redo point (jamsil-engine)
#   Model 3: a snapshot reader against the writers of the published state (jamsil-txn)
#   Model 4: lock grant and wake-up: release, cancel and the gate wait (jamsil-txn)
set -euo pipefail
cd "$(dirname "$0")"
export RUSTFLAGS="${RUSTFLAGS:-} --cfg loom"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/loom}"
# Loom explores every interleaving, which grows very fast. Bounding the preemptions of one execution
# is the usual way to keep a model fast enough to run on every commit; two preemptions find the
# races of these models (the controls below fail with it). Raise it, or unset it, for a deeper run:
#   LOOM_MAX_PREEMPTIONS=5 ./loom.sh
# The schedule (spec 19.2): bound two on every commit (no option), bound three every night
# (`./loom.sh --nightly`), and no bound weekly and before each milestone gate (`./loom.sh --full`; the
# unbounded run of model 3 takes about 38 minutes, model 4 under a minute, model 5 almost nothing).
# `LOOM_NO_BOUND=1 ./loom.sh` still does the same as `--full`.
if [ "${1:-}" = "--nightly" ]; then
  shift
  export LOOM_MAX_PREEMPTIONS="${LOOM_MAX_PREEMPTIONS_NIGHTLY:-3}"
elif [ "${1:-}" = "--full" ]; then
  shift
  export LOOM_NO_BOUND=1
fi
if [ "${LOOM_NO_BOUND:-}" = "1" ]; then
  unset LOOM_MAX_PREEMPTIONS
else
  export LOOM_MAX_PREEMPTIONS="${LOOM_MAX_PREEMPTIONS:-2}"
fi
cargo test --release -p jamsil-engine --test loom_dirty_redo "$@"
cargo test --release -p jamsil-txn --test loom_published "$@"
cargo test --release -p jamsil-txn --test loom_lock "$@"
