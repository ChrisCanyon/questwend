# Repository workflow

Keep `main` and release branches limited to reviewed, verified changes.

- Before editing runtime code, check the current branch and `git status`. Start
  each experiment on a named `experiment/*` branch in a separate worktree from
  a recorded base commit. Do not use a shared source checkout as a scratchpad.
- If a checkout already contains uncommitted work, preserve it on an experiment
  branch and commit a clearly labeled WIP snapshot before cleaning or starting
  another experiment. Never discard another experiment's changes to make a
  checkout look clean.
- Commit small, coherent steps. Record the source commit, build recipe, runtime
  and model versions, workload, settings, and raw result location. Keep local
  paths, device identifiers, credentials, model weights, and operational logs
  out of this public repository.
- Build and benchmark candidates in isolated worktrees and runtime directories.
  Keep the published executable and saved model profiles unchanged during an
  experiment. Compare against the same pinned baseline with correctness checks
  and matched settings; include startup, prefill, and generation separately.
- Promote only the verified changes needed for the result. Review the complete
  diff, run focused tests and a matched workload, check for regressions outside
  the local use case, and document the evidence before merging or cherry-picking
  into `main` or a release branch. Leave failed or inconclusive work on its
  experiment branch.
- After promotion, confirm the destination branch and worktree are clean and a
  rollback commit or runtime is available.
