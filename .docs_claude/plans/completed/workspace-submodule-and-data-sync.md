# UFM-train: join refseg-workspace, fine-tune on shelf-optflow

## Goal

Make UFM-train fine-tunable on any machine: add it to `refseg-workspace` as a pinned
submodule, pull the `shelf-optflow` umbrella via the artifact registry, and replace the
hardcoded cross-repo dataset path. Fine-tune from public `infinity1096/UFM-Base`
(auto-downloads) — **no checkpoint to sync**.

Companion: `isaac_datagen/.docs_claude/plans/active/artifact-registry.md` (the `art` tool +
the `jeffke613/refseg-datasets` repo this pulls from). Data sync here is just an
`.artifacts.yaml` + `dspull shelf-optflow`.

## Settled facts / decisions

- **Fine-tune only.** `resume_model=infinity1096/UFM-Base` (public, auto-download). Traced
  `train_pl.py:483-516`: build fresh → `load_state_dict(strict=True)`, so the encoder comes
  from the checkpoint; `uniception_pth_root` not needed for fine-tuning. The smoke
  `outputs/.../last.ckpt` are not shipped.
- **Training data = `shelf-optflow`** (4.7 G, 3 renders), a *combined* render that also feeds
  the verifier/segmenter; lives in the shared `/data` tree → `jeffke613/refseg-datasets`.
- UFM-train depends on `vision_core` (`../vision_core` editable, `pyproject.toml:108`),
  resolving to the sibling submodule inside the workspace.
- **Remote = fork to `jeffrey-ke/UFM`** (settled): `train` is 2 ahead of `origin/train`
  (`UniFlowMatch/UFM`) → pin not reachable yet.

## Phase A — portable optflow data path + registry config

`configs/dataset/optflow_isaac/train/default.yaml` (and `val/default.yaml`):

| field | before | after |
|---|---|---|
| `dataset_dir` | `/home/jeffk/repo/isaac_datagen/src/isaac_datagen/datasets/shelf-optflow` | `${machine.root_data_dir}/shelf-optflow` |

`configs/machine/refseg.yaml` (new — launch `machine=refseg`):

```yaml
defaults:
  - default
# Hydra runs chdir=True (cwd becomes outputs/<run>/), so a bare relative path would resolve
# against the run dir. ${hydra:runtime.cwd} is the launch dir (repo root) — absolute & chdir-proof.
root_data_dir: ${hydra:runtime.cwd}/datasets            # dspull lands shelf-optflow here
uniception_pth_root: ${hydra:runtime.cwd}/checkpoints   # unused for fine-tuning; non-MISSING so Hydra resolves
tartanair_root_data_dir: ${hydra:runtime.cwd}/datasets  # unused for optflow fine-tuning
```

`UFM-train/.artifacts.yaml`:

```yaml
dataset: { repo: jeffke613/refseg-datasets, dir: datasets, require_name: true }
```

`.gitignore`: add `/datasets` and `/outputs` (Hydra run dirs). On *this* machine, optionally
`ln -s /data/user/jeffk/datasets datasets` to skip a 4.7 G re-pull; fresh machine = real dir
+ `dspull shelf-optflow`.

## Phase B — fork, push, submodule

```bash
cd ~/repo/UFM-train
gh repo fork UniFlowMatch/UFM --clone=false          # -> jeffrey-ke/UFM  (no --remote w/ a repo arg)
git remote add fork git@github.com:jeffrey-ke/UFM.git
# Commit the Phase-A edits FIRST (optflow integration + registry config, minus outputs/) so the
# pin is functional, THEN push — the workspace pins this commit, not the pre-edit b902444:
git add -A && git commit -m "Optflow fine-tuning integration + artifact registry config"   # -> 658fee4
git push fork train

cd ~/repo/refseg-workspace
git submodule add -b train git@github.com:jeffrey-ke/UFM.git UFM-train   # pins 658fee4
git -C UFM-train submodule update --init --recursive                     # UniCeption (castacks, public)
#   ../vision_core resolves to refseg-workspace/vision_core (already a submodule)
git add -A && git commit -m "Add UFM-train submodule (optflow fine-tune)" && git push
```

`README.md` UFM bringup (own venv — never unify):

```bash
cd refseg-workspace/UFM-train
uv sync
dspull shelf-optflow                                 # -> datasets/shelf-optflow
env -u PYTHONPATH uv run python scripts/train.py \
    machine=refseg dataset=optflow \
    resume_model=infinity1096/UFM-Base               # auto-downloads from public HF
```

## Verification

- Fresh workspace clone: `cd UFM-train && uv sync` (resolves `../vision_core` from sibling
  submodule); `dspull shelf-optflow`; short `machine=refseg dataset=optflow
  resume_model=infinity1096/UFM-Base` launch pulls UFM-Base, reads the dataset, steps the loss.

## Decision log

- `root_data_dir = datasets` (in-repo, scoped pull → self-contained) vs sibling
  `../isaac_datagen/datasets`. Chose in-repo.
- Checkpoint sync: none (public UFM-Base/UFM-Refine).
