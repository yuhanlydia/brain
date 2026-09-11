# Real-backend preparation follow-up

This updates the readiness picture after the original synthetic result bundle. It does not change the archived synthetic measurements or claim a completed P1/P2/P3/P4 run.

## Executed checks

- Downloaded the subject-01 VINDEX CLIP-224 encoder checkpoint, its configuration, and LLaVA-7B projector from pinned public Hugging Face revision `ae1c5993cee072950869486391c3d2603fd8fcd8`. The three downloaded files passed Hub cache verification.
- VINDEX source at `ad0dab49e99d18a179097e9b79246f7ca3559f1b` imports `BrainXS` in its entrypoints but does not define that class in `src/model/model.py`. The upstream UMBRAE implementation at `8c1795c434a41cdd38ce6628b7e1a774f31d9483` supplies it. All 78 checkpoint tensors loaded with `strict=True` into this implementation.
- Ran the loaded encoder on eight real brain arrays from the first subject-01 training shard of `pscotti/naturalscenesdataset`, revision `421fb80ea4ca5600c7364ed400f9a5941380c387`. Each call used repeat-axis index 0, without averaging. All outputs were finite and had shape `[1, 256, 1024]`. Peak allocated CUDA memory was 471,245,824 bytes. `real_encoder_probe.json` records per-sample checks.
- A zero-input forward produced different embeddings, recorded as an implementation diagnostic. This is not task accuracy, NDG, or evidence that the method meets its scientific criterion.
- Inspected a MindEye2 metadata shard at revision `26421f100e4c6012a35ecadb272a0ec1d999202d`: its `behav.npy` has 17 columns including image index, subject, session, run, trial and global-trial index. This supplies a candidate source for the required single-trial provenance; joining it to beta arrays and validating the split still remain.

## Limits and next work

The encoder probe had zero optimizer steps and no loaded LLaVA language model. It is a preparatory component check, not the complete P1 real-shape optimization smoke. The original grouped NSD shard has three brain rows but one trial ID per sample, so its repeat-to-global-trial association has not been established.

The public VINDEX checkpoint inventory lists subject-01 encoders only. The inspected BrainJanus checkout (`6d797688c26c0f2ac2ff2be21f4598fe2b1ed946`) contains `src/train_final.py`, but not its imported `args.py`, `data/dataset.py`, `models/brain_omni.py`, or `models/utils.py`. A complete BrainJanus source/checkpoint is still needed for that extension.

The exact source inventories and revisions are included. LLaVA-7B weights and subject-01 single-trial beta downloads were started separately; this report does not claim those downloads or full model integration are complete.

## Sources

- [VINDEX](https://github.com/weihaox/VINDEX)
- [UMBRAE](https://github.com/weihaox/UMBRAE)
- [MindEye2 metadata format](https://github.com/MedARC-AI/MindEyeV2#faq)
- [BrainJanus](https://github.com/HaitaoWuTJU/BrainJanus)
