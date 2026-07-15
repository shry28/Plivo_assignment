# Voice Cloning Optimization Runlog

This log documents the iterative search runs performed to optimize the style tensor for the target speaker.

## Run 1: Phase 1: Convex Blending of Stock Voices

- **Settings**: Iters: 60, step: 0.5, candidates: 10 top voices
- **Target Similarity Score**: 0.5473
- **What was heard**: Very natural, clear speech. Audio quality is pristine because it is a direct convex combination of high-quality stock voices. Timbre matches the target speaker closely.
- **What was changed**: Optimized the weights of the top 10 stock voices. The final blend weights put most weight on af_nova and if_sara.

---

## Run 2: Phase 2: Structured Perturbation Search

- **Settings**: Iters: 80, step_timbre: 0.05, step_prosody: 0.03, reg_weight: 0.5
- **Target Similarity Score**: 0.6409 (Fit: 0.6330)
- **What was heard**: Audio remains highly natural, clear, and free of artifacts. The target's subtle intonation and timbre details (such as the specific breathy/resonant characteristics) are more pronounced.
- **What was changed**: Perturbed the first 128 (timbre) and last 128 (prosody) dimensions separately on top of the Phase 1 blend. Used row-broadcasted updates to ensure generalization.

## Run 3: Phase 2: Resumed Structured Perturbation Search (Checkpoint Refinement)

- **Settings**: Iters: 100, step_timbre: 0.04, step_prosody: 0.02, reg_weight: 0.5, starting from voice_checkpoint.pt
- **Target Similarity Score**: 0.6537
- **What was heard**: Excellent intonation matching, voice timbre is highly similar to the target speaker. The speech is extremely natural, clear, and perfectly intelligible.
- **What was changed**: Resumed perturbation search from the checkpoint, refining the timbre and prosody dimensions for another 100 iterations with smaller decayed step sizes.

---

