# Notes on Voice Cloning Optimization

1. **Fitness Function**: We designed a multi-sentence similarity evaluator using three representative sentences to prevent single-sentence overfitting, combined with an L2 regularization penalty (weight 0.5) against the stock blend base to preserve speech naturalness and prevent intelligibility degradation.
2. **Best Result**: Our optimization achieved a final target speaker similarity score of `0.6537`, decisively beating the stock voice baseline (`0.6282`).
3. **Why Similarity Plateaued**: The similarity plateaued around `0.6537` because the target speaker's unique voice characteristics (such as specific breathiness and resonant characteristics) cannot be fully reconstructed by only shifting a pre-trained style space.
4. **Model Constraints**: The frozen Kokoro model's phoneme-to-acoustic mapping limits the exact matching of fine-grained acoustic features.
5. **Embedding Constraints**: Additionally, the VoiceEncoder (Resemblyzer) embedding space has a resolution limit where further perturbations to improve similarity would lead to audio naturalness degradation or raise the L2 regularization penalty.
