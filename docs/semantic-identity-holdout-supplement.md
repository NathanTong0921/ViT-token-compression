# Supplement: class-heldout task coverage

The fixed class partition leaves only 2 original validation pairs with two new classes in Pets and 10 in Flowers. Before inspecting class-heldout outcomes, add a supplementary panel to improve class-pair coverage without changing seeds, mappings, models, images or training.

Use the original validation tasks plus original training-manifest pairs containing at least one held class. The latter were all removed before fitting the class-heldout models, so they remain unseen combinations for these models. Known/known tasks stay exactly the original validation tasks. Images remain the original validation images. Do not read official test images or reserved final-task manifests. This expanded panel is evaluated only for class-heldout models, because the all-class models were trained on the additional pairs.

Keep the original panel and its results alongside the supplement. Report pair counts by origin (original validation versus removed training tasks), absolute full-CLIP performance and margins, real-text versus both fixed identity mappings, and the same wide baseline. The supplement changes coverage, not the selected checkpoint or hyperparameters.

For both panels, additionally divide mixed known/new tasks by the true image class: known versus held. Both sides retain the same candidate pairs. This distinguishes transferring useful evidence to an unseen object from merely rejecting an unseen distractor. Each task has equal weight within either side; aggregate mixed-task results remain available. The accompanying interaction is the real-minus-permuted effect for true-new images minus that for true-known images, with image-cluster uncertainty. It is supplementary analysis, not a replacement primary endpoint.

The GPU evaluator loads the twelve fixed class-heldout checkpoints per dataset and reuses the existing suffix evaluation path. No teacher arrays or retraining are needed. It saves the combined task manifest, source labels, per-example features/selections/outcomes, standard summary, and class-side details.

```bash
# Run once the class_holdout checkpoints have all finished.
PYTHONPATH=src python scripts/evaluate_semantic_identity_holdout.py \
  --models runs/pets-semantic-identity-v1 --device cuda:2 \
  --output runs/pets-semantic-identity-holdout-expanded-v1

PYTHONPATH=src python scripts/evaluate_semantic_identity_holdout.py \
  --models runs/flowers-semantic-identity-v1 --device cuda:6 \
  --output runs/flowers-semantic-identity-holdout-expanded-v1

# CPU-only mixed-task analysis of the retained original panel.
PYTHONPATH=src python scripts/analyze_semantic_identity_holdout.py \
  --source runs/pets-semantic-identity-v1/class_holdout \
  --protocol runs/pets-semantic-identity-v1/protocol.json \
  --output runs/pets-semantic-identity-v1/class_holdout/class_novelty_details.json

PYTHONPATH=src python scripts/analyze_semantic_identity_holdout.py \
  --source runs/flowers-semantic-identity-v1/class_holdout \
  --protocol runs/flowers-semantic-identity-v1/protocol.json \
  --output runs/flowers-semantic-identity-v1/class_holdout/class_novelty_details.json

PYTHONPATH=src python -m unittest discover -s tests -p test_semantic_identity_holdout.py
```
