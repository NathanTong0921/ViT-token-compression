# Pets v1 manifests

Generated on 2026-10-16 from Oxford's original `trainval.txt` and `test.txt`, using `scripts/prepare_pets.py`. Image seed: 2026; pair seed: 2027; validation fraction: 0.2. The separate balancing seed is 2028, recorded in the baseline configuration.

| Axis | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Images | 2,944 | 736 | 3,669 |
| Unordered class pairs | 532 | 66 | 68 |
| Classes appearing in those pairs | 37 | 36 | 36 |
| Classes with at least two pairs | 37 | 31 | 32 |
| Pairs of tasks sharing an anchor class | 14,861 | 217 | 245 |

The 7,349 annotated images span 37 breeds (12 cat breeds and 25 dog breeds). Official test membership is unchanged. Raw image archives contain additional files; only image IDs listed here enter experiments.

The task split is a fixed random split of pairs. Validation pairs omit `american bulldog`; test pairs omit `sphynx`. The test split also gives only one pair each to `abyssinian`, `great pyrenees`, `newfoundland`, and `pomeranian`. Thus 32 test anchor classes can support same-image comparisons between two held-out tasks before any difficulty matching. This is a property of the pair split, not an exclusion from the full 37-class image evaluation.

Files:

- `images.csv`: image ID, zero-based breed ID, breed name, species, image split, original split.
- `classes.json`: the ID-to-name mapping, ordered by the official annotation IDs.
- `tasks.csv`: unordered pair ID, its two class IDs, and task split.
- `protocol_summary.json`: seeds, counts, class coverage and pair degrees.

See [the full protocol](../../../docs/pets-protocol.md). Dataset attribution: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), Parkhi et al., *Cats and Dogs*, CVPR 2012; dataset license CC BY-SA 4.0. Image/model archives are excluded from Git.
