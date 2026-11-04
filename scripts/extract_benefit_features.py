"""Read existing selector archives and build pre-pruning benefit predictors.

Feature construction receives only text separation and selector logits. Labels,
full-model scores and pruned CLIP scores enter separate outcome arrays only.
"""

from collections import Counter
import csv
import io
import json
import zipfile

import numpy as np
from scipy.stats import rankdata


FEATURE_NAMES = (
    "text_distance", "inverse_text_distance", "shared_boundary_gap",
    "selection_disagreement", "conditional_boundary_gap", "rank_disagreement",
    "disagreement_x_shared_boundary",
)
MIX_FEATURE_NAMES = FEATURE_NAMES + ("donor_disagreement_excess",)
ABSOLUTE_OUTCOME_NAMES = ("accuracy", "geom_margin", "raw_score_gap")
GAIN_NAMES = ("geom_margin_gain", "raw_score_gap_gain")


def selected_positions(logits, k):
    return np.sort(np.argsort(-logits, axis=1, kind="stable")[:, :k], axis=1)


def disagreement(left, right):
    """Fraction of one equal-sized mask not shared by the other mask."""
    return 1.0 - (left[:, :, None] == right[:, None, :]).any(axis=-1).mean(axis=-1)


def boundary_gap(logits, k):
    ordered = np.sort(logits, axis=1)
    std = logits.std(axis=1)
    assert np.all(std > 0), "Normalized boundary is undefined for constant logits"
    return (ordered[:, -k] - ordered[:, -k - 1]) / std


def feature_matrix(conditioned, shared, text_distance, k, donor=None):
    """Return the fixed seven/eight scale-invariant, label-free features."""
    c, u = np.asarray(conditioned, float), np.asarray(shared, float)
    distance = np.asarray(text_distance, float)
    assert c.shape == u.shape and distance.shape == (len(c),)
    assert np.all(distance > 0)
    cmask, umask = selected_positions(c, k), selected_positions(u, k)
    changed = disagreement(cmask, umask)
    ugap, cgap = boundary_gap(u, k), boundary_gap(c, k)
    cranks, uranks = rankdata(c, axis=1), rankdata(u, axis=1)
    cranks -= cranks.mean(axis=1, keepdims=True)
    uranks -= uranks.mean(axis=1, keepdims=True)
    rho = (cranks * uranks).sum(axis=1) / np.sqrt(
        (cranks * cranks).sum(axis=1) * (uranks * uranks).sum(axis=1))
    columns = [distance, 1. / distance, ugap, changed, cgap, 1. - rho, changed * ugap]
    if donor is not None:
        donor_mask = selected_positions(np.asarray(donor, float), k)
        columns.append(disagreement(cmask, donor_mask) - disagreement(umask, donor_mask))
    return np.column_stack(columns)


def absolute_outcomes(scores, labels, pairs, text_distance):
    """Absolute binary accuracy, signed geometric margin and signed raw gap."""
    scores = np.asarray(scores, float)
    others = np.where(labels == pairs[:, 0], pairs[:, 1], pairs[:, 0])
    row = np.arange(len(labels))
    own, other = scores[row, labels], scores[row, others]
    gap = own - other
    correct = (own > other) | ((own == other) & (labels < others))
    return np.column_stack((correct.astype(float), gap / text_distance, gap))


def _rows(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode())))


def _arrays(archive, name, keys):
    # Nested ZIP reads avoid extracting or retaining a second copy of raw runs.
    with archive.open(name) as member, np.load(member, allow_pickle=False) as values:
        return {key: values[key].copy() for key in keys}


def _panel(x, alternative, base, run_ids, feature_names):
    gain = alternative - base
    return dict(X=np.stack(x), outcomes=gain[:, :, 1:], accuracy_gain=gain[:, :, 0],
                base_outcomes=base, alternative_outcomes=alternative,
                feature_names=list(feature_names), run_ids=np.asarray(run_ids), routing=True)


def load_setting(core_zip, hybrid_zip, setting):
    """Return aligned predictor and outcome panels for one existing setting.

    Shapes: X [runs, rows, features], outcomes [runs, rows, (geom, raw)],
    absolute outcomes [runs, rows, (accuracy, geom, raw)]. Primary uses three
    training seeds; mix/increment use all six ordered distinct-seed pairs.
    Increment is a difference-in-differences target, so has no routing endpoints.
    """
    cp, hp = f"runs/core-protection-v1/{setting}/", f"runs/hybrid-specificity-v1/{setting}/"
    with zipfile.ZipFile(core_zip) as core, zipfile.ZipFile(hybrid_zip) as hybrid:
        cprotocol = json.loads(core.read(cp + "summary.json"))
        hprotocol = json.loads(hybrid.read(hp + "protocol.json"))
        seeds = cprotocol["training_seeds"]
        assert seeds == hprotocol["training_seeds"]
        k = cprotocol["config"]["keep_patch_tokens"]
        assert k == hprotocol["config"]["keep_patch_tokens"]
        rows, hrows = _rows(core, cp + "examples.csv"), _rows(hybrid, hp + "examples.csv")
        keys = [(r["image_id"], r["task_id"]) for r in rows]
        hlookup = {(r["image_id"], r["task_id"]): i for i, r in enumerate(hrows)}
        assert len(keys) == len(set(keys)) == len(hlookup) and set(keys) == set(hlookup)
        order = np.asarray([hlookup[key] for key in keys])
        for row, hi in zip(rows, order):
            assert all(row[key] == hrows[hi][key] for key in ("image_index", "class_id", "class_a", "class_b"))
        evidence = _arrays(core, cp + "evidence.npz", ("image_ids", "labels", "text_features", "full_scores"))
        hevidence = _arrays(hybrid, hp + "evidence.npz", ("image_ids", "text_features"))
        for key in ("image_ids", "text_features"):
            np.testing.assert_array_equal(evidence[key], hevidence[key])
        ids, tasks = np.asarray([r["image_id"] for r in rows]), np.asarray([r["task_id"] for r in rows])
        ix = np.asarray([int(r["image_index"]) for r in rows])
        labels = np.asarray([int(r["class_id"]) for r in rows])
        pairs = np.asarray([[int(r["class_a"]), int(r["class_b"])] for r in rows])
        np.testing.assert_array_equal(ids, evidence["image_ids"][ix])
        np.testing.assert_array_equal(labels, evidence["labels"][ix])
        weights = np.asarray([float(r["task_class_weight"]) for r in rows])
        counts = Counter((r["task_id"], r["class_id"]) for r in rows)
        expected = [1. / (2 * len(set(tasks)) * counts[(r["task_id"], r["class_id"])]) for r in rows]
        np.testing.assert_allclose(weights, expected, atol=1e-12, rtol=0)
        text = evidence["text_features"].astype(float)
        distance = np.linalg.norm(text[pairs[:, 0]] - text[pairs[:, 1]], axis=1)
        full = absolute_outcomes(evidence["full_scores"][ix], labels, pairs, distance)
        source, features, c_out, u_out = {}, [], [], []
        for seed in seeds:
            a = _arrays(core, cp + f"seed_{seed}.npz",
                        ("variants", "conditioned_logits", "wide_logits", "patch_indices", "cosine_scores"))
            ci, ui = [a["variants"].tolist().index(method) for method in ("conditioned", "wide")]
            c, u = a["conditioned_logits"].astype(float), a["wide_logits"][ix].astype(float)
            for logits, mi in ((c, ci), (u, ui)):
                np.testing.assert_array_equal(selected_positions(logits, k), a["patch_indices"][mi])
            cscore, uscore = a["cosine_scores"][ci], a["cosine_scores"][ui]
            source[seed] = dict(c=c, u=u, cscore=cscore, uscore=uscore)
            features.append(feature_matrix(c, u, distance, k))
            c_out.append(absolute_outcomes(cscore, labels, pairs, distance))
            u_out.append(absolute_outcomes(uscore, labels, pairs, distance))
        c_out, u_out = np.stack(c_out), np.stack(u_out)
        primary = _panel(features, c_out, u_out, seeds, FEATURE_NAMES)

        mix_x, cu_out, uu_out, c_recipient, u_recipient = [], [], [], [], []
        run_pairs = hprotocol["primary_seed_pairs"]
        for s, t in run_pairs:
            a = _arrays(hybrid, hp + f"pair_{s}_{t}.npz", ("methods", "cosine_scores"))
            methods = a["methods"].tolist()
            scores = a["cosine_scores"][:, order]
            for name, expected_score in (("conditioned", source[s]["cscore"]),
                                          ("shared_recipient", source[s]["uscore"]),
                                          ("shared_donor", source[t]["uscore"])):
                np.testing.assert_array_equal(scores[methods.index(name)], expected_score)
            mix_x.append(feature_matrix(source[s]["c"], source[s]["u"], distance, k, source[t]["u"]))
            cu_out.append(absolute_outcomes(scores[methods.index("cu_half")], labels, pairs, distance))
            uu_out.append(absolute_outcomes(scores[methods.index("uu_half")], labels, pairs, distance))
            c_recipient.append(c_out[seeds.index(s)])
            u_recipient.append(u_out[seeds.index(s)])
        cu_out, uu_out = np.stack(cu_out), np.stack(uu_out)
        mix = _panel(mix_x, cu_out, uu_out, run_pairs, MIX_FEATURE_NAMES)
        increment = (cu_out - np.stack(c_recipient)) - (uu_out - np.stack(u_recipient))
        increment_panel = dict(X=mix["X"].copy(), outcomes=increment[:, :, 1:],
                               accuracy_gain=increment[:, :, 0], base_outcomes=None,
                               alternative_outcomes=None, feature_names=list(MIX_FEATURE_NAMES),
                               run_ids=np.asarray(run_pairs), routing=False)
    return dict(setting=setting, rows=rows, weights=weights, ids=ids, tasks=tasks,
                full_signed_margin=full[:, 1], full_raw_gap=full[:, 2],
                full_outcomes=full, text_distance=distance,
                panels=dict(primary=primary, mix=mix, increment=increment_panel),
                absolute_outcome_names=list(ABSOLUTE_OUTCOME_NAMES), gain_names=list(GAIN_NAMES))
