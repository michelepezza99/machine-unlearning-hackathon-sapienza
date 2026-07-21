"""Focused synthetic tests for progressive search configuration generation."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from machine_unlearning.progressive import (
    deduplicate_candidates,
    generate_stage2_config,
    generate_stage3_config,
    generate_stage4_config,
    select_stage2_families,
    select_stage3_structures,
    select_stage4_finalists,
    semantic_candidate_fingerprint,
)
from machine_unlearning.search import (
    build_effective_search_config,
    merge_candidate_configs,
    validate_search_config,
)
from machine_unlearning.unlearning import progressive_search
from scripts import generate_progressive_configs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CONFIG_PATH = REPOSITORY_ROOT / "configs/search_stage1_coarse.json"


def _stage1_config() -> dict[str, Any]:
    payload = json.loads(STAGE1_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _successful_candidate_result(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": deepcopy(config),
        "metrics": {
            "precision_at_10": 0.8,
            "validation_bce": 0.2,
            "forget_bce": 0.3,
            "local_privacy_proxy": 0.75,
            "execution_time_seconds": 1.0,
            "local_search_score": 0.8,
            "best_epoch": 1,
            "utility_floor_pass": True,
        },
        "mask_metadata": {
            "selected": 1,
            "selected_fraction_of_eligible": 0.01,
        },
        "dampening_metadata": {},
        "repair_history": pd.DataFrame(),
        "gradient_ascent_history": pd.DataFrame(),
    }


def _evidence_row(
    name: str,
    *,
    top_fraction: float,
    dampening: float,
    score: float,
    privacy: float,
    precision: float,
    execution_time: float,
    quantile: float = 0.5,
    batchnorm: bool = False,
    valid: object = True,
    utility_floor_pass: object = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "valid": valid,
        "utility_floor_pass": utility_floor_pass,
        "top_fraction": top_fraction,
        "minimum_dampening_factor": dampening,
        "forget_absolute_quantile": quantile,
        "recalibrate_batchnorm": batchnorm,
        "local_search_score": score,
        "local_privacy_proxy": privacy,
        "precision_at_10": precision,
        "execution_time_seconds": execution_time,
    }


def test_stage1_config_validates_and_has_exact_budget() -> None:
    config = _stage1_config()
    validate_search_config(config)

    names = [candidate["name"] for candidate in config["candidates"]]
    assert names == [
        "tf_0p25_damp_0p95",
        "tf_0p25_damp_0p90",
        "tf_0p5_damp_0p95",
        "tf_0p5_damp_0p90",
        "tf_0p5_damp_0p85",
        "tf_1_damp_0p95",
        "tf_1_damp_0p90",
        "tf_1_damp_0p85",
        "tf_1_damp_0p82",
        "tf_2_damp_0p90",
        "tf_2_damp_0p85",
        "tf_2_damp_0p82",
    ]
    assert len(names) == 12
    assert len(set(names)) == 12
    assert config["add_gradient_ascent_variants"] == 4
    assert config["quick"]["candidate_count"] == 4


def test_full_stage1_progressive_search_expands_to_sixteen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _stage1_config()

    def fake_execute(candidate: dict[str, Any], **_: Any) -> dict[str, Any]:
        return _successful_candidate_result(candidate)

    monkeypatch.setattr(
        "machine_unlearning.unlearning.execute_search_candidate", fake_execute
    )
    best, comparison, results = progressive_search(
        merge_candidate_configs(config),
        execute_kwargs={"seed": 92},
        baseline_precision_at_10=0.8,
        utility_floor_ratio=float(config["utility_floor_ratio"]),
        add_gradient_ascent_variants=int(config["add_gradient_ascent_variants"]),
    )

    assert best is not None
    assert len(results) == 16
    assert len(comparison) == 16
    assert comparison["name"].nunique() == 16
    original_names = {candidate["name"] for candidate in config["candidates"]}
    generated_names = {
        str(result["config"]["name"])
        for result in results
        if str(result["config"]["name"]) not in original_names
    }
    assert len(generated_names) == 4


def test_quick_stage1_keeps_four_candidates_and_disables_ga() -> None:
    config = _stage1_config()
    effective = build_effective_search_config(
        config,
        quick=True,
        max_candidates=None,
    )

    assert len(effective["candidates"]) == 4
    assert effective["add_gradient_ascent_variants"] == 0
    assert [candidate["name"] for candidate in effective["candidates"]] == [
        candidate["name"] for candidate in config["candidates"][:4]
    ]


def _stage1_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _evidence_row(
                "family_a",
                top_fraction=0.0025,
                dampening=0.95,
                score=0.90,
                privacy=0.88,
                precision=0.81,
                execution_time=10.0,
            ),
            _evidence_row(
                "family_a_ga",
                top_fraction=0.0025,
                dampening=0.95,
                score=0.99,
                privacy=0.90,
                precision=0.80,
                execution_time=11.0,
                valid="true",
                utility_floor_pass="true",
            ),
            _evidence_row(
                "family_a_ga2",
                top_fraction=0.0025,
                dampening=0.95,
                score=0.98,
                privacy=0.89,
                precision=0.80,
                execution_time=12.0,
            ),
            _evidence_row(
                "family_b",
                top_fraction=0.005,
                dampening=0.90,
                score=0.89,
                privacy=0.87,
                precision=0.82,
                execution_time=9.0,
            ),
            _evidence_row(
                "family_c",
                top_fraction=0.01,
                dampening=0.85,
                score=0.88,
                privacy=0.86,
                precision=0.83,
                execution_time=8.0,
            ),
            _evidence_row(
                "invalid_family",
                top_fraction=0.02,
                dampening=0.82,
                score=1.0,
                privacy=1.0,
                precision=1.0,
                execution_time=1.0,
                valid=False,
                utility_floor_pass=False,
            ),
        ]
    )


def test_stage2_uses_distinct_families_and_generates_deterministic_names() -> None:
    evidence = _stage1_evidence()
    families = select_stage2_families(evidence)
    family_keys = {
        (family["top_fraction"], family["minimum_dampening_factor"])
        for family in families
    }
    assert len(families) == 3
    assert len(family_keys) == 3
    assert families[0]["representative_configuration"] == "family_a_ga"

    first = generate_stage2_config(_stage1_config(), evidence)
    second = generate_stage2_config(_stage1_config(), evidence)
    validate_search_config(first)
    names = [candidate["name"] for candidate in first["candidates"]]
    assert names == [candidate["name"] for candidate in second["candidates"]]
    assert names == [
        f"{family}_{suffix}"
        for family in ("tf_0p25_d095", "tf_0p5_d090", "tf_1_d085")
        for suffix in (
            "q040_no_bn",
            "q040_with_bn",
            "q050_no_bn",
            "q050_with_bn",
            "q060_no_bn",
            "q060_with_bn",
        )
    ]
    assert len(names) == len(set(names)) == 18
    assert first["add_gradient_ascent_variants"] == 6


def _stage2_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _evidence_row(
                "structure_a",
                top_fraction=0.0025,
                dampening=0.95,
                quantile=0.4,
                batchnorm=False,
                score=0.90,
                privacy=0.96,
                precision=0.81,
                execution_time=10.0,
            ),
            _evidence_row(
                "structure_a_ga",
                top_fraction=0.0025,
                dampening=0.95,
                quantile=0.4,
                batchnorm=False,
                score=0.99,
                privacy=0.95,
                precision=0.82,
                execution_time=11.0,
            ),
            _evidence_row(
                "structure_b",
                top_fraction=0.005,
                dampening=0.90,
                quantile=0.5,
                batchnorm=True,
                score=0.89,
                privacy=0.94,
                precision=0.83,
                execution_time=9.0,
            ),
            _evidence_row(
                "structure_c",
                top_fraction=0.01,
                dampening=0.85,
                quantile=0.6,
                batchnorm=False,
                score=0.88,
                privacy=0.80,
                precision=0.84,
                execution_time=8.0,
            ),
        ]
    )


def test_stage3_selects_two_structures_and_builds_twenty_unique_configs() -> None:
    stage2 = generate_stage2_config(_stage1_config(), _stage1_evidence())
    evidence = _stage2_evidence()
    structures = select_stage3_structures(evidence)
    structure_keys = {
        (
            structure["top_fraction"],
            structure["minimum_dampening_factor"],
            structure["forget_absolute_quantile"],
            structure["recalibrate_batchnorm"],
        )
        for structure in structures
    }
    assert len(structures) == len(structure_keys) == 2

    generated = generate_stage3_config(stage2, evidence)
    validate_search_config(generated)
    assert generated["add_gradient_ascent_variants"] == 0
    assert len(generated["candidates"]) == 20
    names = [candidate["name"] for candidate in generated["candidates"]]
    fingerprints = {
        semantic_candidate_fingerprint(generated["common_candidate"], candidate)
        for candidate in generated["candidates"]
    }
    assert len(names) == len(set(names)) == 20
    assert len(fingerprints) == 20


def test_candidate_deduplication_uses_effective_semantics() -> None:
    common = {
        "repair_learning_rate": 1e-4,
        "repair_batch_size": 1024,
        "selected_parameter_weight": 1,
        "gradient_ascent_steps": 0,
        "gradient_ascent_learning_rate": 1e-6,
        "gradient_ascent_batch_size": 128,
        "gradient_ascent_retain_distillation_weight": 1.0,
        "recalibrate_batchnorm": False,
        "batchnorm_recalibration_batch_size": 2048,
    }
    candidates = [
        {"name": "implicit_common"},
        {
            "name": "explicit_equivalent",
            "repair_learning_rate": 1e-4,
            "repair_batch_size": 1024.0,
            "selected_parameter_weight": 1.0,
            "gradient_ascent_steps": 0.0,
            "gradient_ascent_learning_rate": 999.0,
            "gradient_ascent_batch_size": 999,
            "gradient_ascent_retain_distillation_weight": 999.0,
            "recalibrate_batchnorm": False,
            "batchnorm_recalibration_batch_size": 999,
        },
        {"name": "different", "gradient_ascent_steps": 2},
    ]

    assert semantic_candidate_fingerprint(
        common, candidates[0]
    ) == semantic_candidate_fingerprint(common, candidates[1])
    unique = deduplicate_candidates(candidates, common_candidate=common)
    assert [candidate["name"] for candidate in unique] == [
        "implicit_common",
        "different",
    ]


def _stage3_evidence(stage3_config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(merge_candidate_configs(stage3_config)):
        rows.append(
            {
                **candidate,
                "valid": True,
                "utility_floor_pass": True,
                "local_privacy_proxy": 1.0 - index * 0.01,
                "precision_at_10": 0.85 - index * 0.001,
                "utility_ratio": 1.0 - index * 0.001,
                "execution_time_seconds": 100.0 + index,
                "selected_parameter_fraction": 0.01 + index * 0.001,
                "local_search_score": 0.90 - index * 0.001,
            }
        )
    return pd.DataFrame(rows)


def test_stage4_selects_four_deterministic_diverse_finalists() -> None:
    stage2 = generate_stage2_config(_stage1_config(), _stage1_evidence())
    stage3 = generate_stage3_config(stage2, _stage2_evidence())
    evidence = _stage3_evidence(stage3)

    first = select_stage4_finalists(evidence)
    second = select_stage4_finalists(evidence)
    first_names = [finalist["name"] for finalist in first]
    assert first_names == [finalist["name"] for finalist in second]
    assert len(first_names) == len(set(first_names)) == 4
    structure_counts = Counter(
        (
            finalist["effective_candidate"]["top_fraction"],
            finalist["effective_candidate"]["minimum_dampening_factor"],
            finalist["effective_candidate"]["forget_absolute_quantile"],
            finalist["effective_candidate"]["recalibrate_batchnorm"],
        )
        for finalist in first
    )
    assert sorted(structure_counts.values()) == [2, 2]

    generated = generate_stage4_config(stage3, evidence)
    validate_search_config(generated)
    assert len(generated["candidates"]) == 4
    assert generated["add_gradient_ascent_variants"] == 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_evidence_bundle(
    root: Path,
    *,
    mode: str = "full",
    effective_mutation: tuple[str, str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    template = _stage1_config()
    template_path = root / "template.json"
    _write_json(template_path, template)
    results_path = root / "evidence/search_comparison.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _stage1_evidence().to_csv(results_path, index=False)
    effective = build_effective_search_config(
        template,
        quick=mode == "quick",
        max_candidates=None,
    )
    if effective_mutation is not None:
        section, key, value = effective_mutation
        if section == "__root__":
            effective[key] = value
        else:
            effective[section][key] = value
    _write_json(results_path.parent / "effective_search_config.json", effective)
    _write_json(
        results_path.parent / "search_metadata.json",
        {
            "status": "completed",
            "mode": mode,
            "effective_search_config": effective,
        },
    )
    return results_path, template_path, template


def test_generator_records_validated_full_evidence(tmp_path: Path) -> None:
    results_path, template_path, _ = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "generated/search_stage2_refinement.json"

    assert (
        generate_progressive_configs.main(
            [
                "stage2",
                "--results",
                str(results_path),
                "--template-config",
                str(template_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    generated = json.loads(output_path.read_text(encoding="utf-8"))
    validation = generated["progressive_generation"]["evidence_validation"]
    assert validation["validated"] is True
    assert validation["producer_status"] == "completed"
    assert validation["producer_mode"] == "full"
    assert validation["shared_settings_match"] is True
    assert validation["metadata_embedded_config_checked"] is True


def test_generator_rejects_quick_evidence(tmp_path: Path) -> None:
    results_path, template_path, _ = _write_evidence_bundle(
        tmp_path,
        mode="quick",
    )
    output_path = tmp_path / "generated.json"

    with pytest.raises(ValueError, match="mode='full'"):
        generate_progressive_configs.main(
            [
                "stage2",
                "--results",
                str(results_path),
                "--template-config",
                str(template_path),
                "--output",
                str(output_path),
            ]
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        (("fisher", "fisher_batch_size", 64), "fisher"),
        (("__root__", "seed", 93), "seed"),
        (("__root__", "add_gradient_ascent_variants", 0), "add_gradient"),
    ],
)
def test_generator_rejects_shared_setting_mismatch(
    tmp_path: Path,
    mutation: tuple[str, str, Any],
    expected_field: str,
) -> None:
    results_path, template_path, _ = _write_evidence_bundle(
        tmp_path,
        effective_mutation=mutation,
    )
    output_path = tmp_path / "generated.json"

    with pytest.raises(
        ValueError,
        match=rf"impostazioni condivise.*{expected_field}",
    ):
        generate_progressive_configs.main(
            [
                "stage2",
                "--results",
                str(results_path),
                "--template-config",
                str(template_path),
                "--output",
                str(output_path),
            ]
        )
    assert not output_path.exists()


def test_generator_requires_both_sibling_evidence_files(tmp_path: Path) -> None:
    results_path = tmp_path / "evidence/search_comparison.csv"
    results_path.parent.mkdir(parents=True)
    _stage1_evidence().to_csv(results_path, index=False)
    template = _stage1_config()
    template_path = tmp_path / "template.json"
    _write_json(template_path, template)

    with pytest.raises(FileNotFoundError, match="search_metadata.json"):
        generate_progressive_configs._validate_evidence_bundle(
            results_path,
            template_path,
            template,
        )
    _write_json(
        results_path.parent / "search_metadata.json",
        {"status": "completed", "mode": "full"},
    )
    with pytest.raises(FileNotFoundError, match="effective_search_config.json"):
        generate_progressive_configs._validate_evidence_bundle(
            results_path,
            template_path,
            template,
        )


def test_generator_refuses_protected_output_paths(tmp_path: Path) -> None:
    results_path = tmp_path / "search_comparison.csv"
    template_path = tmp_path / "template.json"
    protected_outputs = (
        results_path,
        template_path,
        REPOSITORY_ROOT / "configs/final_config.json",
        REPOSITORY_ROOT / "outputs/final_run/generated.json",
        REPOSITORY_ROOT / "submission/nested/generated.json",
    )

    for output_path in protected_outputs:
        with pytest.raises(ValueError, match="Output rifiutato"):
            generate_progressive_configs._assert_safe_output_path(
                output_path,
                results_path=results_path,
                template_path=template_path,
            )

    generate_progressive_configs._assert_safe_output_path(
        tmp_path / "safe/generated.json",
        results_path=results_path,
        template_path=template_path,
    )
