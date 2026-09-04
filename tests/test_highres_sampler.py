from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from ai_image_detector.training import (
    PAIRED_COMPONENT_BINARY_SAMPLER,
    PairedComponentBinarySampler,
    PairedGroupSampler,
    make_loader,
    train_sampler_metadata,
)


def _frozen_component_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    generators = ("sd21", "sd3", "sdxl")
    for component_number in range(9):
        component = f"component-{component_number}"
        rows.extend(
            (
                {
                    "path": f"{component}/real.png",
                    "label": 0,
                    "generator": "real",
                    "leakage_group": component,
                    "source_id": f"source-{component}-real",
                },
                {
                    "path": f"{component}/fake.png",
                    "label": 1,
                    "generator": generators[component_number % len(generators)],
                    "leakage_group": component,
                    "source_id": f"source-{component}-fake",
                },
            )
        )
    return pd.DataFrame(rows)


def test_component_binary_sampler_keeps_frozen_pairs_balanced_and_reproducible() -> None:
    frame = _frozen_component_frame()
    sampler = PairedComponentBinarySampler(frame, seed=17)
    indices = list(sampler)

    assert indices == list(PairedComponentBinarySampler(frame, seed=17))
    assert len(indices) == 2 * frame["leakage_group"].nunique()

    pairs: list[str] = []
    fake_generators: list[str] = []
    for real_index, fake_index in zip(indices[::2], indices[1::2], strict=True):
        real = frame.iloc[real_index]
        fake = frame.iloc[fake_index]
        assert real["label"] == 0
        assert fake["label"] == 1
        assert real["leakage_group"] == fake["leakage_group"]
        pairs.append(str(real["leakage_group"]))
        fake_generators.append(str(fake["generator"]))

    assert Counter(pairs) == {f"component-{index}": 1 for index in range(9)}
    assert Counter(fake_generators) == {"sd21": 3, "sd3": 3, "sdxl": 3}
    assert sampler.metadata() == {
        "choice": PAIRED_COMPONENT_BINARY_SAMPLER,
        "group_column": "leakage_group",
        "groups_per_epoch": 9,
        "paired_samples_per_epoch": 18,
        "fake_generators": ["sd21", "sd3", "sdxl"],
        "pairing": "one_real_and_one_fake_uniform_within_component",
    }

    sampler.set_epoch(4)
    epoch_four = list(sampler)
    rerun = PairedComponentBinarySampler(frame, seed=17)
    rerun.set_epoch(4)
    assert epoch_four == list(rerun)


@pytest.mark.parametrize(
    ("rows", "message"),
    (
        (
            [
                {"label": 0, "leakage_group": "only-real"},
                {"label": 1, "leakage_group": "complete"},
                {"label": 0, "leakage_group": "complete"},
            ],
            "missing fake=True",
        ),
        (
            [
                {"label": 1, "leakage_group": "only-fake"},
                {"label": 1, "leakage_group": "complete"},
                {"label": 0, "leakage_group": "complete"},
            ],
            "missing real=True",
        ),
    ),
)
def test_component_binary_sampler_rejects_incomplete_components(
    rows: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PairedComponentBinarySampler(pd.DataFrame(rows), seed=17)


def test_component_binary_sampler_is_routed_without_relaxing_h1n_sampler() -> None:
    frame = _frozen_component_frame()
    loader = make_loader(
        frame,
        transform=lambda image: image,
        batch_size=2,
        train=True,
        sampler_protocol=PAIRED_COMPONENT_BINARY_SAMPLER,
        seed=17,
    )

    assert isinstance(loader.sampler, PairedComponentBinarySampler)
    assert train_sampler_metadata(loader)["choice"] == PAIRED_COMPONENT_BINARY_SAMPLER
    with pytest.raises(ValueError, match="Expected 5 fake generators"):
        PairedGroupSampler(frame, seed=17)
