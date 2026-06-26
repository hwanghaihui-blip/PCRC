from __future__ import annotations

from pcrc.config import load_config
from pcrc.constants import MAINLINE_METHODS, OFFPOLICY_METHODS, SMOKE_METHODS
from pcrc.contract import artifact_manifest_frame, canonical_method_rosters, dataset_sources_frame, method_label_map, method_sources_frame


def test_artifact_manifest_uses_natural_number_order():
    frame = artifact_manifest_frame()
    overview = frame[(frame["experiment"] == "exp1_overview") & (frame["artifact_type"].isin(["figure", "table"]))]
    assert overview["number"].tolist() == ["fig1_1", "table1_1", "table1_2"]
    d0_figures = frame[(frame["experiment"] == "exp2_phase_transition") & (frame["artifact_type"] == "figure")]
    assert d0_figures["number"].tolist()[:4] == ["fig2_1", "fig2_2", "fig2_3", "fig2_4"]
    assert d0_figures["number"].tolist()[-1] == "fig2_10"


def test_method_sources_are_pcrc_only():
    frame = method_sources_frame()
    assert frame["method"].tolist() == ["PCRC"]


def test_mainline_configs_share_the_same_pcrc_roster():
    for config_name in ["exp2_phase_transition", "exp3_m5_case", "exp4_credit_case"]:
        assert load_config(config_name).methods == MAINLINE_METHODS


def test_special_track_configs_keep_track_specific_rosters():
    assert load_config("exp5_offpolicy").methods == OFFPOLICY_METHODS
    assert load_config("smoke_d0").methods == SMOKE_METHODS


def test_method_labels_and_rosters_expose_user_facing_names():
    labels = method_label_map()
    assert labels == {"PCRC": "PCRC"}
    assert canonical_method_rosters()["mainline"] == MAINLINE_METHODS


def test_dataset_sources_cover_packaged_experiments():
    frame = dataset_sources_frame()
    assert set(frame["dataset"]) == {"D0 synthetic", "D1 M5", "D2 Credit"}
