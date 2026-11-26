"""Unit, integration, determinism, rollback, and exact-inheritance tests for CORTEX-ECCNR."""
from __future__ import annotations

import numpy as np
import pytest

from eccnr.connectome import build_synthetic_connectome, rewire_connectome_degree_preserving
from eccnr.env import ACTION_DIM, ECCNRWorld, OBS_DIM, TASKS
from eccnr.evolution import LearnedRepairController, diagnose_parent
from eccnr.experiments import run_single_condition, run_successive_action_experiment
from eccnr.fusion import fuse_parents
from eccnr.replay import run_adaptation_stage, run_developmental_trace_replay
from eccnr.substrate import ConnectomeBrain, SubstrateBudgets


def test_env_contract_and_state_cloning():
    env = ECCNRWorld(max_steps=40, seed=123)
    for task in TASKS:
        obs = env.reset(task=task, seed=123)
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()

    env.reset(task="sequential", seed=77)
    snap = env.clone_state()
    act = np.array([0.5, -0.2, 0.0, 0.1, 0.0], dtype=np.float32)
    r1 = env.step(act)
    env.restore_state(snap)
    r2 = env.step(act)
    assert np.allclose(r1.obs, r2.obs)
    assert r1.reward == pytest.approx(r2.reward)


def test_cow_weights_dales_law_and_rollback():
    ref = build_synthetic_connectome(n_nodes=48, seed=11)
    base_w = ref.to_initial_weights()
    b1 = ConnectomeBrain(ref, lineage_id="L1", generation=0, seed=11, shared_base_weights=base_w)
    b2 = ConnectomeBrain(ref, lineage_id="L2", generation=0, seed=12, shared_base_weights=base_w)
    assert b1.bio_weights is base_w
    assert b2.bio_weights is base_w

    snap = b1.snapshot()
    h_before = b1.weight_hash()

    obs = np.ones(OBS_DIM, dtype=np.float32) * 0.1
    act, pred = b1.forward(obs)
    assert act.shape == (ACTION_DIM,)
    b1.update_plasticity(obs, act, pred, obs, reward=0.8, lr=0.05, update_recurrent=True)
    # CoW triggered for b1, b2 untouched
    assert b1.bio_weights is not base_w
    assert b2.bio_weights is base_w

    # Dale's law check on biological edges
    pre_signs = ref.signs[ref.indices]
    nonzero = pre_signs != 0
    assert np.all(np.sign(b1.bio_weights[nonzero]) * pre_signs[nonzero] >= 0)

    # Neurogenesis & rollback
    nid = b1.add_neuron(role="central", sign=1)
    assert nid is not None and b1.n_active == 49
    b1.restore(snap)
    assert b1.n_active == 48
    assert b1.weight_hash() == h_before


def test_fusion_exact_inheritance_and_provenance():
    ref = build_synthetic_connectome(n_nodes=32, seed=5)
    base_w = ref.to_initial_weights()
    p1 = ConnectomeBrain(ref, lineage_id="L0", generation=1, seed=5, shared_base_weights=base_w)
    p2 = ConnectomeBrain(ref, lineage_id="L1", generation=1, seed=6, shared_base_weights=base_w)
    p1.add_neuron(role="central", sign=1, hypothesis="p1_spec")
    p2.add_neuron(role="sensory", sign=-1, hypothesis="p2_spec")
    p2.add_edge(0, 32, weight=-0.15)

    d1 = diagnose_parent(p1, seeds=[1, 2], max_steps=20)
    d2 = diagnose_parent(p2, seeds=[1, 2], max_steps=20)
    fused, prov, pmaps = fuse_parents([p1, p2], [d1, d2], strategy="hybrid_exact", seed=99)

    assert prov.exact_inheritance_verified is True
    best_p = p1 if d1.mean_reward >= d2.mean_reward else p2
    assert np.array_equal(fused.bio_weights, best_p.bio_weights)
    assert len(pmaps) == 2


def test_rewire_preserves_degrees():
    ref = build_synthetic_connectome(n_nodes=40, seed=7)
    rew = rewire_connectome_degree_preserving(ref, seed=77, swaps_per_edge=3)
    in_deg_orig = np.diff(ref.indptr)
    in_deg_rew = np.diff(rew.indptr)
    assert np.array_equal(in_deg_orig, in_deg_rew)
    assert int(ref.synapse_counts.sum()) == int(rew.synapse_counts.sum())


def test_dtr_and_end_to_end_smoke():
    ref = build_synthetic_connectome(n_nodes=32, seed=9)
    res = run_single_condition(
        condition="full_dtr_finetune",
        ref=ref,
        train_seeds=[1, 2],
        val_seeds=[3, 4],
        sealed_seeds=[5, 6],
        total_env_budget=800,
        n_lineages=2,
        n_generations=2,
        max_episode_steps=25,
        seed=9,
    )
    assert res["exact_inheritance_verified"] is True
    assert np.isfinite(res["sealed_mean_reward"])
    assert res["env_steps_used"] > 0

    succ_res = run_successive_action_experiment(ref, seeds=[1, 2, 3, 4], max_episode_steps=25)
    assert np.isfinite(succ_res["successive_handoff_mean_reward"])


def test_bptt_preserves_dales_law_and_topology():
    ref = build_synthetic_connectome(n_nodes=32, seed=15)
    res = run_single_condition(
        condition="full_dtr_finetune",
        ref=ref,
        train_seeds=[1, 2],
        val_seeds=[3, 4],
        sealed_seeds=[5, 6],
        total_env_budget=600,
        n_lineages=2,
        n_generations=2,
        max_episode_steps=25,
        seed=15,
        trainer="bptt",
    )
    assert res["exact_inheritance_verified"] is True
    assert np.isfinite(res["sealed_mean_reward"])
    assert res["env_steps_used"] > 0


def test_dendritic_delays_and_astrocyte_dynamics():
    ref = build_synthetic_connectome(n_nodes=32, seed=21)
    b = ConnectomeBrain(ref, lineage_id="DEND", generation=0, seed=21, enable_delays=True, enable_dendritic_gating=True)
    obs = np.ones(OBS_DIM, dtype=np.float32) * 0.25
    for _ in range(6):
        act, pred = b.forward(obs)
    assert act.shape == (ACTION_DIM,)
    assert pred.shape == (9,)
    assert 0.0 <= b.astrocyte_ca <= 1.0
    tel = b.export_graph_telemetry()
    assert tel["n_active_nodes"] == 32
    assert len(tel["nodes"]) == 32


def test_qd_archive_ucb_and_gated_specialist_routing():
    from eccnr.evolution import QualityDiversityArchive, cross_lineage_exchange

    ref = build_synthetic_connectome(n_nodes=32, seed=31)
    base_w = ref.to_initial_weights()
    p1 = ConnectomeBrain(ref, lineage_id="L0", generation=1, seed=31, shared_base_weights=base_w)
    p2 = ConnectomeBrain(ref, lineage_id="L1", generation=1, seed=32, shared_base_weights=base_w)
    p1.add_neuron(role="central", sign=1)
    p2.add_neuron(role="motor", sign=-1)
    p2.add_edge(0, 32, weight=-0.2)

    d1 = diagnose_parent(p1, seeds=[1, 2], max_steps=18)
    d2 = diagnose_parent(p2, seeds=[1, 2], max_steps=18)

    qd = QualityDiversityArchive()
    assert qd.consider(p1, d1) is True
    qd.consider(p2, d2)
    assert len(qd.cells) >= 1
    cross_lineage_exchange(p1, p2.snapshot(), d2, max_graft_edges=2)

    ctrl = LearnedRepairController(seed=31)
    prop = ctrl.propose(d1, mode="ucb_learned")
    assert prop.edit_kind in ctrl.EDIT_KINDS

    fused, prov, _ = fuse_parents([p1, p2], [d1, d2], strategy="gated_specialist_routing", seed=88)
    assert prov.exact_inheritance_verified is True
    assert len(fused.specialist_w_outs) == 2
    act, _ = fused.forward(np.zeros(OBS_DIM, dtype=np.float32))
    assert act.shape == (ACTION_DIM,)
    assert pytest.approx(float(np.sum(fused.last_routing_weights)), rel=1e-4) == 1.0


def test_ewc_dtr_and_gated_fusion_ewc_conditions():
    ref = build_synthetic_connectome(n_nodes=32, seed=41)
    for cond in ("ewc_dtr_finetune", "gated_fusion_ewc"):
        res = run_single_condition(
            condition=cond,
            ref=ref,
            train_seeds=[1, 2],
            val_seeds=[3, 4],
            sealed_seeds=[5, 6],
            total_env_budget=600,
            n_lineages=2,
            n_generations=2,
            max_episode_steps=20,
            seed=41,
            trainer="bptt",
        )
        assert res["exact_inheritance_verified"] is True
        assert np.isfinite(res["sealed_mean_reward"])
        assert res["qd_elites"] >= 1


def test_checkpoint_sha256_persistence_and_journal(tmp_path):
    from eccnr.persistence import AppendOnlyJournal, load_brain_checkpoint, save_brain_checkpoint

    ref = build_synthetic_connectome(n_nodes=24, seed=51)
    b = ConnectomeBrain(ref, lineage_id="CKPT_TEST", generation=2, seed=51, enable_delays=True)
    b.add_neuron(role="central", sign=1)
    b.add_edge(0, 24, weight=0.18)
    manifest = save_brain_checkpoint(b, tmp_path, tag="v1")
    meta_path = tmp_path / f"{manifest['checkpoint_id']}.json"

    loaded = load_brain_checkpoint(meta_path, ref)
    assert loaded.n_active == 25
    assert np.array_equal(loaded.bio_weights, b.bio_weights)
    assert (0, 24) in loaded.overlay.novel_edges

    journal = AppendOnlyJournal(tmp_path / "ledger.jsonl")
    rec_auto = journal.log_intervention("c1", {}, {}, "ok", "automatic_only", "accept", "auto", "none")
    rec_human = journal.log_intervention("c1", {}, {}, "ok", "human_gui", "bounded_edit", "manual", "neurogenesis")
    assert rec_auto["genuinely_reviewed"] is False
    assert rec_human["genuinely_reviewed"] is True
    assert len(journal.read_all()) == 2


def test_gui_fastapi_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from eccnr.gui_server import create_app

    app = create_app(project_root=tmp_path)
    client = TestClient(app)

    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "CORTEX-ECCNR Observatory" in r_html.text

    r_stat = client.get("/api/status")
    assert r_stat.status_code == 200
    assert r_stat.json()["n_bio_nodes"] == 96

    r_step = client.post("/api/world/step", json={"n_steps": 3, "online_plasticity": True})
    assert r_step.status_code == 200
    assert r_step.json()["world"]["t"] == 3

    r_int = client.post("/api/brain/intervene", json={"action": "neurogenesis", "rationale": "unit test"})
    assert r_int.status_code == 200
    assert r_int.json()["graph"]["n_novel_neurons"] >= 1

    r_evo = client.post("/api/evolution/mini_run", json={
        "n_lineages": 2,
        "n_generations": 2,
        "steps_per_gen": 60,
        "trainer": "eligibility",
        "fusion_strategy": "gated_specialist_routing",
    })
    assert r_evo.status_code == 200
    assert len(r_evo.json()["fusion_comparisons"]) == 7

    r_j = client.get("/api/journal")
    assert r_j.status_code == 200
    assert len(r_j.json()["entries"]) >= 1
