#!/usr/bin/env python3
"""
WP9d3 — Verified Computational Reproducibility Audit

Audits the actual WP1–WP8 script pipeline plus frozen WP9a–WP9c artefacts.
The audit is source-based and non-executing: it does not rerun analyses or
recompute scientific results.

Unknown values are reported as "Not verified". A FAIL is produced only for a
verified missing required script/artefact, an uncontrolled stochastic stage,
a broken declared dependency, or a circular dependency.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "framework_validation" / "09d3_computational_reproducibility_audit"
NOT_VERIFIED = "Not verified"
PASS, PARTIAL, FAIL = "PASS", "PARTIAL", "FAIL"

@dataclass(frozen=True)
class ScriptSpec:
    order: int
    wp: str
    script: str
    aliases: tuple[str, ...] = ()
    stochastic: bool = False
    expected_seeds: tuple[int, ...] = ()
    upstream: tuple[str, ...] = ()

SCRIPTS = (
    ScriptSpec(1,"WP1","01_build_master_dataset.py"),
    ScriptSpec(2,"WP2","02_company_statistics.py",upstream=("01_build_master_dataset.py",)),
    ScriptSpec(3,"WP3","03_company_semantic_profiles.py",upstream=("01_build_master_dataset.py",)),
    ScriptSpec(4,"WP4","04_company_significance.py",stochastic=True,expected_seeds=(20260716,),upstream=("01_build_master_dataset.py","03_company_semantic_profiles.py")),
    ScriptSpec(5,"WP5","05a_build_temporal_profiles.py",upstream=("01_build_master_dataset.py",)),
    ScriptSpec(6,"WP5","05b_semantic_drift.py",upstream=("05a_build_temporal_profiles.py",)),
    ScriptSpec(7,"WP5","05c_temporal_trajectories.py",upstream=("05a_build_temporal_profiles.py","05b_semantic_drift.py")),
    ScriptSpec(8,"WP5","05d_semantic_change_points.py",upstream=("05a_build_temporal_profiles.py","05b_semantic_drift.py")),
    ScriptSpec(9,"WP5","05e_temporal_stability_summary.py",upstream=("05b_semantic_drift.py","05c_temporal_trajectories.py","05d_semantic_change_points.py")),
    ScriptSpec(10,"WP6","06a_similarity_metric_sensitivity.py",upstream=("03_company_semantic_profiles.py",)),
    ScriptSpec(11,"WP6","06b_bootstrap_company_profiles.py",stochastic=True,expected_seeds=(20260717,),upstream=("01_build_master_dataset.py","03_company_semantic_profiles.py")),
    ScriptSpec(12,"WP6","06c_leave_one_artist_out.py",upstream=("01_build_master_dataset.py","03_company_semantic_profiles.py")),
    ScriptSpec(13,"WP6","06d_balanced_company_sampling.py",stochastic=True,expected_seeds=(20260718,),upstream=("01_build_master_dataset.py","03_company_semantic_profiles.py")),
    ScriptSpec(14,"WP6","06e_temporal_window_sensitivity.py",upstream=("01_build_master_dataset.py","05a_build_temporal_profiles.py")),
    ScriptSpec(15,"WP6","06f_community_detection_sensitivity.py",stochastic=True,expected_seeds=(20260718,20260719,20260720,20260721,20260722),upstream=("01_build_master_dataset.py",)),
    ScriptSpec(16,"WP6","06g_summary_robustness.py",upstream=("06a_similarity_metric_sensitivity.py","06b_bootstrap_company_profiles.py","06c_leave_one_artist_out.py","06d_balanced_company_sampling.py","06e_temporal_window_sensitivity.py","06f_community_detection_sensitivity.py")),
    ScriptSpec(17,"WP7","07a_company_prediction.py",stochastic=True,expected_seeds=(20260718,),upstream=("01_build_master_dataset.py",)),
    ScriptSpec(18,"WP7","07b_temporal_forecasting.py",upstream=("05a_build_temporal_profiles.py",)),
    ScriptSpec(19,"WP7","07c_company_identifiability.py",aliases=("07c_company_identifiability(1).py",),stochastic=True,expected_seeds=(20260719,),upstream=("01_build_master_dataset.py",)),
    ScriptSpec(20,"WP7","07d_prediction_summary.py",aliases=("07d_prediction_summary(1).py",),upstream=("07a_company_prediction.py","07b_temporal_forecasting.py","07c_company_identifiability.py")),
    ScriptSpec(21,"WP8","08a_semantic_signature_decomposition.py",aliases=("08a_semantic_signature_decomposition(1).py",),upstream=("03_company_semantic_profiles.py",)),
    ScriptSpec(22,"WP8","08b_community_contribution_analysis.py",aliases=("08b_community_contribution_analysis(1).py",),upstream=("03_company_semantic_profiles.py",)),
    ScriptSpec(23,"WP8","08b1_community_semantic_dictionary.py",aliases=("08b1_community_semantic_dictionary(1).py",),upstream=("08b_community_contribution_analysis.py",)),
    ScriptSpec(24,"WP8","08c_semantic_transition_explanation.py",aliases=("08c_semantic_transition_explanation(1).py",),upstream=("05b_semantic_drift.py","05c_temporal_trajectories.py","05d_semantic_change_points.py")),
    ScriptSpec(25,"WP8","08d_organisational_semantic_archetypes.py",upstream=("08a_semantic_signature_decomposition.py","08b_community_contribution_analysis.py","08b1_community_semantic_dictionary.py")),
    ScriptSpec(26,"WP8","08e_company_differentiation_matrix.py",upstream=("08a_semantic_signature_decomposition.py","08b_community_contribution_analysis.py","08d_organisational_semantic_archetypes.py")),
    ScriptSpec(27,"WP8","08f_integrated_organisational_identity.py",upstream=("08a_semantic_signature_decomposition.py","08b_community_contribution_analysis.py","08b1_community_semantic_dictionary.py","08c_semantic_transition_explanation.py","08d_organisational_semantic_archetypes.py","08e_company_differentiation_matrix.py","06g_summary_robustness.py","07d_prediction_summary.py")),
)

REQUIRED_ARTEFACTS = (
    ("WP1","outputs/tables/company_master_analysis_v1.parquet"),
    ("WP3","outputs/company_profiles/tables/table_company_semantic_profiles.csv"),
    ("WP3","outputs/company_profiles/tables/company_semantic_diversity.csv"),
    ("WP4","outputs/company_significance/tables/artist_level_global_metric_tests.csv"),
    ("WP5","outputs/temporal/rolling5/temporal_stability_summary.csv"),
    ("WP6","outputs/robustness/summary_robustness/wp6_all_robustness_records.csv"),
    ("WP9a","outputs/framework_validation/09a4_cross_workpackage_consistency/cross_wp_consistency_summary.csv"),
    ("WP9b","outputs/framework_validation/09b_evidence_traceability_network/traceability_run_summary.csv"),
    ("WP9c","outputs/framework_validation/09c_methodological_transferability/transferability_run_summary.csv"),
)

EXCLUDED = {".git","outputs","__pycache__",".venv","venv","env","node_modules"}

def rel(p: Path) -> str:
    try: return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError: return p.as_posix()

def locate(names: Iterable[str]) -> Path | None:
    names = tuple(names)
    direct_dirs = (ROOT, ROOT/"scripts", ROOT/"src", ROOT/"analysis", ROOT/"scripts"/"company_semantics")
    for d in direct_dirs:
        for n in names:
            p=d/n
            if p.is_file(): return p
    matches=[]
    for n in names:
        for p in ROOT.rglob(n):
            if p.is_file() and not any(x in EXCLUDED for x in p.parts): matches.append(p)
    return sorted(matches,key=lambda p: rel(p))[0] if matches else None

def source_info(path: Path) -> dict:
    text=path.read_text(encoding="utf-8",errors="replace")
    try:
        tree=ast.parse(text)
        syntax="Yes"
    except SyntaxError:
        tree=None; syntax="No"
    nums={int(x) for x in re.findall(r"\b20\d{6}\b",text)}
    random_markers=bool(re.search(r"default_rng|random_state|KMeans\(|bootstrap|permutation",text,re.I))
    main_guard="Yes" if re.search(r"if\s+__name__\s*==\s*[\"']__main__[\"']",text) else "No"
    root_convention="Yes" if re.search(r"Path\(__file__\)\.resolve\(\)\.parents\[2\]",text) else ("Alternative verified" if "Path(__file__)" in text and ("PROJECT_ROOT" in text or "ROOT" in text) else NOT_VERIFIED)
    interactive="Yes" if re.search(r"\binput\s*\(|tkinter|click\.prompt|questionary",text) else "No"
    return {"syntax_valid":syntax,"main_guard":main_guard,"root_convention":root_convention,"interactive_input":interactive,"seed_values":sorted(nums),"random_markers":random_markers,"source_lines":len(text.splitlines())}

def build_script_audit() -> pd.DataFrame:
    rows=[]
    for s in SCRIPTS:
        p=locate((s.script,*s.aliases))
        if p is None:
            rows.append({"order":s.order,"workpackage":s.wp,"script":s.script,"resolved_path":NOT_VERIFIED,"script_present":"No","syntax_valid":NOT_VERIFIED,"entrypoint":"Not available","root_convention":NOT_VERIFIED,"interactive_input":NOT_VERIFIED,"stochastic_stage":"Yes" if s.stochastic else "No","required_seeds":"; ".join(map(str,s.expected_seeds)) or "Not applicable","detected_seeds":"","seed_control":"No" if s.stochastic else "Not applicable","determinism":"Not verified","upstream_scripts":"; ".join(s.upstream),"status":FAIL,"reason":"Required pipeline script not found"})
            continue
        info=source_info(p)
        missing=[x for x in s.expected_seeds if x not in info["seed_values"]]
        seed_control=("Yes" if not missing else "No") if s.stochastic else "Not applicable"
        deterministic="Yes" if (not s.stochastic or not missing) and info["interactive_input"]=="No" else "No"
        entry="Main guard" if info["main_guard"]=="Yes" else "Top-level executable script"
        failures=[]
        if info["syntax_valid"]!="Yes": failures.append("invalid Python syntax")
        if info["interactive_input"]=="Yes": failures.append("interactive input detected")
        if s.stochastic and missing: failures.append("required seed not detected: "+", ".join(map(str,missing)))
        status=FAIL if failures else PASS
        rows.append({"order":s.order,"workpackage":s.wp,"script":s.script,"resolved_path":rel(p),"script_present":"Yes","syntax_valid":info["syntax_valid"],"entrypoint":entry,"root_convention":info["root_convention"],"interactive_input":info["interactive_input"],"stochastic_stage":"Yes" if s.stochastic else "No","required_seeds":"; ".join(map(str,s.expected_seeds)) or "Not applicable","detected_seeds":"; ".join(map(str,info["seed_values"])) or "None","seed_control":seed_control,"determinism":deterministic,"source_lines":info["source_lines"],"upstream_scripts":"; ".join(s.upstream),"status":status,"reason":"; ".join(failures) if failures else "Verified from source"})
    return pd.DataFrame(rows)

def build_dependencies(audit: pd.DataFrame) -> pd.DataFrame:
    present=dict(zip(audit.script,audit.script_present))
    rows=[]
    for s in SCRIPTS:
        for up in s.upstream:
            ok=present.get(up)=="Yes"
            rows.append({"downstream_script":s.script,"upstream_script":up,"upstream_present":"Yes" if ok else "No","dependency_status":PASS if ok else FAIL})
    return pd.DataFrame(rows)

def has_cycle() -> bool:
    g={s.script:set(s.upstream) for s in SCRIPTS}; visiting=set(); done=set()
    def dfs(n):
        if n in visiting:return True
        if n in done:return False
        visiting.add(n)
        if any(dfs(x) for x in g.get(n,set())):return True
        visiting.remove(n);done.add(n);return False
    return any(dfs(n) for n in g)

def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    audit=build_script_audit()
    deps=build_dependencies(audit)
    artefacts=pd.DataFrame([{"workpackage":wp,"required_artefact":p,"exists":"Yes" if (ROOT/p).is_file() else "No","status":PASS if (ROOT/p).is_file() else FAIL} for wp,p in REQUIRED_ARTEFACTS])
    cycle=has_cycle()
    checks=pd.DataFrame([
        {"check":"All 27 WP1-WP8 scripts located","result":"Yes" if (audit.script_present=="Yes").all() else "No","status":PASS if (audit.script_present=="Yes").all() else FAIL},
        {"check":"All scripts have valid Python syntax","result":"Yes" if (audit.syntax_valid=="Yes").all() else "No","status":PASS if (audit.syntax_valid=="Yes").all() else FAIL},
        {"check":"All stochastic stages have explicit seed control","result":"Yes" if ((audit.stochastic_stage!="Yes")|(audit.seed_control=="Yes")).all() else "No","status":PASS if ((audit.stochastic_stage!="Yes")|(audit.seed_control=="Yes")).all() else FAIL},
        {"check":"No interactive/manual runtime input detected","result":"Yes" if (audit.interactive_input=="No").all() else "No","status":PASS if (audit.interactive_input=="No").all() else FAIL},
        {"check":"All declared script dependencies resolve","result":"Yes" if (deps.dependency_status==PASS).all() else "No","status":PASS if (deps.dependency_status==PASS).all() else FAIL},
        {"check":"No circular dependencies","result":"Yes" if not cycle else "No","status":PASS if not cycle else FAIL},
        {"check":"Canonical retained artefacts are present","result":"Yes" if (artefacts.status==PASS).all() else "No","status":PASS if (artefacts.status==PASS).all() else FAIL},
    ])
    overall=PASS if (audit.status==PASS).all() and (deps.dependency_status==PASS).all() and (artefacts.status==PASS).all() and not cycle else FAIL
    summary=pd.DataFrame([{"metric":"scripts_audited","value":len(audit)},{"metric":"scripts_pass","value":int((audit.status==PASS).sum())},{"metric":"scripts_fail","value":int((audit.status==FAIL).sum())},{"metric":"stochastic_stages","value":int((audit.stochastic_stage=="Yes").sum())},{"metric":"seed_controlled_stochastic_stages","value":int(((audit.stochastic_stage=="Yes")&(audit.seed_control=="Yes")).sum())},{"metric":"dependencies_audited","value":len(deps)},{"metric":"unresolved_dependencies","value":int((deps.dependency_status==FAIL).sum())},{"metric":"required_artefacts","value":len(artefacts)},{"metric":"missing_required_artefacts","value":int((artefacts.status==FAIL).sum())},{"metric":"circular_dependencies","value":int(cycle)},{"metric":"overall_computational_reproducibility","value":overall}])
    run=pd.DataFrame([{"project_root":str(ROOT),"output_directory":str(OUT),"generated_at_utc":datetime.now(timezone.utc).isoformat(),"scripts_audited":len(audit),"scripts_pass":int((audit.status==PASS).sum()),"scripts_fail":int((audit.status==FAIL).sum()),"stochastic_stages":int((audit.stochastic_stage=="Yes").sum()),"seed_controlled_stochastic_stages":int(((audit.stochastic_stage=="Yes")&(audit.seed_control=="Yes")).sum()),"unresolved_dependencies":int((deps.dependency_status==FAIL).sum()),"missing_required_artefacts":int((artefacts.status==FAIL).sum()),"circular_dependencies":int(cycle),"overall_computational_reproducibility":overall,"audit_scope":"Verified WP1-WP8 source scripts, explicit stochastic control, dependencies, retained canonical artefacts, and WP9a-WP9c continuity","upstream_scripts_rerun":"No","new_statistics_computed":"No"}])
    audit.to_csv(OUT/"verified_script_inventory.csv",index=False)
    deps.to_csv(OUT/"verified_dependency_integrity.csv",index=False)
    artefacts.to_csv(OUT/"verified_artefact_manifest.csv",index=False)
    checks.to_csv(OUT/"reproducibility_checklist.csv",index=False)
    summary.to_csv(OUT/"computational_reproducibility_summary.csv",index=False)
    run.to_csv(OUT/"reproducibility_run_summary.csv",index=False)
    print("="*88);print("WP9d3 VERIFIED COMPUTATIONAL REPRODUCIBILITY AUDIT COMPLETE");print("="*88)
    print(f"Project root: {ROOT}");print(f"Output directory: {OUT}")
    print(f"WP1-WP8 scripts audited: {len(audit)}")
    print(f"Script status: {(audit.status==PASS).sum()} PASS, {(audit.status==FAIL).sum()} FAIL")
    print(f"Stochastic stages: {(audit.stochastic_stage=='Yes').sum()} ({((audit.stochastic_stage=='Yes')&(audit.seed_control=='Yes')).sum()} seed-controlled)")
    print(f"Dependencies: {len(deps)-(deps.dependency_status==FAIL).sum()} resolved, {(deps.dependency_status==FAIL).sum()} unresolved")
    print(f"Required artefacts: {len(artefacts)-(artefacts.status==FAIL).sum()} present, {(artefacts.status==FAIL).sum()} missing")
    print(f"Circular dependencies: {int(cycle)}")
    print(f"Overall computational reproducibility: {overall}")
    return 0 if overall==PASS else 1

if __name__=="__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt: raise SystemExit(130)
