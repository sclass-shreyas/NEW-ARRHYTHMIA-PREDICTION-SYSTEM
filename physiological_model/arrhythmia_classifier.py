"""
PERSON 3 — Physiological Model
Day 3: Arrhythmia Rule Engine

File: physiological_model/arrhythmia_classifier.py

What this module does:
    Takes the HRV feature dict from hrv_features.py and applies
    clinically grounded threshold rules to:
        1. Flag specific physiological violations
        2. Infer arrhythmia type
        3. Compute a weighted physiological risk score (0.0 – 1.0)
        4. Generate a human-readable explanation string

    This is the INTERPRETABILITY core of the digital twin.
    Unlike the LSTM (Person 2) which is a black box, this module
    explains WHY a beat is flagged — making it XAI-compliant.

Clinical references:
    - Task Force of ESC/NASPE, Eur Heart J, 1996 (HRV standards)
    - AHA/ACC Guidelines for arrhythmia classification
    - Poincaré analysis: Brennan et al., IEEE Trans Biomed Eng, 2001

Usage:
    from physiological_model.arrhythmia_classifier import classify_beat
    result = classify_beat(features)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# CLINICAL THRESHOLDS  (grounded in literature)
# ─────────────────────────────────────────────────────────────

THR = {
    # Heart rate
    'hr_tachy'        : 100.0,   # bpm — above = tachycardia
    'hr_brady'        : 60.0,    # bpm — below = bradycardia

    # SDNN — overall HRV
    'sdnn_low'        : 20.0,    # ms  — below = severely depressed HRV
    'sdnn_high'       : 150.0,   # ms  — above = excessive variability

    # RMSSD — short-term / vagal HRV
    'rmssd_high'      : 80.0,    # ms  — above = excessive vagal / ectopic

    # pNN50 — beat-to-beat irregularity
    'pnn50_high'      : 50.0,    # %   — above = highly irregular (AF risk)

    # Poincaré geometry
    'sd1sd2_low'      : 0.15,    # ratio — below = overly rigid rhythm
    'sd1sd2_high'     : 1.0,     # ratio — above = chaotic / AF-like

    # LF/HF — autonomic balance
    'lfhf_high'       : 2.5,     # ratio — above = sympathetic dominance

    # Sample Entropy — signal complexity
    'sampen_high'     : 1.5,     # above = high irregularity
}

# ─────────────────────────────────────────────────────────────
# RISK WEIGHTS  (how much each flag contributes to final score)
# Higher weight = stronger evidence of arrhythmia
# Weights sum to 1.0 — tuned to match clinical importance
# ─────────────────────────────────────────────────────────────

WEIGHTS = {
    'hr_flag'         : 0.15,   # HR out of range
    'sdnn_flag'       : 0.20,   # HRV depression (strong predictor)
    'rmssd_flag'      : 0.15,   # Short-term variability excess
    'pnn50_flag'      : 0.20,   # Beat-to-beat irregularity (AF marker)
    'poincare_flag'   : 0.15,   # Geometric irregularity
    'lfhf_flag'       : 0.10,   # Autonomic imbalance
    'sampen_flag'     : 0.05,   # Entropy (supplementary)
}


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def classify_beat(features: dict) -> dict:
    """
    Applies physiological rule engine to HRV features.

    Parameters
    ----------
    features : dict
        Output of compute_hrv_features() from hrv_features.py.
        Must contain at minimum: mean_hr_bpm, sdnn_ms, rmssd_ms,
        pnn50_pct, sd1_sd2_ratio. Frequency and nonlinear features
        are optional (used if present).

    Returns
    -------
    dict with keys:
        phys_risk_score  : float (0.0 – 1.0)
        classification   : 'normal' | 'abnormal'
        arrhythmia_type  : str  (specific condition inferred)
        flags            : dict of bool (which rules fired)
        explanation      : str  (human-readable, pipe-separated)
        risk_breakdown   : dict (per-flag weighted contribution)
    """

    # ── Extract features with safe defaults ──────────────────────────────
    hr     = features.get('mean_hr_bpm',    75.0)
    sdnn   = features.get('sdnn_ms',        50.0)
    rmssd  = features.get('rmssd_ms',       30.0)
    pnn50  = features.get('pnn50_pct',       0.0)
    sd1sd2 = features.get('sd1_sd2_ratio',   0.5)
    lf_hf  = features.get('lf_hf_ratio',   None)
    sampen = features.get('sample_entropy', None)

    flags        = {}
    explanations = []
    risk_scores  = {}

    # ─────────────────────────────────────────────────────────────────────
    # RULE 1 — Heart Rate Boundaries
    # ─────────────────────────────────────────────────────────────────────
    if hr > THR['hr_tachy']:
        flags['hr_flag'] = True
        flags['tachycardia'] = True
        explanations.append(
            f"HR {hr:.1f} bpm > {THR['hr_tachy']} bpm threshold → Sinus Tachycardia risk"
        )
        severity = min(1.0, (hr - THR['hr_tachy']) / 50.0)   # scales with degree
        risk_scores['hr_flag'] = WEIGHTS['hr_flag'] * (0.5 + 0.5 * severity)

    elif hr < THR['hr_brady']:
        flags['hr_flag'] = True
        flags['bradycardia'] = True
        explanations.append(
            f"HR {hr:.1f} bpm < {THR['hr_brady']} bpm threshold → Sinus Bradycardia risk"
        )
        severity = min(1.0, (THR['hr_brady'] - hr) / 30.0)
        risk_scores['hr_flag'] = WEIGHTS['hr_flag'] * (0.5 + 0.5 * severity)

    else:
        flags['hr_flag'] = False
        risk_scores['hr_flag'] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # RULE 2 — SDNN (overall HRV — strongest single predictor)
    # ─────────────────────────────────────────────────────────────────────
    if sdnn < THR['sdnn_low']:
        flags['sdnn_flag'] = True
        flags['low_sdnn'] = True
        explanations.append(
            f"SDNN {sdnn:.1f} ms < {THR['sdnn_low']} ms → Severely depressed HRV, "
            f"autonomic dysfunction"
        )
        severity = min(1.0, (THR['sdnn_low'] - sdnn) / THR['sdnn_low'])
        risk_scores['sdnn_flag'] = WEIGHTS['sdnn_flag'] * (0.6 + 0.4 * severity)

    elif sdnn > THR['sdnn_high']:
        flags['sdnn_flag'] = True
        flags['high_sdnn'] = True
        explanations.append(
            f"SDNN {sdnn:.1f} ms > {THR['sdnn_high']} ms → Excessive beat-to-beat "
            f"variation, possible ectopic activity"
        )
        severity = min(1.0, (sdnn - THR['sdnn_high']) / 100.0)
        risk_scores['sdnn_flag'] = WEIGHTS['sdnn_flag'] * (0.5 + 0.5 * severity)

    else:
        flags['sdnn_flag'] = False
        risk_scores['sdnn_flag'] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # RULE 3 — RMSSD (short-term / vagal HRV)
    # ─────────────────────────────────────────────────────────────────────
    if rmssd > THR['rmssd_high']:
        flags['rmssd_flag'] = True
        flags['high_rmssd'] = True
        explanations.append(
            f"RMSSD {rmssd:.1f} ms > {THR['rmssd_high']} ms → Elevated parasympathetic "
            f"activity or ectopic beat influence"
        )
        severity = min(1.0, (rmssd - THR['rmssd_high']) / 100.0)
        risk_scores['rmssd_flag'] = WEIGHTS['rmssd_flag'] * (0.5 + 0.5 * severity)
    else:
        flags['rmssd_flag'] = False
        risk_scores['rmssd_flag'] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # RULE 4 — pNN50 (beat-to-beat irregularity — primary AF marker)
    # ─────────────────────────────────────────────────────────────────────
    if pnn50 > THR['pnn50_high']:
        flags['pnn50_flag'] = True
        flags['high_pnn50'] = True
        explanations.append(
            f"pNN50 {pnn50:.1f}% > {THR['pnn50_high']}% → Highly irregular RR succession, "
            f"Atrial Fibrillation pattern"
        )
        severity = min(1.0, (pnn50 - THR['pnn50_high']) / 50.0)
        risk_scores['pnn50_flag'] = WEIGHTS['pnn50_flag'] * (0.6 + 0.4 * severity)
    else:
        flags['pnn50_flag'] = False
        risk_scores['pnn50_flag'] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # RULE 5 — Poincaré SD1/SD2 ratio (geometric irregularity)
    # ─────────────────────────────────────────────────────────────────────
    if sd1sd2 > THR['sd1sd2_high']:
        flags['poincare_flag'] = True
        flags['poincare_chaotic'] = True
        explanations.append(
            f"SD1/SD2 ratio {sd1sd2:.3f} > {THR['sd1sd2_high']} → Chaotic Poincaré geometry, "
            f"loss of long-term rhythm coherence"
        )
        severity = min(1.0, (sd1sd2 - THR['sd1sd2_high']) / 0.5)
        risk_scores['poincare_flag'] = WEIGHTS['poincare_flag'] * (0.6 + 0.4 * severity)

    elif sd1sd2 < THR['sd1sd2_low']:
        flags['poincare_flag'] = True
        flags['poincare_rigid'] = True
        explanations.append(
            f"SD1/SD2 ratio {sd1sd2:.3f} < {THR['sd1sd2_low']} → Overly rigid rhythm, "
            f"autonomic suppression"
        )
        severity = min(1.0, (THR['sd1sd2_low'] - sd1sd2) / THR['sd1sd2_low'])
        risk_scores['poincare_flag'] = WEIGHTS['poincare_flag'] * (0.4 + 0.6 * severity)

    else:
        flags['poincare_flag'] = False
        risk_scores['poincare_flag'] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # RULE 6 — LF/HF ratio (autonomic balance — only if freq available)
    # ─────────────────────────────────────────────────────────────────────
    if lf_hf is not None:
        if lf_hf > THR['lfhf_high']:
            flags['lfhf_flag'] = True
            flags['sympathetic_dominance'] = True
            explanations.append(
                f"LF/HF ratio {lf_hf:.2f} > {THR['lfhf_high']} → Sympathetic dominance, "
                f"elevated cardiac stress state"
            )
            severity = min(1.0, (lf_hf - THR['lfhf_high']) / 3.0)
            risk_scores['lfhf_flag'] = WEIGHTS['lfhf_flag'] * (0.5 + 0.5 * severity)
        else:
            flags['lfhf_flag'] = False
            risk_scores['lfhf_flag'] = 0.0
    else:
        # Not enough beats for freq domain — don't penalize, just skip
        flags['lfhf_flag'] = False
        risk_scores['lfhf_flag'] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # RULE 7 — Sample Entropy (signal complexity — only if nonlinear avail)
    # ─────────────────────────────────────────────────────────────────────
    if sampen is not None and sampen > 0:
        if sampen > THR['sampen_high']:
            flags['sampen_flag'] = True
            flags['high_entropy'] = True
            explanations.append(
                f"SampEn {sampen:.3f} > {THR['sampen_high']} → High signal entropy, "
                f"complex irregular rhythm"
            )
            severity = min(1.0, (sampen - THR['sampen_high']) / 1.0)
            risk_scores['sampen_flag'] = WEIGHTS['sampen_flag'] * (0.5 + 0.5 * severity)
        else:
            flags['sampen_flag'] = False
            risk_scores['sampen_flag'] = 0.0
    else:
        flags['sampen_flag'] = False
        risk_scores['sampen_flag'] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # ARRHYTHMIA TYPE INFERENCE
    # Rules are checked in priority order — most specific first
    # ─────────────────────────────────────────────────────────────────────
    arrhythmia_type = _infer_arrhythmia_type(flags, hr, sdnn, rmssd, pnn50, sd1sd2)

    # ─────────────────────────────────────────────────────────────────────
    # FINAL RISK SCORE  (weighted sum, clamped to [0, 1])
    # ─────────────────────────────────────────────────────────────────────
    raw_score    = sum(risk_scores.values())
    phys_risk    = float(np.clip(raw_score, 0.0, 1.0))
    classification = 'abnormal' if phys_risk >= 0.3 else 'normal'

    return {
        'phys_risk_score' : round(phys_risk, 4),
        'classification'  : classification,
        'arrhythmia_type' : arrhythmia_type,
        'flags'           : flags,
        'explanation'     : ' | '.join(explanations) if explanations
                            else 'No physiological anomalies detected.',
        'risk_breakdown'  : {k: round(v, 4) for k, v in risk_scores.items()},
    }


# ─────────────────────────────────────────────────────────────
# PRIVATE HELPER — Arrhythmia type inference logic
# ─────────────────────────────────────────────────────────────

def _infer_arrhythmia_type(flags: dict, hr: float, sdnn: float,
                            rmssd: float, pnn50: float,
                            sd1sd2: float) -> str:
    """
    Infers the most likely arrhythmia type from flag combinations.
    Priority: most dangerous / specific conditions checked first.

    AAMI alignment:
        Suspected AF          → Class 1 (Supraventricular)
        Suspected PVC         → Class 2 (Ventricular ectopic)
        Sinus Tachycardia     → Class 0 (borderline) or Class 1
        Sinus Bradycardia     → Class 0 (borderline)
        Normal Sinus Rhythm   → Class 0
    """

    # AF: highly irregular rhythm + chaotic Poincaré + high pNN50
    if (flags.get('high_pnn50') and
            flags.get('poincare_chaotic') and
            flags.get('high_rmssd')):
        return "Suspected Atrial Fibrillation"

    # AF-like even without all three — two strong markers sufficient
    if flags.get('high_pnn50') and flags.get('poincare_chaotic'):
        return "Suspected Atrial Fibrillation"

    # PVC / ectopic: high SDNN + high RMSSD but not full AF pattern
    if flags.get('high_sdnn') and flags.get('high_rmssd') and not flags.get('high_pnn50'):
        return "Suspected PVC / Ectopic Activity"

    # Sinus Tachycardia: elevated HR + depressed HRV (low SDNN)
    if flags.get('tachycardia') and flags.get('low_sdnn'):
        return "Sinus Tachycardia with Depressed HRV"

    # Sinus Tachycardia: elevated HR alone
    if flags.get('tachycardia'):
        return "Sinus Tachycardia"

    # Sinus Bradycardia: low HR + depressed HRV
    if flags.get('bradycardia') and flags.get('low_sdnn'):
        return "Sinus Bradycardia with Autonomic Suppression"

    # Sinus Bradycardia: low HR alone
    if flags.get('bradycardia'):
        return "Sinus Bradycardia"

    # Autonomic suppression without rate abnormality
    if flags.get('poincare_rigid') and flags.get('low_sdnn'):
        return "Autonomic Suppression (Rigid Rhythm)"

    # Sympathetic stress without overt arrhythmia
    if flags.get('sympathetic_dominance') and flags.get('high_entropy'):
        return "Sympathetic Stress Pattern"

    # Any single flag with no clear pattern
    active_flags = [k for k, v in flags.items() if v is True]
    if active_flags:
        return "Nonspecific Rhythm Abnormality"

    return "Normal Sinus Rhythm"


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from hrv_features import compute_hrv_features

    def run_test(label, rr_array, expected_type_keyword):
        features = compute_hrv_features(rr_array)
        if features is None:
            print(f"  {label}: SKIPPED (insufficient beats)")
            return
        result = classify_beat(features)
        passed = expected_type_keyword.lower() in result['arrhythmia_type'].lower()
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  {status} — {label}")
        print(f"    Risk Score    : {result['phys_risk_score']}")
        print(f"    Class         : {result['classification']}")
        print(f"    Type          : {result['arrhythmia_type']}")
        print(f"    Explanation   : {result['explanation'][:120]}...")
        print(f"    Risk Breakdown: {result['risk_breakdown']}")

    print("\n" + "="*60)
    print("  ARRHYTHMIA CLASSIFIER — Unit Tests")
    print("="*60)

    # Test 1: Clean normal sinus rhythm
    rr_normal = np.random.normal(0.80, 0.025, 30)
    rr_normal = np.clip(rr_normal, 0.6, 1.1)
    run_test("Normal Sinus Rhythm", rr_normal, "normal")

    # Test 2: Atrial Fibrillation (highly irregular)
    rr_af = np.random.uniform(0.40, 1.10, 30)
    run_test("Atrial Fibrillation", rr_af, "fibrillation")

    # Test 3: Sinus Tachycardia (fast + low variability)
    rr_tachy = np.random.normal(0.50, 0.01, 30)   # ~120 bpm, very regular
    run_test("Sinus Tachycardia", rr_tachy, "tachycardia")

    # Test 4: Sinus Bradycardia (slow + low variability)
    rr_brady = np.random.normal(1.20, 0.01, 30)   # ~50 bpm
    run_test("Sinus Bradycardia", rr_brady, "bradycardia")

    # Test 5: PVC pattern (occasional large RR spikes)
    rr_pvc = np.random.normal(0.80, 0.02, 28).tolist()
    rr_pvc.insert(10, 0.45)   # short pre-ectopic interval
    rr_pvc.insert(11, 1.20)   # compensatory pause
    rr_pvc = np.array(rr_pvc)
    run_test("PVC / Ectopic Activity", rr_pvc, "ectopic")

    # Test 6: Real MIT-BIH record 100
    npy_path = "physiological_model/data/100_verified.npy"
    if os.path.exists(npy_path):
        print(f"\n  Running on full MIT-BIH Record 100...")
        data = np.load(npy_path, allow_pickle=True).item()
        rr_full   = data['rr_intervals']
        labels    = data['beat_labels']
        n_beats   = data['n_beats']

        results = []
        for i in range(1, n_beats):          # skip beat 0 (+marker)
            start  = max(0, i - 10)
            end    = min(len(rr_full), i + 10)
            window = rr_full[start:end]
            feat   = compute_hrv_features(window)
            if feat is None:
                results.append(None)
                continue
            res = classify_beat(feat)
            results.append(res)

        valid     = [(r, labels[i+1]) for i, r in enumerate(results) if r is not None]
        scores    = np.array([r['phys_risk_score'] for r, _ in valid])
        gt_labels = np.array([l for _, l in valid])

        print(f"    Beats processed   : {len(valid)}")
        print(f"    Mean risk (Normal beats, label=0) : "
              f"{np.mean(scores[gt_labels == 0]):.4f}")
        print(f"    Mean risk (Supra  beats, label=1) : "
              f"{np.mean(scores[gt_labels == 1]):.4f}"
              if np.any(gt_labels == 1) else "    No label=1 beats")
        print(f"    Mean risk (PVC    beats, label=2) : "
              f"{np.mean(scores[gt_labels == 2]):.4f}"
              if np.any(gt_labels == 2) else "    No label=2 beats")
        print(f"    Overall mean risk score           : {np.mean(scores):.4f}")

    print("\n" + "="*60)
    print("  ✓ Classifier tests complete.")
    print("="*60 + "\n")