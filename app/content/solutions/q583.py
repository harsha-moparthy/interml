def q583(real_train, synthetic, heldout_patients):
    """Validate and assemble augmentation manifests. Rows are dicts with id,patient;
    synthetic rows additionally list source_patients and
    pseudo_labeler_training_patients. Reject duplicate IDs or ANY held-out patient in
    training/provenance. Return shallow copies of real-only and augmented manifests.
    This split/provenance core does not train a DDPM or assert augmentation benefit.
    """
    blocked = set(heldout_patients)
    seen = set()
    for row in list(real_train) + list(synthetic):
        if row["id"] in seen:
            raise ValueError("duplicate ID")
        seen.add(row["id"])
        provenance = (
            {row["patient"]}
            | set(row.get("source_patients", []))
            | set(row.get("pseudo_labeler_training_patients", []))
        )
        if provenance & blocked:
            raise ValueError("held-out patient leakage")
    baseline = [dict(r) for r in real_train]
    return (baseline, baseline + [dict(r) for r in synthetic])
