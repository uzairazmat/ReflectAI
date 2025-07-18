def classify_fatigue(ear, fatigue_detector):
    if ear is None:
        return {"status": "no face", "severity": 0}

    severity = 0
    if ear < fatigue_detector.ear_threshold:
        severity = min(
            fatigue_detector.fatigue_counter / fatigue_detector.fatigue_frame_threshold,
            1.0
        )

    return {
        "status": "not fatigue" if severity == 0 else
        "fully fatigue" if severity >= 1.0 else
        "mild fatigue",
        "severity": round(severity, 2)
    }