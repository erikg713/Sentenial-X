def simulate_encryption(files: list[str]) -> list[str]:
    # Safe simulation: Just append .encrypted to file names
    return [f + ".encrypted" for f in files]
