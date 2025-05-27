def extract_logs(log_path: str, keywords: list) -> list:
    """
    Extracts lines from logs that match specific keywords (case-insensitive).
    """
    if not os.path.exists(log_path):
        return []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    matches = [line for line in lines if any(kw.lower() in line.lower() for kw in keywords)]
    return matches