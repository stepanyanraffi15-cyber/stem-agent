def read_line(filename: str, lineno: int) -> str:
    try:
        with open(filename) as f:
            lines = f.readlines()
        return lines[lineno]
    except (OSError, IndexError):
        return ""
