VALID_BASES = {"A", "T", "C", "G"}


def is_valid_dna(seq: str) -> bool:
    return all(base in VALID_BASES for base in seq)


def extract_species(desc: str) -> str | None:
    tokens = desc.split()
    if len(tokens) < 3:
        return None
    return f"{tokens[1]} {tokens[2]}".lower()