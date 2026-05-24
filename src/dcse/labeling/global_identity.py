import parasail


def global_identity(seq1: str, seq2: str) -> float:
    result = parasail.nw_trace(seq1, seq2, 10, 1, parasail.dnafull)

    a1 = result.traceback.query
    a2 = result.traceback.ref

    identical = 0
    aln_cols = 0

    for x, y in zip(a1, a2):
        if x == "-" and y == "-":
            continue
        aln_cols += 1
        if x == y and x != "-":
            identical += 1

    return round(100.0 * identical / aln_cols, 2) if aln_cols else 0.0