import os
import tempfile
import subprocess
from Bio import SeqIO


def run_blast(query_rec, subject_rec):
    query_file = subject_file = result_file = db_prefix = None
    hsps = []

    try:
        with tempfile.NamedTemporaryFile("w", delete=False) as qf:
            SeqIO.write(query_rec, qf, "fasta")
            query_file = qf.name

        with tempfile.NamedTemporaryFile("w", delete=False) as sf:
            SeqIO.write(subject_rec, sf, "fasta")
            subject_file = sf.name

        db_prefix = subject_file + "_db"

        subprocess.run(
            f"makeblastdb -in {subject_file} -dbtype nucl -out {db_prefix}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        result_file = tempfile.NamedTemporaryFile(delete=False).name

        subprocess.run(
            f"blastn -query {query_file} -db {db_prefix} "
            f"-task megablast -dust yes "
            f"-outfmt '6 qseqid sseqid qlen slen qstart qend sstart send pident' "
            f"-out {result_file}",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with open(result_file) as rf:
            for line in rf:
                p = line.strip().split("\t")
                if not p:
                    continue

                hsps.append({
                    "qlen": int(p[2]),
                    "slen": int(p[3]),
                    "qstart": int(p[4]),
                    "qend": int(p[5]),
                    "sstart": int(p[6]),
                    "send": int(p[7]),
                    "hsp_identity": float(p[8]),
                })

    finally:
        for f in [query_file, subject_file, result_file]:
            if f and os.path.exists(f):
                os.remove(f)

        if db_prefix:
            for ext in [".nhr", ".nin", ".nsq"]:
                try:
                    os.remove(db_prefix + ext)
                except FileNotFoundError:
                    pass

    return hsps