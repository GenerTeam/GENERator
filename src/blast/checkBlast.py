from Bio.Blast import NCBIWWW, NCBIXML

# Tell BLAST to ONLY search within these dangerous organisms
# Any match = dangerous by definition
DANGEROUS_TAXIDS_QUERY = (
    "txid632[ORGN] OR "      # Yersinia pestis (plague)
    "txid1392[ORGN] OR "     # Bacillus anthracis (anthrax)
    "txid263[ORGN] OR "      # Francisella tularensis
    "txid186538[ORGN] OR "   # Ebola virus
    "txid11269[ORGN] OR "    # Marburg virus
    "txid1491[ORGN] OR "     # Clostridium botulinum
    "txid13373[ORGN] OR "    # Burkholderia mallei
    "txid28450[ORGN]"        # Burkholderia pseudomallei
)

def blast_check(sequence: str) -> dict:
    """
    Search ONLY within dangerous pathogen sequences.
    Any match = dangerous. No filtering needed.
    """
    print("Running BLAST against dangerous pathogen database...")
    print("(30-60 seconds)")

    result_handle = NCBIWWW.qblast(
        program="blastn",
        database="nt",
        sequence=sequence,
        entrez_query=DANGEROUS_TAXIDS_QUERY,  # ← restricts to dangerous only
        hitlist_size=5,
        perc_ident=80,
    )

    for record in NCBIXML.parse(result_handle):
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                identity = (hsp.identities / hsp.align_length) * 100
                if identity >= 80:
                    # No name checking needed
                    # If it matched, it's dangerous — that's all we know
                    return {
                        "flagged": True,
                        "identity": f"{identity:.1f}%",
                        "match": alignment.title[:80],
                        "reason": (
                            f"Matches dangerous pathogen sequence "
                            f"at {identity:.1f}% identity"
                        )
                    }
    return {"flagged": False}

if __name__ == "__main__":
    example_sequence = "ATCGATCGATCGATCG"  # Example DNA sequence
    result = blast_check(example_sequence)
    print("Result:", result)