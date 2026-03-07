from checkBlast import blast_check


def final_check(sequence: str) -> str:

    result = blast_check(sequence)
    if not result:
        return
    
    # pathoLM

if __name__ == "__main__":
    # example sequence for now
    final_check("AGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATATGCTTAAGCATGAAGTGTGGCGAACGGGTGAGTAACGCGTGAGCAACCTGCCCTTAGGACTGAGAGATGGCGACGGGCGGTCCAATCCGAACTGAGCCACGCCGCAAGGCCAAAACTCTATCAGTAGCGGGCGACTGAGAGGTTGTCGACGTTACTCTGAGAGGGCGAAAAA")