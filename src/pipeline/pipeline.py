from checkBlast import blast_check


def final_check(sequence: str) -> str:

    result = blast_check(sequence)
    if not result:
        return
    
    # pathoLM

    print(f'\nFinal sequence: {sequence}\n')

if __name__ == "__main__":
    #generate sequence
    sequence = "AGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATATGCTTAAGCATGAAGTGTGGCGAACGGGTGAGTAACGCGTGAGCAACCTGCCCTTAGGACTGAGAGATGGCGACGGGCGGTCCAATCCGAACTGAGCCACGCCGCAAGGCCAAAACTCTATCAGTAGCGGGCGACTGAGAGGTTGTCGACGTTACTCTGAGAGGGCGAAAAA"
    
    final_check(sequence)