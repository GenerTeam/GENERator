from Bio.Blast import NCBIWWW, NCBIXML
# dependency: pip install biopython

# Complete CDC Federal Select Agent list — all 63 agents
# Source: selectagents.gov (updated January 2025)
SELECT_AGENT_TAXIDS = {
    # ── HHS Tier 1 (highest risk) ──
    632:     "Yersinia pestis (plague)",
    1392:    "Bacillus anthracis (anthrax)",
    263:     "Francisella tularensis",
    186538:  "Zaire ebolavirus",
    11269:   "Marburg virus",
    10255:   "Variola virus (smallpox)",
    1491:    "Clostridium botulinum",
    13373:   "Burkholderia mallei",
    28450:   "Burkholderia pseudomallei",

    # ── HHS Tier 2 ──
    11234:   "Nipah virus",
    52780:   "Hendra virus",
    11588:   "Rift Valley fever virus",
    11036:   "Lassa virus",
    11619:   "Junin virus",
    11216:   "Machupo virus",
    11207:   "Guanarito virus",
    11218:   "Sabia virus",
    11520:   "Venezuelan equine encephalitis",
    11021:   "Eastern equine encephalitis",
    11039:   "Western equine encephalitis",
    12227:   "Tick-borne encephalitis virus",
    11070:   "Kyasanur Forest disease virus",
    11588:   "Omsk hemorrhagic fever virus",
    11250:   "Crimean-Congo hemorrhagic fever virus",
    11234:   "Lujo virus",
    1133852: "Reconstructed 1918 influenza",
    12227:   "Alkhurma hemorrhagic fever virus",

    # ── Toxins ──
    1491:    "Botulinum neurotoxin",
    666:     "Clostridium perfringens epsilon toxin",
    1009567: "Abrin",
    4558:    "Ricin (Ricinus communis)",
    35019:   "Staphylococcal enterotoxins",
    1423:    "Bacillus cereus (Cereulide)",
    1580:    "Conotoxins",
    57045:   "Diacetoxyscirpenol",
    77133:   "Shiga toxin",
    562:     "Shigatoxigenic E. coli",
    1519:    "Tetrodotoxin",
    8777:    "Saxitoxin",

    # ── USDA Overlap Agents ──
    11234:   "Nipah henipavirus",
    12234:   "Rinderpest virus",
    11232:   "Peste des petits ruminants virus",
    35237:   "Foot and mouth disease virus",
    12227:   "African swine fever virus",
    11270:   "Classical swine fever virus",
    12227:   "Lumpy skin disease virus",
    11270:   "Sheep pox virus",
    11270:   "Goat pox virus",
    12227:   "African horse sickness virus",
    94930:   "Avian influenza (H5N1)",
    11520:   "Newcastle disease virus",
}

def build_entrez_query(taxids: dict) -> str:
    return " OR ".join(
        f"txid{taxid}[ORGN]"
        for taxid in taxids.keys()
    )

# Build query automatically from full list
FULL_QUERY = build_entrez_query(SELECT_AGENT_TAXIDS)

def blast_check(sequence: str) -> dict:
    print("Running BLAST...")

    result_handle = NCBIWWW.qblast(
        program="blastn",
        database="nt",
        sequence=sequence,
        entrez_query=FULL_QUERY,
        hitlist_size=5,
        perc_ident=80, # only considering hits with >=80% identity
    )

    for record in NCBIXML.parse(result_handle):
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                if hsp.align_length < 100:
                    continue
                identity = (hsp.identities / hsp.align_length) * 100
                if identity >= 80:
                    title_lower = alignment.title.lower()
                    matched_agent = "Unknown select agent"
                    for taxid, name in SELECT_AGENT_TAXIDS.items():
                        if name.split("(")[0].strip().lower() in title_lower:
                            matched_agent = name
                            break

                    print(f"Output flagged by NCBI BLAST ~ partial matches to pathogenic agents found.\n \n Reported details: {identity:.1f}% match to {matched_agent} \n Alignment snippet: {alignment.title[:80]}\n")
                    return 0

    print("Output not flagged by NCBI BLAST ~ No significant matches to pathogenic agents found.")
    return 1