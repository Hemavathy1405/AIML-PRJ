"""
Dataset Generation & Labelling Pipeline
Course: 24DSP501 - Technical Communication in Data Science
Step 1: Clean and structure question-answer pairs and reference evidence chunks.
Step 4: Construct hallucination labels (factual, partial, contradicted variants).
Step 5: Normalize and perform stratified train/test split.
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import sys
# Add parent dir to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    REFERENCE_CORPUS_FILE, CURATED_QA_FILE, DATASET_LABELED_FILE,
    TRAIN_FEATURES_FILE, TEST_FEATURES_FILE, RANDOM_SEED, TEST_SPLIT_RATIO,
    ALL_FEATURES
)
from src.features.feature_extractor import HallucinationFeatureExtractor

# Seed for deterministic generation
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CURATED_CORPUS = [
    {
        "chunk_id": "bio_01",
        "title": "CRISPR-Cas9 Gene Editing Mechanism",
        "domain": "Biomedicine",
        "text": "CRISPR-Cas9 is a biochemical system derived from bacterial immune defenses that enables targeted RNA-guided DNA endonuclease cleavage. The Cas9 endonuclease is directed by a single guide RNA (sgRNA) containing a 20-nucleotide sequence that matches the genomic target directly adjacent to a Protospacer Adjacent Motif (PAM), typically 5'-NGG-3' in Streptococcus pyogenes. Once double-strand breaks (DSBs) are created, host cellular repair mechanisms—non-homologous end joining (NHEJ) or homology-directed repair (HDR)—resolve the lesion, enabling targeted knockout or gene insertion."
    },
    {
        "chunk_id": "bio_02",
        "title": "mRNA Vaccines for Viral Pathogens",
        "domain": "Biomedicine",
        "text": "mRNA vaccines deliver synthetic nucleoside-modified messenger RNA encapsulated in lipid nanoparticles (LNPs) to host dendritic cells and myocytes. The mRNA instructs ribosomes to synthesize the viral spike glycoprotein in its prefusion stabilized conformation, prompting both humoral neutralising antibody production and cytotoxic CD8+ T-cell responses without introducing active viral particles. The synthetic mRNA degrades naturally within host cells within several days."
    },
    {
        "chunk_id": "cs_01",
        "title": "Transformer Attention Architecture",
        "domain": "Computer Science",
        "text": "The Transformer architecture, introduced by Vaswani et al. in 2017, eliminates recurrent and convolutional layers in favor of stacked Multi-Head Self-Attention and position-wise Feed-Forward networks. Scaled dot-product attention computes Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V. This allows parallel computation over all sequence tokens and models long-range dependencies in O(1) sequential operations, though standard self-attention has O(N^2) computational and memory complexity with respect to sequence length N."
    },
    {
        "chunk_id": "cs_02",
        "title": "RAFT Consensus Algorithm",
        "domain": "Distributed Systems",
        "text": "Raft is a distributed consensus algorithm designed by Ongaro and Ousterhout for state machine replication across a cluster of servers. Raft divides consensus into three modular subproblems: leader election, log replication, and safety. A cluster has at most one leader at any given term; if a leader fails or heartbeats cease within an election timeout, follower nodes transition to candidate state and request votes. Raft guarantees that committed log entries will survive any future leader election."
    },
    {
        "chunk_id": "astro_01",
        "title": "James Webb Space Telescope (JWST) Optics",
        "domain": "Astronomy",
        "text": "The James Webb Space Telescope (JWST) features a 6.5-meter diameter primary mirror composed of 18 hexagonal beryllium segments coated with a 100-nanometer layer of pure gold and a protective silicon dioxide overcoat. Designed primarily for infrared astronomy across wavelengths from 0.6 to 28 micrometers, JWST operates at the second Sun-Earth Lagrange point (L2), approximately 1.5 million kilometers from Earth, protected by a five-layer Kapton sunshield maintaining operating temperatures below 50 Kelvin."
    },
    {
        "chunk_id": "astro_02",
        "title": "Event Horizon Telescope Black Hole Imaging",
        "domain": "Astronomy",
        "text": "In April 2019, the Event Horizon Telescope (EHT) collaboration published the first direct visual evidence of a supermassive black hole at the core of the galaxy Messier 87 (M87*), followed in 2022 by Sagittarius A* in the Milky Way. Using Very Long Baseline Interferometry (VLBI) at 1.3 mm wavelength, radio telescopes across the globe synchronized using hydrogen maser atomic clocks synthesized a virtual Earth-sized telescope aperture, achieving an angular resolution of 20 micro-arcseconds."
    },
    {
        "chunk_id": "phys_01",
        "title": "Quantum Entanglement & Bell Inequalities",
        "domain": "Physics",
        "text": "Quantum entanglement occurs when pairs or groups of particles interact such that the quantum state of each particle cannot be described independently of the others, regardless of spatial separation. John Stewart Bell proved in 1964 that local hidden variable theories impose strict upper bounds on statistical correlations (Bell inequalities), which quantum mechanics violates. Aspect, Clauser, and Zeilinger won the 2022 Nobel Prize in Physics for loophole-free experimental confirmations of Bell violation."
    },
    {
        "chunk_id": "env_01",
        "title": "Ocean Acidification Chemistry",
        "domain": "Environmental Science",
        "text": "Ocean acidification is caused by marine absorption of anthropogenic atmospheric carbon dioxide (CO2). When CO2 dissolves into seawater, it forms carbonic acid (H2CO3), which dissociates into hydrogen ions (H+) and bicarbonate ions (HCO3-). The excess hydrogen ions react with carbonate ions (CO3 2-) to form further bicarbonate, thereby lowering the marine carbonate saturation state (Omega) and impairing calcium carbonate shell formation in organisms such as pteropods, corals, and shellfish."
    },
    {
        "chunk_id": "hist_01",
        "title": "The Rosetta Stone Decipherment",
        "domain": "History & Archaeology",
        "text": "Discovered in 1799 near the town of Rashid (Rosetta) in Egypt by French soldiers during Napoleon's expedition, the Rosetta Stone is a granodiorite stele inscribed with a decree issued at Memphis in 196 BCE on behalf of King Ptolemy V. The decree is written in three distinct scripts: Ancient Egyptian hieroglyphs (sacred script), Egyptian Demotic (administrative script), and Ancient Greek (court script). Jean-François Champollion announced the decipherment of the hieroglyphs in 1822."
    },
    {
        "chunk_id": "fin_01",
        "title": "Basel III Banking Capital Framework",
        "domain": "Finance & Economics",
        "text": "Basel III is an international regulatory accord developed by the Basel Committee on Banking Supervision (BCBS) following the 2007-2008 global financial crisis. It established mandatory minimum capital adequacy ratios, requiring banks to maintain Common Equity Tier 1 (CET1) of at least 4.5% of risk-weighted assets, plus a 2.5% capital conservation buffer. Basel III also introduced the Liquidity Coverage Ratio (LCR) requiring banks to hold high-quality liquid assets sufficient to withstand a 30-day stressed outflow."
    },
    {
        "chunk_id": "cs_03",
        "title": "Zero-Knowledge Proofs (zk-SNARKs)",
        "domain": "Cryptography",
        "text": "A Zero-Knowledge Proof allows a prover to demonstrate to a verifier that a statement is mathematically true without disclosing any underlying secret information beyond the validity of the statement itself. zk-SNARK stands for Zero-Knowledge Succinct Non-Interactive Argument of Knowledge. They rely on pairing-friendly elliptic curves and polynomial arithmetic to produce cryptographic proofs that are only hundreds of bytes long and verifiable in milliseconds."
    },
    {
        "chunk_id": "bio_03",
        "title": "Alzheimer's Disease Amyloid Cascade Hypothesis",
        "domain": "Biomedicine",
        "text": "The amyloid cascade hypothesis posits that the deposition of amyloid-beta (Abeta) peptides in the brain parenchyma is the primary causative event in Alzheimer's disease pathogenesis. Abeta peptides, particularly the 42-amino acid isoform (Abeta-42), are cleaved from amyloid precursor protein (APP) by beta-secretase and gamma-secretase enzymes. Oligomerization and plaque accumulation induce hyperphosphorylation of tau protein, forming neurofibrillary tangles that trigger synaptic loss and dementia."
    },
    {
        "chunk_id": "phys_02",
        "title": "Superconductivity & Meissner Effect",
        "domain": "Physics",
        "text": "Superconductivity is a quantum state of matter characterized by zero electrical resistance and the complete expulsion of magnetic flux fields, an effect discovered by Walther Meissner and Robert Ochsenfeld in 1933. According to BCS theory (Bardeen-Cooper-Schrieffer), below a critical temperature (Tc), electrons overcome Coulomb repulsion via lattice phonon interactions to form bound Cooper pairs. High-temperature cuprate superconductors discovered in 1986 operate above the boiling point of liquid nitrogen (77 K)."
    },
    {
        "chunk_id": "hist_02",
        "title": "The Manhattan Project and Trinity Test",
        "domain": "History of Science",
        "text": "The Manhattan Project was a clandestine research and development undertaking during World War II, led by the United States with support from the United Kingdom and Canada, directed by Major General Leslie Groves and theoretical physicist J. Robert Oppenheimer at Los Alamos Laboratory. On July 16, 1945, the project conducted the first nuclear detonation, code-named the Trinity test, in the Jornada del Muerto desert near Alamogordo, New Mexico, detonating an implosion-type plutonium device known as 'The Gadget'."
    },
    {
        "chunk_id": "env_02",
        "title": "Atmospheric Ozone Depletion and Montreal Protocol",
        "domain": "Environmental Science",
        "text": "Stratospheric ozone depletion is primarily driven by halogenated compounds, particularly chlorofluorocarbons (CFCs) and halons. In the stratosphere, solar ultraviolet radiation photolyzes CFC molecules, releasing free chlorine radicals. A single chlorine atom can catalytically destroy more than 100,000 ozone (O3) molecules. In 1987, the international community signed the Montreal Protocol on Substances that Deplete the Ozone Layer, mandating the phaseout of CFC production, resulting in the documented recovery of the ozone layer."
    }
]

# Diverse questions and response variations
QA_TEMPLATES = [
    # 1. CRISPR
    {
        "chunk_id": "bio_01",
        "question": "How does the CRISPR-Cas9 system identify its DNA target and create genomic edits?",
        "factual": "CRISPR-Cas9 uses a single guide RNA containing a 20-nucleotide sequence that matches the target genomic locus directly adjacent to a PAM sequence, typically 5'-NGG-3'. The Cas9 endonuclease generates double-strand breaks, which are subsequently resolved by cellular repair mechanisms like NHEJ or HDR.",
        "partial": "CRISPR-Cas9 recognizes genomic targets using a guide RNA and cuts the DNA adjacent to a PAM motif. It also actively repairs the DNA strand itself using an intrinsic polymerase sub-domain before releasing the cleaved sequence.",
        "hallucinated": "CRISPR-Cas9 relies on magnetic nanoparticle resonance to locate target chromatin fibers without requiring guide RNAs or PAM motifs. Once located, it vaporizes the entire chromosome via infrared thermal decay.",
        "hallucinated_taxonomy": "Unfounded Fabrication (Entirely Invented Fact)"
    },
    # 2. mRNA Vaccines
    {
        "chunk_id": "bio_02",
        "question": "What is the biological mechanism of action for mRNA vaccines?",
        "factual": "mRNA vaccines encapsulate synthetic nucleoside-modified mRNA inside lipid nanoparticles to enter host cells, directing ribosomes to produce the prefusion stabilized spike glycoprotein and triggering immune responses.",
        "partial": "mRNA vaccines deliver genetic instructions in lipid nanoparticles to produce spike proteins. However, the synthetic mRNA permanently integrates into host genomic DNA inside the nucleus within 48 hours.",
        "hallucinated": "mRNA vaccines contain live-attenuated bacterial spores that replicate inside red blood cells to produce antibody crystals that circulate perpetually in the bloodstream.",
        "hallucinated_taxonomy": "Factual Error (Wrong Entity/Number/Date)"
    },
    # 3. Transformers
    {
        "chunk_id": "cs_01",
        "question": "Why did the Transformer architecture replace recurrence, and what is its computational complexity?",
        "factual": "The Transformer replaced recurrence with multi-head self-attention, enabling parallel computation across all sequence tokens, but standard self-attention has O(N^2) complexity with respect to sequence length N.",
        "partial": "The Transformer uses self-attention instead of recurrence to achieve O(1) sequential steps. It achieves linear O(N) memory complexity across all vanilla self-attention implementations without approximation.",
        "hallucinated": "Transformers rely on bi-directional Markov decision trees that execute sequentially across recurrent hidden states, requiring exactly O(2^N) exponential time complexity to parse text.",
        "hallucinated_taxonomy": "Contradiction to Evidence"
    },
    # 4. RAFT
    {
        "chunk_id": "cs_02",
        "question": "How does the Raft consensus algorithm handle leader election and node failures?",
        "factual": "In Raft, if a follower node ceases to receive heartbeat signals within an election timeout, it transitions to candidate state, increments its term, and requests votes from peers to elect a new leader.",
        "partial": "Raft uses randomized election timeouts to trigger leader elections when heartbeats fail. If a split vote occurs, the algorithm immediately terminates all nodes and reboots the physical hardware servers.",
        "hallucinated": "Raft achieves consensus without leader election by requiring all distributed nodes to broadcast quantum telemetry to a centralized NASA master clock every 10 microseconds.",
        "hallucinated_taxonomy": "Unfounded Fabrication (Entirely Invented Fact)"
    },
    # 5. JWST
    {
        "chunk_id": "astro_01",
        "question": "Where is the James Webb Space Telescope positioned and how is its primary mirror constructed?",
        "factual": "JWST is positioned at the Sun-Earth L2 Lagrange point roughly 1.5 million kilometers from Earth and features a 6.5-meter primary mirror made of 18 gold-coated hexagonal beryllium segments.",
        "partial": "JWST is located at L2 Lagrange point with an 18-segment beryllium mirror, but it operates primarily in ultraviolet wavelengths below 0.1 micrometers and is cooled using liquid hydrogen thrusters.",
        "hallucinated": "The James Webb Space Telescope orbits Earth in Low Earth Orbit at 400 km altitude and carries a monolithic 20-meter glass mirror manufactured by Galileo Galilei in Florence.",
        "hallucinated_taxonomy": "Factual Error (Wrong Entity/Number/Date)"
    },
    # 6. EHT
    {
        "chunk_id": "astro_02",
        "question": "How did the Event Horizon Telescope capture the image of the M87* black hole?",
        "factual": "The EHT used Very Long Baseline Interferometry at 1.3 mm wavelength with atomic-clock synchronized radio observatories worldwide to create an Earth-sized virtual aperture.",
        "partial": "The EHT used VLBI with synchronized atomic clocks across international radio observatories, but the final image was captured using visible light optical lenses mounted on orbital cubesats.",
        "hallucinated": "The EHT sent a robotic probe directly inside the event horizon of M87* which transmitted high-resolution 4K optical camera footage back to Earth over submarine fiber cables.",
        "hallucinated_taxonomy": "Unfounded Fabrication (Entirely Invented Fact)"
    },
    # 7. Bell Inequalities
    {
        "chunk_id": "phys_01",
        "question": "What is the physical significance of Bell inequalities in quantum mechanics?",
        "factual": "Bell inequalities establish an upper bound on statistical correlations for any local hidden variable theory, and their experimental violation confirms that quantum entanglement cannot be explained by local realism.",
        "partial": "Bell inequalities show the bounds of local realism. However, Aspect's experiments in 2022 conclusively proved that Einstein's local hidden variables are completely correct and quantum entanglement is an illusion.",
        "hallucinated": "Bell inequalities are thermodynamic equations proving that energy can be freely created from nothing whenever two electrons bounce off a vacuum chamber wall.",
        "hallucinated_taxonomy": "Contradiction to Evidence"
    },
    # 8. Ocean Acidification
    {
        "chunk_id": "env_01",
        "question": "What chemical reactions occur when excess atmospheric CO2 dissolves into ocean waters?",
        "factual": "Dissolved CO2 forms carbonic acid, which dissociates into hydrogen and bicarbonate ions; the extra hydrogen ions consume carbonate ions, reducing the saturation state needed for calcifying marine life.",
        "partial": "CO2 dissolves into seawater to form carbonic acid which decreases pH. This chemical process miraculously fortifies coral skeletons by increasing carbonate ion availability tenfold.",
        "hallucinated": "Atmospheric CO2 dissolution converts ocean water into pure sulfur trioxide, completely boiling the world's oceans and causing the seas to turn into solid granite by 2030.",
        "hallucinated_taxonomy": "Reasoning / Logic Error"
    },
    # 9. Rosetta Stone
    {
        "chunk_id": "hist_01",
        "question": "What three scripts are inscribed on the Rosetta Stone and who deciphered them?",
        "factual": "The Rosetta Stone features Egyptian hieroglyphs, Egyptian Demotic script, and Ancient Greek, and its hieroglyphic script was successfully deciphered by Jean-François Champollion in 1822.",
        "partial": "The Rosetta Stone was written in Ancient Greek, Demotic, and Egyptian hieroglyphs. It was discovered by Napoleon himself in 1945 during the German occupation of Cairo.",
        "hallucinated": "The Rosetta Stone contains Sumerian cuneiform, binary code, and modern Spanish, and was translated by William Shakespeare in 1605 using an electronic typewriter.",
        "hallucinated_taxonomy": "Factual Error (Wrong Entity/Number/Date)"
    },
    # 10. Basel III
    {
        "chunk_id": "fin_01",
        "question": "What capital and liquidity requirements were established under the Basel III framework?",
        "factual": "Basel III requires banks to hold Common Equity Tier 1 (CET1) of at least 4.5% of risk-weighted assets plus a 2.5% conservation buffer, along with a 30-day Liquidity Coverage Ratio.",
        "partial": "Basel III mandates a CET1 ratio of 4.5% and a capital conservation buffer, but it completely eliminated all liquidity oversight and allows banks to operate with zero cash reserves.",
        "hallucinated": "Basel III abolished paper money worldwide and decreed that all international debt must be denominated exclusively in unregulated speculative doge-tokens without capital reserves.",
        "hallucinated_taxonomy": "Unfounded Fabrication (Entirely Invented Fact)"
    },
    # 11. zk-SNARKs
    {
        "chunk_id": "cs_03",
        "question": "What does zk-SNARK stand for and what are its key mathematical properties?",
        "factual": "zk-SNARK stands for Zero-Knowledge Succinct Non-Interactive Argument of Knowledge, generating tiny cryptographic proofs verifiable in milliseconds without revealing the prover's secret.",
        "partial": "zk-SNARKs produce succinct proofs without revealing secret data, but they require the verifier to download the entire multi-gigabyte secret key database before checking any single statement.",
        "hallucinated": "zk-SNARK stands for Zero Kinetic Sonic Nuclear Aerial Rocket Kinetic, which is a jet engine turbine system that compresses air using blockchain token combustion.",
        "hallucinated_taxonomy": "Factual Error (Wrong Entity/Number/Date)"
    },
    # 12. Alzheimer's Amyloid
    {
        "chunk_id": "bio_03",
        "question": "According to the amyloid cascade hypothesis, how do Abeta peptides trigger neurodegeneration?",
        "factual": "Abeta peptides cleaved from APP oligomerize into plaques in brain parenchyma, inducing hyperphosphorylation of tau protein into tangles that disrupt synapses and cause neurodegeneration.",
        "partial": "Abeta peptides accumulate into brain plaques. However, the hypothesis states that amyloid plaques actually regenerate brain tissue and reverse memory loss in late-stage dementia patients.",
        "hallucinated": "Alzheimer's is caused exclusively by microscopic silicon microchips injected by dental fillings that emit high-frequency ultrasonic static into the cerebral spinal fluid.",
        "hallucinated_taxonomy": "Unfounded Fabrication (Entirely Invented Fact)"
    },
    # 13. Superconductivity
    {
        "chunk_id": "phys_02",
        "question": "What is the Meissner effect and what microscopic mechanism explains BCS superconductivity?",
        "factual": "The Meissner effect is the complete expulsion of magnetic flux fields from a superconductor, explained in BCS theory by electrons forming bound Cooper pairs via lattice phonon interactions below Tc.",
        "partial": "The Meissner effect expels magnetic fields below a critical temperature. In BCS theory, this occurs because electrons fuse into radioactive uranium atoms inside the lattice structure.",
        "hallucinated": "The Meissner effect causes superconductors to burst into spontaneous combustion whenever a magnet approaches, proving that electricity does not exist at low temperatures.",
        "hallucinated_taxonomy": "Contradiction to Evidence"
    },
    # 14. Manhattan Project
    {
        "chunk_id": "hist_02",
        "question": "Who led the Manhattan Project and where was the first nuclear test conducted?",
        "factual": "The Manhattan Project was led by Leslie Groves and J. Robert Oppenheimer, and the first nuclear detonation (the Trinity test) was conducted in the desert near Alamogordo, New Mexico on July 16, 1945.",
        "partial": "Leslie Groves and J. Robert Oppenheimer led the Manhattan Project in New Mexico. However, the Trinity test device was a solar-powered fusion engine that caused zero blast wave or radiation.",
        "hallucinated": "The Manhattan Project was an architectural project directed by Leonardo da Vinci in 1812 to construct high-rise skyscrapers in downtown Manhattan using titanium alloy.",
        "hallucinated_taxonomy": "Factual Error (Wrong Entity/Number/Date)"
    },
    # 15. Ozone Depletion
    {
        "chunk_id": "env_02",
        "question": "How do CFCs destroy stratospheric ozone and what international treaty addressed this issue?",
        "factual": "Solar UV radiation photolyzes CFCs to release free chlorine radicals that catalytically destroy ozone molecules; the 1987 Montreal Protocol mandated the phaseout of CFC production.",
        "partial": "CFCs release chlorine atoms that destroy ozone molecules, but the Montreal Protocol mandated that global industries increase CFC manufacturing by 500% to fill the ozone hole.",
        "hallucinated": "Ozone depletion is caused by high-altitude flying eagles flapping their wings too fast, which was outlawed by the United Nations Geneva Convention in 1492.",
        "hallucinated_taxonomy": "Unfounded Fabrication (Entirely Invented Fact)"
    }
]

def generate_multi_samples(base_text: str, variant_type: str) -> list:
    """Produces multi-sample generations to evaluate semantic consistency and entropy."""
    if variant_type == "factual":
        return [
            f"As stated in verified documentation, {base_text}",
            f"The evidence demonstrates that {base_text}",
            f"In summary, {base_text}"
        ]
    elif variant_type == "partial":
        return [
            f"While the core finding holds, {base_text}",
            f"According to some reports, {base_text} although other sources disagree.",
            f"Preliminary data suggests {base_text}"
        ]
    else: # hallucinated
        # Even when hallucinating, the model might repeat its fabricated claim confidently
        return [
            f"Official sources confirm that {base_text}",
            f"It is well known that {base_text}",
            f"Without question, {base_text}"
        ]

def build_full_dataset(num_replications: int = 14) -> pd.DataFrame:
    """
    Expands the curated questions across domains to create a robust dataset of
    ~420+ balanced records (Factual, Partial, Hallucinated) with diverse linguistic variations.
    """
    corpus_map = {c["chunk_id"]: c for c in CURATED_CORPUS}
    records = []
    
    extractor = HallucinationFeatureExtractor()

    print("[*] Generating dataset records and computing 7 hallucination signals...")

    for rep in range(num_replications):
        for item in QA_TEMPLATES:
            chunk = corpus_map[item["chunk_id"]]
            evidence_text = chunk["text"]
            q = item["question"]

            # Add minor natural lexical variation per replication
            prefixes = [
                "", "Specifically, ", "In technical terms, ", "To be precise, ",
                "According to current analysis, ", "As established, ", "Empirical findings show that "
            ]
            prefix = prefixes[rep % len(prefixes)]

            # 1. FACTUAL VARIANT (Label = 0: LOW risk)
            resp_factual = f"{prefix}{item['factual']}"
            samples_fact = generate_multi_samples(item['factual'], "factual")
            # Low uncertainty / high confidence
            tok_fact = [-0.03 - (rep * 0.002), -0.05, -0.02, -0.04]
            feats_fact = extractor.extract_features(
                reference_evidence=evidence_text,
                response=resp_factual,
                sampled_responses=samples_fact,
                token_logprobs=tok_fact
            )
            records.append({
                "record_id": f"rec_{len(records):04d}",
                "domain": chunk["domain"],
                "chunk_id": chunk["chunk_id"],
                "question": q,
                "reference_evidence": evidence_text,
                "response": resp_factual,
                "sampled_responses": json.dumps(samples_fact),
                "variant_type": "factual",
                "risk_label": 0, # LOW
                "risk_category": "LOW",
                "is_hallucination": 0,
                "error_taxonomy": "None (Grounded & Faithful)",
                **feats_fact
            })

            # 2. PARTIAL VARIANT (Label = 1: MEDIUM risk)
            resp_partial = f"{prefix}{item['partial']}"
            samples_part = generate_multi_samples(item['partial'], "partial")
            # Moderate uncertainty
            tok_part = [-0.45 - (rep * 0.01), -0.72, -0.38, -0.65]
            feats_part = extractor.extract_features(
                reference_evidence=evidence_text,
                response=resp_partial,
                sampled_responses=samples_part,
                token_logprobs=tok_part
            )
            records.append({
                "record_id": f"rec_{len(records):04d}",
                "domain": chunk["domain"],
                "chunk_id": chunk["chunk_id"],
                "question": q,
                "reference_evidence": evidence_text,
                "response": resp_partial,
                "sampled_responses": json.dumps(samples_part),
                "variant_type": "partial",
                "risk_label": 1, # MEDIUM
                "risk_category": "MEDIUM",
                "is_hallucination": 1,
                "error_taxonomy": "Overconfidence in Extrapolation",
                **feats_part
            })

            # 3. HALLUCINATED VARIANT (Label = 2: HIGH risk)
            resp_halluc = f"{prefix}{item['hallucinated']}"
            samples_halluc = generate_multi_samples(item['hallucinated'], "hallucinated")
            # Dangerously confident (low uncertainty / confident-sounding), high contradiction!
            tok_halluc = [-0.04, -0.02, -0.05, -0.03]
            feats_halluc = extractor.extract_features(
                reference_evidence=evidence_text,
                response=resp_halluc,
                sampled_responses=samples_halluc,
                token_logprobs=tok_halluc
            )
            records.append({
                "record_id": f"rec_{len(records):04d}",
                "domain": chunk["domain"],
                "chunk_id": chunk["chunk_id"],
                "question": q,
                "reference_evidence": evidence_text,
                "response": resp_halluc,
                "sampled_responses": json.dumps(samples_halluc),
                "variant_type": "hallucinated",
                "risk_label": 2, # HIGH
                "risk_category": "HIGH",
                "is_hallucination": 1,
                "error_taxonomy": item["hallucinated_taxonomy"],
                **feats_halluc
            })

    df = pd.DataFrame(records)
    return df

def main():
    print("==================================================================")
    print("AI Hallucination Risk Prediction: Dataset Generation & Prep")
    print("==================================================================")

    # 1. Save raw reference corpus
    with open(REFERENCE_CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(CURATED_CORPUS, f, indent=2)
    print(f"[+] Saved {len(CURATED_CORPUS)} reference knowledge chunks to: {REFERENCE_CORPUS_FILE}")

    # 2. Save curated QA pairs
    with open(CURATED_QA_FILE, "w", encoding="utf-8") as f:
        json.dump(QA_TEMPLATES, f, indent=2)
    print(f"[+] Saved {len(QA_TEMPLATES)} curated base QA templates to: {CURATED_QA_FILE}")

    # 3. Generate full labeled dataset with all 7 extracted features
    df = build_full_dataset(num_replications=10) # 15 items * 3 variants * 10 = 450 balanced samples
    df.to_csv(DATASET_LABELED_FILE, index=False)
    print(f"[+] Generated {len(df)} total labeled records with 7 extracted features.")
    print(f"    Class distribution:\n{df['risk_category'].value_counts().to_string()}")
    print(f"[+] Saved full labeled dataset to: {DATASET_LABELED_FILE}")

    # 4. Grouped / stratified split (Step 5): hold out entire questions so the
    # same template does not leak from train into test via lexical prefixes.
    try:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED
        )
        train_idx, test_idx = next(
            splitter.split(df, df["risk_label"], groups=df["question"])
        )
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        # Keep class balance reasonably even; fall back if a split is degenerate
        if train_df["risk_label"].nunique() < 3 or test_df["risk_label"].nunique() < 3:
            raise ValueError("Degenerate grouped split")
    except Exception:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SPLIT_RATIO,
            stratify=df["risk_label"],
            random_state=RANDOM_SEED
        )

    train_df.to_csv(TRAIN_FEATURES_FILE, index=False)
    test_df.to_csv(TEST_FEATURES_FILE, index=False)
    print(f"[+] Stratified train split ({len(train_df)} samples) saved to: {TRAIN_FEATURES_FILE}")
    print(f"[+] Stratified test split ({len(test_df)} samples) saved to: {TEST_FEATURES_FILE}")
    print("==================================================================")

if __name__ == "__main__":
    main()
