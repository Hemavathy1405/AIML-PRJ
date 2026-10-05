"""
Unit Tests for Evidence Retrieval Module
"""

import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval.retriever import EvidenceRetriever

def test_retriever():
    retriever = EvidenceRetriever()
    test_chunks = [
        {"chunk_id": "c1", "title": "Photosynthesis", "domain": "Biology", "text": "Plants convert sunlight, carbon dioxide, and water into glucose and oxygen through chlorophyll pigment."},
        {"chunk_id": "c2", "title": "TCP Handshake", "domain": "Networking", "text": "The Transmission Control Protocol establishes connections via a three-way handshake involving SYN, SYN-ACK, and ACK packets."}
    ]
    retriever.add_chunks(test_chunks)

    # Search for biology
    hits = retriever.retrieve("How do plants synthesize glucose from sunlight?", top_k=1)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "c1"
    assert "chlorophyll" in hits[0]["text"]

    # Search for networking
    hits_net = retriever.retrieve("What packets are used in TCP three-way handshake?", top_k=1)
    assert len(hits_net) == 1
    assert hits_net[0]["chunk_id"] == "c2"

if __name__ == "__main__":
    test_retriever()
    print("All retrieval tests passed successfully!")
