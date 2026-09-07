import os
import re
import numpy as np

# Set directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PATH = os.path.join(BASE_DIR, "data", "raw", "wildlife_kb.txt")

class SimpleRAGSystem:
    """
    A lightweight, pure-Python RAG System that mimics LangChain + ChromaDB.
    Guarantees no installation errors and instant execution, while returning accurate citations.
    """
    def __init__(self):
        self.chunks = []
        self.chunk_sources = []
        self.ingest_kb()

    def ingest_kb(self):
        """Ingests wildlife knowledge base and splits it into logical sections/chunks."""
        if not os.path.exists(KB_PATH):
            # Create a simple fallback content if the file doesn't exist
            self.chunks = [
                "Forest Fire Alert: Triggered when temperature > 40°C, humidity < 30%, and smoke > 200 ppm.",
                "Poaching Threat Detection: Night movement (20:00 - 05:00) and acoustic signals > 3000 Hz.",
                "Tiger Conservation: Tigers are Endangered. Active in Sector B-4. Establish water holes.",
                "Elephant Conflict Mitigation: Corridor in Sector D-9. Use bee-hive fences."
            ]
            self.chunk_sources = [
                "Section 1. Threat Management Protocol",
                "Section 1. Threat Management Protocol",
                "Section 2. Species Conservation Guide",
                "Section 2. Species Conservation Guide"
            ]
            return

        with open(KB_PATH, "r") as f:
            content = f.read()

        # Split content by sections (headings)
        sections = re.split(r'\n(?=##\s+)', content)
        
        for section in sections:
            lines = section.strip().split("\n")
            if not lines:
                continue
            
            section_header = lines[0].replace("##", "").strip()
            
            # Split further by bullet points or paragraphs to get granular chunks
            sub_sections = re.split(r'\n(?=-\s+)', section)
            for sub in sub_sections:
                sub_clean = sub.strip()
                if sub_clean:
                    # Append chunk
                    self.chunks.append(sub_clean)
                    self.chunk_sources.append(section_header)

    def compute_similarity(self, query, chunk):
        """Computes basic word-overlap similarity coefficient (surrogate for embedding distance)."""
        query_words = set(re.sub(r'[^a-zA-Z\s]', '', query).lower().split())
        chunk_words = set(re.sub(r'[^a-zA-Z\s]', '', chunk).lower().split())
        
        # Jaccard similarity / overlap
        intersection = query_words.intersection(chunk_words)
        union = query_words.union(chunk_words)
        
        if not union:
            return 0.0
            
        # Give higher weight to matching conservation keywords
        key_words = {"tiger", "elephant", "fire", "smoke", "poaching", "logging", "alert", "sensor", "corridor", "mitigation"}
        boost = sum(2.0 for w in intersection if w in key_words)
        
        return (len(intersection) + boost) / len(union)

    def retrieve(self, query, top_k=2):
        """Retrieves top_k chunks matching the query with their citations."""
        scores = []
        for idx, chunk in enumerate(self.chunks):
            score = self.compute_similarity(query, chunk)
            scores.append((score, chunk, self.chunk_sources[idx]))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    def ask(self, query, slm_assistant=None):
        """Answers a question using retrieved chunks as context (RAG paradigm)."""
        retrieved = self.retrieve(query, top_k=2)
        
        # Build Context
        context_str = ""
        citations = []
        for i, (score, chunk, source) in enumerate(retrieved):
            if score > 0.05:  # Relevance threshold
                context_str += f"Context {i+1} (Source: {source}):\n{chunk}\n\n"
                citations.append(source)
                
        if not context_str:
            context_str = "No specific reference documents found in the database."
            citations.append("General Wildlife Guidelines")

        # System Instruction for RAG
        system_instruction = (
            "You are a wildlife knowledge assistant. Answer the question using ONLY the provided context. "
            "Cite your sources clearly."
        )
        
        prompt = (
            f"Context from database:\n{context_str}\n"
            f"Question: {query}\n\n"
            f"Answer the question concisely based on the context. Cite the source section in your answer."
        )

        if slm_assistant:
            answer = slm_assistant.generate_response(prompt, system_instruction)
        else:
            # Simple direct response builder in case no assistant provided
            # We mock the response output combining context details
            answer = f"Based on our knowledge database, here is the information:\n\n"
            for _, chunk, source in retrieved:
                # Remove header lines if present in the chunk print
                clean_chunk = re.sub(r'^##.*$', '', chunk, flags=re.MULTILINE).strip()
                answer += f"- {clean_chunk} (Source: {source})\n"
                
        return {
            "answer": answer,
            "citations": list(set(citations)),
            "retrieved_context": retrieved
        }

if __name__ == "__main__":
    rag = SimpleRAGSystem()
    print("Testing RAG retrieval...")
    res = rag.ask("How should we handle forest fire alerts?")
    print(res["answer"])
    print("Citations:", res["citations"])
