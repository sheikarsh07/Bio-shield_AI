import os
import re
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_PATH = os.path.join(BASE_DIR, "data", "raw", "ranger_reports.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class NLPManager:
    """End-to-end NLP processing for ranger reports and wildlife documents."""
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.categories = ["Animal Sighting", "Poaching", "Fire", "Habitat Destruction"]
        self.load_models()

    def preprocess_text(self, text):
        """Preprocesses text: cleaning, lowercasing, and basic tokenization."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize (whitespace split)
        tokens = text.split()
        
        # Simple lemmatization dictionary for wildlife terms
        lemma_dict = {
            "spotted": "spot", "spotting": "spot", "sightings": "sighting",
            "sighted": "sighting", "captured": "capture", "capturing": "capture",
            "footprints": "footprint", "animals": "animal", "tigers": "tiger",
            "elephants": "elephant", "deers": "deer", "ran": "run", "running": "run",
            "grazed": "graze", "grazing": "graze", "poachers": "poacher"
        }
        
        lemmatized_tokens = [lemma_dict.get(token, token) for token in tokens]
        return " ".join(lemmatized_tokens)

    def extract_entities(self, text):
        """Extracts Named Entities: Rangers, Species, Sectors, and Stations using regex/lexicon."""
        # Species mapping
        species_list = ["tiger", "elephant", "deer", "leopard", "wild boar", "boar", "carnivore", "predator"]
        # Rangers list
        rangers_list = ["kumar", "sharma", "singh", "naidu", "roy", "patel"]
        
        # Regex patterns
        sector_pattern = r'[sS]ector\s+([A-Z]-\d+|[A-Z]\d+)'
        station_pattern = r'[sS]tation\s+(ST-\d+|ST\d+)'
        
        found_species = []
        found_rangers = []
        
        cleaned_text_lower = text.lower()
        
        for sp in species_list:
            if sp in cleaned_text_lower:
                found_species.append(sp.capitalize())
        
        for rg in rangers_list:
            if rg in cleaned_text_lower:
                found_rangers.append(rg.capitalize())
                
        sectors = re.findall(sector_pattern, text)
        stations = re.findall(station_pattern, text)
        
        return {
            "Rangers": list(set(found_rangers)),
            "Species": list(set(found_species)),
            "Sectors": list(set(sectors)),
            "Stations": list(set(stations))
        }

    def analyze_sentiment(self, text):
        """Predicts positive/negative/neutral sentiment from the report content."""
        # Clean rule-based lexicon mapping
        positive_words = ["healthy", "peaceful", "active", "grazing", "spotted", "sighting", "safe", "stable"]
        negative_words = ["poaching", "suspect", "fire", "smoke", "carcass", "shot", "dead", "killing", "illegal", "logging", "destruction", "encroach", "snare"]
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        else:
            return "Neutral"

    def summarize_text(self, text, num_sentences=2):
        """Performs basic sentence-scoring extractive summarization."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= num_sentences:
            return text
            
        # Tokenize and score words based on frequency
        words = re.sub(r'[^a-zA-Z\s]', '', text).lower().split()
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
            
        # Score sentences
        sentence_scores = {}
        for sent in sentences:
            score = 0
            for word in sent.lower().split():
                if word in word_freq:
                    score += word_freq[word]
            sentence_scores[sent] = score
            
        # Get top sentences
        sorted_sents = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        summary_sentences = sorted_sents[:num_sentences]
        # Maintain original sentence order
        summary = [s for s in sentences if s in summary_sentences]
        return " ".join(summary)

    def train_classifier(self):
        """Trains TF-IDF + LogisticRegression on synthetic ranger reports."""
        if not os.path.exists(REPORTS_PATH):
            raise FileNotFoundError(f"Ranger reports not found at {REPORTS_PATH}. Run the data generator first.")
            
        with open(REPORTS_PATH, "r") as f:
            data = json.load(f)
            
        texts = [item["text"] for item in data]
        labels = [item["category"] for item in data]
        
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(t) for t in texts]
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        X = self.vectorizer.fit_transform(preprocessed_texts)
        
        self.classifier = LogisticRegression(class_weight="balanced")
        self.classifier.fit(X, labels)
        
        # Save vectorizer and classifier
        with open(os.path.join(MODELS_DIR, "nlp_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(MODELS_DIR, "nlp_classifier.pkl"), "wb") as f:
            pickle.dump(self.classifier, f)
            
        print("NLP Classifier model trained and saved successfully.")

    def load_models(self):
        """Loads vectorizer and classifier models from disk if they exist."""
        vec_path = os.path.join(MODELS_DIR, "nlp_vectorizer.pkl")
        cls_path = os.path.join(MODELS_DIR, "nlp_classifier.pkl")
        
        if os.path.exists(vec_path) and os.path.exists(cls_path):
            with open(vec_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(cls_path, "rb") as f:
                self.classifier = pickle.load(f)
        else:
            print("NLP Models not found. Call train_classifier() first.")

    def classify_text(self, text):
        """Classifies ranger report into threat categories."""
        if not self.classifier or not self.vectorizer:
            # Fallback mock classifier if model files not trained
            text_lower = text.lower()
            if "fire" in text_lower or "smoke" in text_lower:
                return "Fire", 0.90
            elif "poach" in text_lower or "gunshot" in text_lower or "snare" in text_lower:
                return "Poaching", 0.95
            elif "logging" in text_lower or "tree" in text_lower:
                return "Habitat Destruction", 0.88
            else:
                return "Animal Sighting", 0.85
                
        prep = self.preprocess_text(text)
        features = self.vectorizer.transform([prep])
        pred_label = self.classifier.predict(features)[0]
        probs = self.classifier.predict_proba(features)[0]
        confidence = float(np.max(probs))
        
        return pred_label, confidence

if __name__ == "__main__":
    manager = NLPManager()
    manager.train_classifier()
    
    # Test pipeline
    test_report = "Suspicious vehicle spotted near Station ST-204 in Sector B-4. Possible poachers footprints nearby."
    label, conf = manager.classify_text(test_report)
    sentiment = manager.analyze_sentiment(test_report)
    entities = manager.extract_entities(test_report)
    print(f"\nTest classification: {label} (confidence: {conf:.2f})")
    print(f"Sentiment: {sentiment}")
    print(f"Entities: {entities}")
