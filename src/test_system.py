import os
import sys
import unittest
import numpy as np

# Add project root to python path to avoid import errors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test compilation and basic interface compliance
from src.dl.image_classifier import Predictor
from src.nlp.nlp_pipeline import NLPManager
from src.rag.rag_system import SimpleRAGSystem
from src.agent.agent_manager import AgenticManager

class TestBiodiversitySystem(unittest.TestCase):
    def setUp(self):
        self.nlp = NLPManager()
        self.rag = SimpleRAGSystem()
        self.agent = AgenticManager()
        
    def test_nlp_pipeline(self):
        """Verifies text preprocessing, NER, sentiment evaluation and classification."""
        text = "Ranger Sharma spotted a tiger resting peacefully near Sector B-4."
        
        # Test entities
        entities = self.nlp.extract_entities(text)
        self.assertIn("Sharma", entities["Rangers"])
        self.assertIn("Tiger", entities["Species"])
        self.assertIn("B-4", entities["Sectors"])
        
        # Test sentiment
        sentiment = self.nlp.analyze_sentiment(text)
        self.assertEqual(sentiment, "Positive")
        
        # Test classification (fallback/trained)
        label, conf = self.nlp.classify_text(text)
        self.assertIn(label, self.nlp.categories)
        self.assertTrue(0.0 <= conf <= 1.0)
        
    def test_rag_system(self):
        """Verifies document semantic chunking and retrieval citations."""
        query = "How should we handle forest fire alerts?"
        res = self.rag.ask(query)
        
        self.assertIsNotNone(res["answer"])
        self.assertTrue(len(res["citations"]) > 0)
        
    def test_agent_manager(self):
        """Verifies agentic telemetry assessment and approval mechanisms."""
        # Test Normal telemetries (Auto executed)
        res_normal = self.agent.evaluate_sensor_readings(
            temp=22.0, humidity=70.0, soil_moisture=50.0, smoke=15.0, acoustic=150.0, pir=0
        )
        self.assertFalse(res_normal["requires_approval"])
        self.assertEqual(res_normal["status"], "Normal")
        
        # Test Fire threat telemetries (Human-in-the-loop)
        res_fire = self.agent.evaluate_sensor_readings(
            temp=55.0, humidity=12.0, soil_moisture=8.0, smoke=350.0, acoustic=100.0, pir=0
        )
        self.assertTrue(res_fire["requires_approval"])
        self.assertIsNotNone(res_fire["action_id"])
        
        # Test resolving human approval
        action_id = res_fire["action_id"]
        res_approval = self.agent.process_human_approval(action_id, approved=True)
        self.assertEqual(res_approval["status"], "Success")

if __name__ == "__main__":
    unittest.main()
