import os
import sys
import json
from datetime import datetime

# Add project root to python path to avoid import errors
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.rag_system import SimpleRAGSystem
from src.slm.slm_assistant import SLMAssistant

# Set directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_PATH = os.path.join(BASE_DIR, "data", "processed", "agentic_decision_log.json")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

class AgenticManager:
    """
    Autonomous Agentic AI Conservation Manager.
    Orchestrates sensors, threat detection, RAG lookups, actions, and human approval.
    """
    def __init__(self):
        self.rag = SimpleRAGSystem()
        self.slm = SLMAssistant()
        self.pending_actions = {}
        self.load_pending_actions()

    def load_pending_actions(self):
        """Loads pending actions from memory if any (simulated memory)."""
        # Memory is simply kept in-memory for the demo session.
        self.pending_actions = {}

    def log_decision(self, event_type, details, action_taken, status="Completed"):
        """Logs agentic activities and decisions to persistent JSON log."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "action_taken": action_taken,
            "status": status
        }
        
        history = []
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []
                
        history.append(log_entry)
        
        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=4)

    def evaluate_sensor_readings(self, temp, humidity, soil_moisture, smoke, acoustic, pir):
        """
        Agentic reasoning loop.
        Analyzes inputs, queries knowledge base via RAG, and recommends mitigation actions.
        """
        status = "Normal"
        recommended_action = "Maintain baseline surveillance and keep sensor arrays active."
        requires_approval = False
        action_id = f"ACT-{int(datetime.now().timestamp())}"
        
        # Reasoner rules (mimicking ML model output checks)
        if temp > 40.0 and smoke > 200.0:
            status = "Fire Threat Detected"
            requires_approval = True
            # Query RAG
            rag_res = self.rag.ask("How should we handle forest fire alerts?", self.slm)
            recommended_action = (
                f"ACTIVATE DRONE PATROLS & WILDFIRE TEAM DEPLOYMENT. "
                f"Protocol Details: {rag_res['answer']}"
            )
            
        elif acoustic > 3000.0 and pir == 1:
            status = "Poaching Threat Detected"
            requires_approval = True
            rag_res = self.rag.ask("How should we handle poaching threats?", self.slm)
            recommended_action = (
                f"DISPATCH INTERCEPT PATROL TO GPS STATION COORDINATES. "
                f"Protocol Details: {rag_res['answer']}"
            )
            
        elif temp > 35.0 and soil_moisture < 20.0:
            status = "Environmental Alert (Drought)"
            requires_approval = False
            recommended_action = "Activate automated water sprinklers at tiger corridor zones to dry soil."
            
        # Log decision
        if requires_approval:
            self.pending_actions[action_id] = {
                "status": status,
                "recommended_action": recommended_action,
                "timestamp": datetime.now().isoformat(),
                "inputs": {
                    "temp": temp, "humidity": humidity, "soil_moisture": soil_moisture,
                    "smoke": smoke, "acoustic": acoustic, "pir": pir
                }
            }
            self.log_decision(status, f"Sensors: Temp={temp}, Smoke={smoke}, Acoustic={acoustic}, PIR={pir}", 
                              f"Queued for human approval: {recommended_action}", "Pending Approval")
            return {
                "action_id": action_id,
                "status": status,
                "action": recommended_action,
                "requires_approval": True
            }
        else:
            self.log_decision(status, f"Sensors: Temp={temp}, Moisture={soil_moisture}", 
                              recommended_action, "Executed")
            return {
                "action_id": None,
                "status": status,
                "action": recommended_action,
                "requires_approval": False
            }

    def process_human_approval(self, action_id, approved=True):
        """Executes or cancels pending action based on human-in-the-loop input."""
        if action_id not in self.pending_actions:
            return {"status": "Error", "message": f"Action ID {action_id} not found."}
            
        action = self.pending_actions.pop(action_id)
        action_text = action["recommended_action"]
        threat = action["status"]
        
        status_text = "Approved & Executed" if approved else "Rejected & Cancelled"
        self.log_decision(
            f"Human-in-the-Loop {status_text}", 
            f"Action ID: {action_id} for {threat}", 
            f"Operator resolved: {action_text}", 
            status_text
        )
        return {
            "status": "Success",
            "message": f"Action {action_id} successfully {status_text}."
        }

    def get_logs(self):
        """Returns decision logs from disk."""
        if not os.path.exists(LOG_PATH):
            return []
        try:
            with open(LOG_PATH, "r") as f:
                return json.load(f)[::-1]  # Return newest first
        except Exception:
            return []

if __name__ == "__main__":
    agent = AgenticManager()
    # Simulate fire
    res = agent.evaluate_sensor_readings(temp=45.0, humidity=20.0, soil_moisture=10.0, smoke=300.0, acoustic=150.0, pir=0)
    print("Agent Response:", res)
    if res["requires_approval"]:
        print("Approving pending action...")
        print(agent.process_human_approval(res["action_id"], approved=True))
