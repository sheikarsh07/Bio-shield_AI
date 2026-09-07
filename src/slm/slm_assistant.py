import os
import requests
import json

class SLMAssistant:
    """
    Assistant interface using local Small Language Models (e.g., Phi-3 Mini, TinyLlama).
    Connects to Ollama backend with a robust mock fallback for presentation safety.
    """
    def __init__(self, model_name="phi3", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url

    def get_ollama_status(self):
        """Checks if local Ollama service is up and running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                return True, models
            return False, []
        except Exception:
            return False, []

    def generate_response(self, prompt, system_instruction="You are an expert wildlife conservation AI assistant."):
        """Generates response using Ollama, or falls back to a smart mock response if Ollama is offline."""
        is_active, available_models = self.get_ollama_status()
        
        # If Ollama is active, perform request
        if is_active:
            print(f"Ollama is active. Generating response via model: {self.model_name}")
            payload = {
                "model": self.model_name if self.model_name in available_models else (available_models[0] if available_models else self.model_name),
                "prompt": prompt,
                "system": system_instruction,
                "stream": False
            }
            try:
                response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=10)
                if response.status_code == 200:
                    return response.json().get("response", "")
            except Exception as e:
                print(f"Ollama request failed: {e}. Falling back to Rule-Based AI Engine...")
                
        # Smart rule-based simulation engine for offline/presentation mode
        print("Running mock conservation engine...")
        return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt):
        """Generates realistic generative responses based on details in the prompt to mock SLM outputs."""
        prompt_lower = prompt.lower()
        
        # Determine species
        detected_species = []
        for sp in ["tiger", "elephant", "dog", "horse", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel", "leopard", "deer", "wild_boar"]:
            if sp in prompt_lower or (sp + "s") in prompt_lower:
                detected_species.append(sp)
        
        # Check action context
        is_threat = any(w in prompt_lower for w in ["poaching", "threat", "gunshot", "snare", "hunter", "trap"])
        is_fire = any(w in prompt_lower for w in ["fire", "smoke", "burn", "flame"])
        is_summary = any(w in prompt_lower for w in ["summarize", "summary", "report"])

        response_header = "**[Local SLM Assistant (Offline Generative Fallback)]**\n\n"

        if is_fire:
            return response_header + (
                "⚠️ **Emergency Protocol: Forest Fire & Thermal Event**\n\n"
                f"Your query regarding fire/smoke activity has initiated standard operating protocols. "
                f"We recommend deploying drones to inspect the coordinates. All field staff in sectors affected "
                f"must prepare containment lines and wear protective gear immediately."
            )
        
        if is_threat:
            sp_str = f" involving {', '.join(detected_species)}" if detected_species else ""
            return response_header + (
                f"🚨 **Threat Response Log{sp_str}**\n\n"
                f"An anti-poaching or security threat has been queried. Directives:\n"
                f"1. Silently dispatch nearest ground patrol ranger team to query zone.\n"
                f"2. Establish perimeter check-posts at the surrounding sector exits.\n"
                f"3. Do not engage armed suspect groups directly; request base backup."
            )

        if is_summary:
            sp_str = f"including sightings of {', '.join(detected_species)}" if detected_species else "focusing on recent telemetry alerts"
            return response_header + (
                f"📋 **Ranger Operations Summary Report**\n\n"
                f"This generative report summarizes active logs, {sp_str}.\n"
                f"- **Sensor Status**: All IoT thresholds normal.\n"
                f"- **Observations**: Wildlife activity patterns remain consistent with seasonal migration routes.\n"
                f"- **Recommendation**: Continue routine patrols along perimeter buffer zones."
            )

        # Standard species recommendation response
        if detected_species:
            sp = detected_species[0]
            if sp == "tiger":
                return response_header + (
                    "🐯 **Species Profile: Bengal Tiger (Panthera tigris)** - *Endangered status*\n\n"
                    "**Management Directives**:\n"
                    "1. Ensure buffer zones between local forest reserves and village boundaries are strictly secured.\n"
                    "2. Avoid human activity in core sectors B-4 and C-12 where tiger movement is frequent.\n"
                    "3. Maintain active camera trap grids to map individual stripe patterns and track territory ranges."
                )
            elif sp == "elephant":
                return response_header + (
                    "🐘 **Species Profile: Elephant (Elephas maximus)** - *Vulnerable status*\n\n"
                    "**Management Directives**:\n"
                    "1. Keep migration pathways clear in Sector D-9.\n"
                    "2. Recommend farm owners use natural deterring methods like beehive fencing or chili grease cords.\n"
                    "3. Conduct periodic checks on waterholes during dry seasons to check water levels."
                )
            else:
                # Generative template for other classes (dog, horse, butterfly, chicken, cat, cow, sheep, spider, squirrel, etc.)
                return response_header + (
                    f"🐾 **Observation Advice: {sp.title()}**\n\n"
                    f"Your query references **{sp}**. Even though this species is not in our original target list, "
                    f"here is a generative mitigation log for this animal:\n"
                    f"- **Activity Level**: Moderate diurnal activity noted in surrounding transition zones.\n"
                    f"- **Recommendation**: Monitor environmental telemetry to ensure no human-wildlife encounters occur. "
                    f"Keep records updated in the central database."
                )
        
        # General response if no key matches
        return response_header + (
            "🌲 **Wildlife Assistant Operations Portal**\n\n"
            f"I analyzed your prompt: \"*{prompt}*\"\n\n"
            "Based on the conservation knowledge index, I recommend checking:\n"
            "- Threat levels in active sectors.\n"
            "- Current species telemetry metrics.\n"
            "- Anti-poaching team assignments.\n\n"
            "Please refine your query with specific terms like 'fire', 'poaching', 'tiger', 'elephant', 'butterfly', etc."
        )


    def generate_report(self, threat_category, details):
        """Generates an official incident report for conservation managers."""
        prompt = f"Write an official incident report for a {threat_category} event with details: {details}."
        system_instruction = "You are a lead conservation manager writing official incident logs."
        return self.generate_response(prompt, system_instruction)

if __name__ == "__main__":
    assistant = SLMAssistant()
    print(assistant.generate_response("What are the conservation recommendations for tigers?"))
