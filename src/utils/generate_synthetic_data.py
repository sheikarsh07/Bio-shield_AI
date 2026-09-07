import os
import csv
import json
import random
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_sensor_data(n_samples=1000):
    """Generates synthetic sensor data for wildlife and forest threat detection."""
    file_path = os.path.join(DATA_DIR, "sensor_data.csv")
    print(f"Generating sensor data at {file_path}...")

    # Headers: Temp, Humidity, Soil_Moisture, Smoke_Sensor (ppm), Acoustic_Freq (Hz), PIR_Motion (0/1), Label
    headers = [
        "temperature", "humidity", "soil_moisture", 
        "smoke_sensor", "acoustic_frequency", "pir_motion", "threat_label"
    ]

    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for _ in range(n_samples):
            # Select scenario to create realistic correlations
            scenario = random.choices(
                ["normal", "fire", "poaching", "unusual_behavior"],
                weights=[0.70, 0.10, 0.10, 0.10],
                k=1
            )[0]

            if scenario == "normal":
                temp = round(random.uniform(15.0, 32.0), 2)
                humidity = round(random.uniform(50.0, 85.0), 2)
                soil_moisture = round(random.uniform(40.0, 75.0), 2)
                smoke = round(random.uniform(10.0, 50.0), 2)
                acoustic = round(random.uniform(20.0, 1000.0), 2)  # bird chirping, leaves rustling
                pir = random.choice([0, 0, 0, 1])  # Occasional bird/small animal
                label = "Normal"

            elif scenario == "fire":
                temp = round(random.uniform(45.0, 75.0), 2)  # Hot
                humidity = round(random.uniform(10.0, 30.0), 2)  # Dry
                soil_moisture = round(random.uniform(5.0, 20.0), 2)  # Extremely dry
                smoke = round(random.uniform(250.0, 800.0), 2)  # High smoke
                acoustic = round(random.uniform(100.0, 2500.0), 2)  # crackling sounds
                pir = random.choice([0, 1])  # animals fleeing
                label = "Fire"

            elif scenario == "poaching":
                temp = round(random.uniform(18.0, 30.0), 2)
                humidity = round(random.uniform(60.0, 80.0), 2)
                soil_moisture = round(random.uniform(40.0, 65.0), 2)
                smoke = round(random.uniform(10.0, 60.0), 2)
                acoustic = round(random.uniform(2500.0, 8000.0), 2)  # Gunshot / truck engine / human voice
                pir = 1  # High motion detection
                label = "Poaching"

            elif scenario == "unusual_behavior":
                temp = round(random.uniform(20.0, 35.0), 2)
                humidity = round(random.uniform(40.0, 80.0), 2)
                soil_moisture = round(random.uniform(30.0, 60.0), 2)
                smoke = round(random.uniform(10.0, 50.0), 2)
                acoustic = round(random.uniform(3000.0, 6000.0), 2)  # distressed animal cries
                pir = 1  # High speed movement (running/stampeding)
                label = "Unusual Behavior"

            writer.writerow([temp, humidity, soil_moisture, smoke, acoustic, pir, label])

    print("Sensor data generated successfully.")

def generate_ranger_reports():
    """Generates synthetic textual reports from patrol rangers."""
    file_path = os.path.join(DATA_DIR, "ranger_reports.json")
    print(f"Generating ranger reports at {file_path}...")

    reports = []
    
    sighting_templates = [
        "During our patrol near sector {sector}, we spotted a healthy family of {species} {behavior}.",
        "Camera trap at station {station} captured {species} in its natural habitat, showing {behavior} patterns.",
        "Ranger {ranger} reported sighting a lone {species} near the watering hole. It was {behavior}."
    ]
    
    poaching_templates = [
        "Suspicious wire snare found and deactivated near sector {sector}. Evidence of footprints matching illegal hunters.",
        "Heard loud gunshot sounds from sector {sector} at approximately 22:00. Bullet shells found.",
        "Encountered campfire and abandoned camp gear belonging to suspected poachers near patrol line {station}.",
        "Fresh carcasses of {species} found with missing tusks/horns in sector {sector}. Alert level high."
    ]

    fire_templates = [
        "Spotted thick black smoke rising from sector {sector}. Initiated fire containment measures.",
        "Small ground fire detected near station {station} due to lightning strike. Team deployed to extinguish.",
        "Dry bush fire rapidly spreading across sector {sector}. Strong winds pushing it south. Emergency team notified."
    ]

    destruction_templates = [
        "Illegal tree logging activity discovered near sector {sector}. Several mature teak trees cut down.",
        "Heavy machinery tire tracks and land clearing detected in sector {sector}. Habitat destruction encroaching.",
        "Severe soil erosion and forest clearing observed near sector {sector} border line."
    ]

    species_list = ["tiger", "elephant", "deer", "leopard", "wild boar"]
    behaviors = ["grazing peacefully", "drinking water", "resting under shade", "running actively"]
    rangers = ["Kumar", "Sharma", "Singh", "Naidu", "Roy"]
    sectors = ["A-1", "B-4", "C-12", "D-9", "E-2"]
    stations = ["ST-101", "ST-102", "ST-204", "ST-305"]

    # Generate sighting reports
    for i in range(100):
        spec = random.choice(species_list)
        beh = random.choice(behaviors)
        sect = random.choice(sectors)
        stat = random.choice(stations)
        rang = random.choice(rangers)
        
        text = random.choice(sighting_templates).format(
            sector=sect, species=spec, behavior=beh, station=stat, ranger=rang
        )
        reports.append({"id": f"REP-S-{i:03d}", "text": text, "category": "Animal Sighting", "sentiment": "Positive"})

    # Generate poaching reports
    for i in range(50):
        sect = random.choice(sectors)
        stat = random.choice(stations)
        spec = random.choice(species_list)
        
        text = random.choice(poaching_templates).format(sector=sect, station=stat, species=spec)
        reports.append({"id": f"REP-P-{i:03d}", "text": text, "category": "Poaching", "sentiment": "Negative"})

    # Generate fire reports
    for i in range(30):
        sect = random.choice(sectors)
        stat = random.choice(stations)
        
        text = random.choice(fire_templates).format(sector=sect, station=stat)
        reports.append({"id": f"REP-F-{i:03d}", "text": text, "category": "Fire", "sentiment": "Negative"})

    # Generate habitat destruction reports
    for i in range(40):
        sect = random.choice(sectors)
        
        text = random.choice(destruction_templates).format(sector=sect)
        reports.append({"id": f"REP-H-{i:03d}", "text": text, "category": "Habitat Destruction", "sentiment": "Negative"})

    with open(file_path, "w") as f:
        json.dump(reports, f, indent=4)

    print("Ranger reports generated successfully.")

def generate_synthetic_images():
    """Generates small synthetic image arrays and structures folder names for DL classification."""
    # We will generate numpy arrays representing images for species & behavior training.
    # To keep files lightweight, we will create a helper dataset loader that dynamically 
    # yields synthetic images for testing, but we'll also save a few actual image files 
    # to data/raw/images so Streamlit can demo them visually.
    species = ["tiger", "elephant", "deer", "wild_boar"]
    behaviors = ["feeding", "running", "resting", "threat-action"]
    
    # Save a few dummy images to drive UI upload test cases
    try:
        import cv2
    except ImportError:
        # If cv2 is not installed yet, we can write simple matrices or use PIL
        cv2 = None

    print("Creating folder structure for species and behaviors...")
    for s in species:
        os.makedirs(os.path.join(IMAGE_DIR, s), exist_ok=True)
        # Create a sample image for the UI to showcase
        img_path = os.path.join(IMAGE_DIR, s, f"sample_{s}.jpg")
        if cv2:
            # Create a 224x224 RGB image with some colored box
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            # Add colored box & text to represent animal
            color = (0, 0, 255) if s == "tiger" else (255, 0, 0)
            cv2.rectangle(img, (50, 50), (170, 170), color, -1)
            cv2.putText(img, s.upper(), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(img_path, img)
        else:
            # Fallback to saving raw numpy bytes or dummy text if no cv2
            with open(img_path, "w") as f:
                f.write(f"DUMMY_IMAGE_DATA_FOR_{s}")

    # Generate behavior subdirs
    BEHAVIOR_DIR = os.path.join(DATA_DIR, "behavior_images")
    os.makedirs(BEHAVIOR_DIR, exist_ok=True)
    for b in behaviors:
        os.makedirs(os.path.join(BEHAVIOR_DIR, b), exist_ok=True)
        img_path = os.path.join(BEHAVIOR_DIR, b, f"sample_{b}.jpg")
        if cv2:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.rectangle(img, (40, 40), (180, 180), (0, 255, 0), -1)
            cv2.putText(img, b.upper(), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(img_path, img)
        else:
            with open(img_path, "w") as f:
                f.write(f"DUMMY_IMAGE_DATA_FOR_{b}")

    print("Synthetic folders and test images created.")

def generate_knowledge_base():
    """Generates the RAG wildlife knowledge document."""
    file_path = os.path.join(DATA_DIR, "wildlife_kb.txt")
    print(f"Generating wildlife knowledge base at {file_path}...")
    
    kb_content = """# Wildlife Conservation and Forest Protection Manual

## 1. Threat Management Protocol
- **Forest Fire Alert**: Triggered when ambient temperature exceeds 40°C, humidity drops below 30%, and smoke detectors register > 200 ppm. Conservation teams must dispatch drone monitoring units immediately to evaluate sector boundary firelines.
- **Poaching Threat Detection**: Indicated by motion sensors detecting anomalous movement during nocturnal hours (20:00 - 05:00) alongside high-frequency acoustic signals matching gunshot patterns (> 3000 Hz or high amplitude spikes). Alert regional rangers to deploy blockades in Sectors A-1, B-4, and C-12.
- **Illegal Logging Operations**: Characterized by localized forest canopy reduction and acoustic frequency spikes from chainsaw engines (typically 120-250 Hz range). Ground patrols should secure adjacent logging roads.

## 2. Species Conservation Guide
- **Tiger (Panthera tigris)**:
  - Endangered status. Needs dense vegetative cover, constant water access, and high prey density.
  - Active patrol area: Sector B-4 and C-12.
  - Recommended action: Establish artificial water holes, install camera traps every 500 meters, and strictly control access gates.
- **Asian Elephant (Elephas maximus)**:
  - Endangered status. Large migratory corridors are essential.
  - Conflict mitigation: Install bio-fences (e.g., bee-hives, chili ropes) along farmland boundaries in Sector D-9 to prevent crop raiding.
- **Spotted Deer (Axis axis)**:
  - Vulnerable baseline food source for large carnivores. Monitor herd health and foraging behavior.

## 3. Ranger Safety and Standard Operating Procedures (SOP)
- All field rangers must carry satellite transponders and keep their primary transceiver set to channel 4.
- In case of a wildfire, evacuate upwind. Do not cross firelines unless wearing specialized flame-retardant gear.
- In case of active poaching contact: Keep silent, record GPS coordinates, transmit silent alerts to headquarters, and do not engage suspect groups without back-up.
"""
    with open(file_path, "w") as f:
        f.write(kb_content)
        
    print("Knowledge base generated successfully.")

if __name__ == "__main__":
    generate_sensor_data()
    generate_ranger_reports()
    generate_synthetic_images()
    generate_knowledge_base()
    print("All synthetic data successfully set up!")
