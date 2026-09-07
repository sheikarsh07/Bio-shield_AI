-- ==========================================
-- SYSTEM DATABASE SCHEMA: BIODIVERSITY CONSERVATION
-- Target Engines: PostgreSQL (Production) / SQLite (Local Simulation)
-- ==========================================

-- 1. Sensors Table
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id VARCHAR(50) PRIMARY KEY,
    location_sector VARCHAR(10) NOT NULL,
    latitude DECIMAL(9,6),
    longitude DECIMAL(9,6),
    status VARCHAR(20) DEFAULT 'Active'
);

-- 2. Sensor Readings Table (Time Series)
CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(50) REFERENCES sensors(sensor_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    temperature DECIMAL(5,2),
    humidity DECIMAL(5,2),
    soil_moisture DECIMAL(5,2),
    smoke_sensor DECIMAL(6,2),
    acoustic_frequency DECIMAL(8,2),
    pir_motion INTEGER CHECK (pir_motion IN (0, 1)),
    threat_label VARCHAR(50)
);

-- 3. Camera Trap Detections Table
CREATE TABLE IF NOT EXISTS camera_trap_detections (
    detection_id SERIAL PRIMARY KEY,
    camera_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_url VARCHAR(255),
    predicted_species VARCHAR(50),
    species_confidence DECIMAL(4,3),
    predicted_behavior VARCHAR(50),
    behavior_confidence DECIMAL(4,3),
    verification_status VARCHAR(20) DEFAULT 'Pending' -- Pending, Confirmed, False Positive
);

-- 4. Ranger Patrol Reports Table
CREATE TABLE IF NOT EXISTS ranger_reports (
    report_id VARCHAR(50) PRIMARY KEY,
    ranger_name VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    report_text TEXT,
    classified_category VARCHAR(50),
    sentiment VARCHAR(20),
    extracted_entities JSONB -- Holds Rangers, Species, Sectors, Stations
);

-- 5. Alerts Table (Dispatch)
CREATE TABLE IF NOT EXISTS alerts (
    alert_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity VARCHAR(20), -- Informational, Warning, Critical
    threat_type VARCHAR(50),
    description TEXT,
    resolved VARCHAR(10) DEFAULT 'No',
    resolved_by VARCHAR(50),
    resolution_time TIMESTAMP
);

-- 6. Agentic Decisions Table
CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trigger_event_type VARCHAR(50),
    sensor_details TEXT,
    recommended_action TEXT,
    approval_status VARCHAR(50), -- Pending, Approved, Rejected, Auto-executed
    operator_notes TEXT
);


-- ==========================================
-- MONGO DB COLLECTION SCHEMAS (DOCUMENT STORE)
-- ==========================================
/*
1. "raw_camera_trap_metadata" Collection:
{
  "_id": ObjectId("60c72b2f9b1d8b2bad18a221"),
  "camera_trap_id": "CT-304",
  "capture_time": "2026-07-13T10:00:00Z",
  "file_metadata": {
    "filename": "IMG_9482.jpg",
    "size_bytes": 1048576,
    "resolution": "1920x1080",
    "exif": {
      "iso": 400,
      "exposure": "1/125",
      "flash_fired": false
    }
  },
  "raw_inferences": [
    { "model_version": "v1.2", "class": "tiger", "score": 0.942 },
    { "model_version": "behavior-v1.0", "class": "running", "score": 0.81 }
  ]
}

2. "knowledge_base_chunks" Collection:
{
  "_id": ObjectId("60c72b2f9b1d8b2bad18a225"),
  "document_name": "wildlife_kb.txt",
  "chunk_index": 4,
  "heading_path": "Section 2. Species Conservation Guide > Tiger",
  "text_content": "Tiger (Panthera tigris): Endangered status. Needs dense vegetative cover, constant water access, and high prey density. Active patrol area: Sector B-4.",
  "vector_embedding": [0.0125, -0.0824, 0.1192, ..., 0.0034]
}
*/
