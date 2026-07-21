use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Perception {
    pub id: String,
    pub source: String, // user / tool / sensor / api / llm
    pub raw_data: String,
    pub embedding: Vec<f32>,
    pub timestamp: DateTime<Utc>,
}
