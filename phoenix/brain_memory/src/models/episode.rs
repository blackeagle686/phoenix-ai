use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Episode {
    pub id: String,
    pub goal: String,
    pub context: Value,
    pub perceptions: Vec<String>, // references to Perception IDs
    pub events: Vec<String>,      // references to Event IDs
    pub actions: Vec<String>,     // references to Action IDs
    pub outputs: Vec<String>,     // String outputs or references
    pub final_result: String,
    pub success: bool,
    pub importance: f32,
    pub emotional_weight: f32,
    pub related_episodes: Vec<String>, // references to Episode IDs
    pub reflection_id: Option<String>, // reference to Reflection ID
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}
