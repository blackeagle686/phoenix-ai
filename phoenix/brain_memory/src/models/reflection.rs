use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reflection {
    pub id: String,
    pub episode_id: String,
    pub what_happened: String,
    pub success: bool,
    pub confidence: f32,
    pub failed_assumptions: Vec<String>,
    pub successful_patterns: Vec<String>,
    pub mistakes: Vec<String>,
    pub improvements: Vec<String>,
    pub alternative_actions: Vec<String>,
    pub summary: String,
    pub embedding: Vec<f32>,
    pub timestamp: DateTime<Utc>,
}
