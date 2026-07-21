use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Knowledge {
    pub id: String,
    pub title: String,
    pub description: String,
    pub source_episodes: Vec<String>, // references to Episode IDs
    pub confidence: f32,
    pub usage_count: u32,
    pub domain: String,
    pub rule: String,
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
