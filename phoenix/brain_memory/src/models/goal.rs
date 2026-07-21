use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Goal {
    pub id: String,
    pub title: String,
    pub priority: i32,
    pub status: String,
    pub related_episodes: Vec<String>,
    pub created_at: DateTime<Utc>,
}
