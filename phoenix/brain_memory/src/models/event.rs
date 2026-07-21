use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    pub id: String,
    pub event_type: String, 
    // user_input, llm_response, tool_call, tool_result, error, feedback
    pub perception_id: String,
    pub payload: Value,
    pub embedding: Vec<f32>,
    pub timestamp: DateTime<Utc>,
    pub importance: f32,
}
