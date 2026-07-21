use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Action {
    pub id: String,
    pub action_type: String, // search, call_tool, generate, plan, code
    pub input_data: Value,
    pub output_data: Value,
    pub execution_time_ms: f64,
    pub success: bool,
    pub error_message: Option<String>,
}
