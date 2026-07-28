pub mod core;
pub mod models;

use pyo3::prelude::*;
use crate::core::retrieval::WorkingMemory;
// use crate::models::{Episode, Event, Knowledge, Perception, Reflection};

#[pyclass]
pub struct BrainMemoryClient {
    // Wrap the full BrainMemory controller
    memory: crate::core::memory::BrainMemory,
}

#[pymethods]
impl BrainMemoryClient {
    #[new]
    pub fn new() -> Self {
        Self {
            memory: crate::core::memory::BrainMemory::new(),
        }
    }

    /// Retrieve working memory (synchronous bridge to the async rust backend)
    pub fn retrieve_working_memory(&self, query: &str) -> PyResult<String> {
        // Block on the async method
        let rt = tokio::runtime::Runtime::new().unwrap();
        let wm = rt.block_on(async {
            let query_embedding = vec![0.1f32; 384];
            self.memory.retrieval.retrieve_working_memory(query, &query_embedding).await
        });
        
        // Serialize WorkingMemory to JSON String for easy consumption in Python
        let json_wm = serde_json::to_string(&wm).unwrap();
        Ok(json_wm)
    }

    pub fn add_event(&self, _event_json: &str) -> PyResult<()> {
        // Parse event and send to memory background processing
        // ...
        Ok(())
    }
}

/// The Python module definition
#[pymodule]
fn brain_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BrainMemoryClient>()?;
    Ok(())
}
