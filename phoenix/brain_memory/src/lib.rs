pub mod core;
pub mod models;

use pyo3::prelude::*;
use crate::core::retrieval::WorkingMemory;
// use crate::models::{Episode, Event, Knowledge, Perception, Reflection};

#[pyclass]
pub struct BrainMemoryClient {
    // Internal retrieval pipeline, wrapped to be synchronous for now
    pipeline: crate::core::retrieval::RetrievalPipeline,
}

#[pymethods]
impl BrainMemoryClient {
    #[new]
    pub fn new() -> Self {
        Self {
            pipeline: crate::core::retrieval::RetrievalPipeline::new(),
        }
    }

    /// Retrieve working memory (synchronous bridge to the async rust backend)
    pub fn retrieve_working_memory(&self, query: &str) -> PyResult<String> {
        // Block on the async method
        let rt = tokio::runtime::Runtime::new().unwrap();
        let wm = rt.block_on(async {
            self.pipeline.retrieve_working_memory(query, &[]).await
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
