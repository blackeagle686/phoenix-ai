use crate::models::{Episode, Event, Knowledge, Perception, Reflection};
use crate::core::{DreamEngine, NeuralBus, FractalCompressor, RetrievalPipeline};
use std::sync::{Arc, RwLock};
use vdb_engine::domain::entities::{Engine, EngineTrait};

/// The BrainMemory manager coordinates storing and retrieving events,
/// orchestrating the flow from perception to reflection and knowledge.
pub struct BrainMemory {
    pub engine: Arc<RwLock<Engine>>,
    pub retrieval: RetrievalPipeline,
    pub dream_engine: Arc<DreamEngine>,
    pub neural_bus: Arc<NeuralBus>,
    pub fractal_compressor: Arc<FractalCompressor>,
}

impl Default for BrainMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl BrainMemory {
    pub fn new() -> Self {
        // Initialize central engine
        let mut engine = Engine::new("brain_memory_core");
        let _ = engine.create_collection("knowledge", Some("HNSW"));
        let _ = engine.create_collection("episodes", Some("HNSW"));
        let _ = engine.create_collection("events", Some("HNSW"));
        
        let engine_arc = Arc::new(RwLock::new(engine));
        
        // Pass engine to Neural Bus and start it
        let neural_bus = Arc::new(NeuralBus::new(engine_arc.clone()));
        neural_bus.start_listening();
        
        let retrieval = RetrievalPipeline::new(engine_arc.clone());
        
        Self {
            engine: engine_arc,
            retrieval,
            dream_engine: Arc::new(DreamEngine::new()),
            neural_bus,
            fractal_compressor: Arc::new(FractalCompressor::new(0.5)), // 0.5 decay rate
        }
    }

    /// Wake up the AI and process incoming spikes
    pub async fn wake_up(&self) {
        self.dream_engine.stop_dreaming().await;
        println!("🧠 AI is awake and ready for input.");
    }
    
    /// Put the AI to sleep, triggering the Dream Engine
    pub async fn sleep(&self) {
        self.dream_engine.start_dreaming().await;
    }
}
