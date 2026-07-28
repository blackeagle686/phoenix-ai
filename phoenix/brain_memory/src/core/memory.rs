use crate::models::{Episode, Event, Knowledge, Perception, Reflection};
use crate::core::{DreamEngine, NeuralBus, FractalCompressor};
use std::sync::Arc;

/// The BrainMemory manager coordinates storing and retrieving events,
/// orchestrating the flow from perception to reflection and knowledge.
pub struct BrainMemory {
    pub dream_engine: Arc<DreamEngine>,
    pub neural_bus: Arc<NeuralBus>,
    pub fractal_compressor: Arc<FractalCompressor>,
    // Database connection or VectorDB references will go here
}

impl Default for BrainMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl BrainMemory {
    pub fn new() -> Self {
        let neural_bus = Arc::new(NeuralBus::new());
        // Start the background listening thread for the neural bus
        neural_bus.start_listening();
        
        Self {
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
