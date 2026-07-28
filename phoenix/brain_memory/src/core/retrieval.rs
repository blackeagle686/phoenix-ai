use serde::{Serialize, Deserialize};
use std::sync::{Arc, RwLock};
use vdb_engine::domain::entities::{Engine, EngineTrait, CollectionTrait};

pub struct RetrievalPipeline {
    pub engine: Arc<RwLock<Engine>>,
}

// Default requires no-arg constructor, we might want to remove it or mock it
impl Default for RetrievalPipeline {
    fn default() -> Self {
        let mut engine = Engine::new("default");
        let _ = engine.create_collection("knowledge", Some("HNSW"));
        let _ = engine.create_collection("episodes", Some("HNSW"));
        let _ = engine.create_collection("events", Some("HNSW"));
        Self::new(Arc::new(RwLock::new(engine)))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkingMemory {
    pub knowledge: Vec<crate::models::Knowledge>,
    pub episodes: Vec<crate::models::Episode>,
    pub events: Vec<crate::models::Event>,
    // The final synthesized context to pass to LLM
}

impl RetrievalPipeline {
    pub fn new(engine: Arc<RwLock<Engine>>) -> Self {
        Self {
            engine
        }
    }

    /// Input -> Search Knowledge -> Search Episodes -> Search Related Events -> Build Working Memory -> LLM
    pub async fn retrieve_working_memory(&self, query: &str, query_embedding: &[f32]) -> WorkingMemory {
        // 1. Search Knowledge
        let knowledge = self.search_knowledge(query, query_embedding).await;

        // 2. Search Episodes
        let episodes = self.search_episodes(query, query_embedding).await;

        // 3. Search Related Events
        let events = self.search_related_events(query, query_embedding).await;

        // 4. Build Working Memory
        WorkingMemory {
            knowledge,
            episodes,
            events,
        }
    }

    pub async fn search_knowledge(&self, _query: &str, query_embedding: &[f32]) -> Vec<crate::models::Knowledge> {
        let engine = self.engine.read().unwrap();
        if let Ok(collection) = engine.get_collection("knowledge") {
            if let Ok(Some((id, score))) = collection.query(query_embedding.to_vec()) {
                println!("🧠 [Retrieval] Found Knowledge Match: {} (Similarity: {:.4})", id, score);
            }
        }
        vec![]
    }

    pub async fn search_episodes(&self, _query: &str, query_embedding: &[f32]) -> Vec<crate::models::Episode> {
        let engine = self.engine.read().unwrap();
        if let Ok(collection) = engine.get_collection("episodes") {
            if let Ok(Some((id, score))) = collection.query(query_embedding.to_vec()) {
                println!("🎬 [Retrieval] Found Episode Match: {} (Similarity: {:.4})", id, score);
            }
        }
        vec![]
    }

    pub async fn search_related_events(&self, _query: &str, query_embedding: &[f32]) -> Vec<crate::models::Event> {
        let engine = self.engine.read().unwrap();
        if let Ok(collection) = engine.get_collection("events") {
            if let Ok(Some((id, score))) = collection.query(query_embedding.to_vec()) {
                println!("⚡ [Retrieval] Found Event Match: {} (Similarity: {:.4})", id, score);
            }
        }
        vec![]
    }
}
