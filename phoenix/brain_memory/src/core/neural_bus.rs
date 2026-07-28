use crate::models::{Event, Perception};
use crossbeam_channel::{bounded, Sender, Receiver};
use std::thread;
use std::sync::{Arc, RwLock};
use vdb_engine::domain::entities::{Engine, EngineTrait, CollectionTrait};

/// Neural Spiking Event Bus for ultra-fast, lock-free memory ingestion.
pub struct NeuralBus {
    sender: Sender<Spike>,
    receiver: Receiver<Spike>,
    engine: Arc<RwLock<Engine>>,
}

#[derive(Debug, Clone)]
pub enum Spike {
    PerceptionSpike(Perception),
    EventSpike(Event),
    // Can add more spike types like ReflectionSpike, ErrorSpike
}

impl NeuralBus {
    pub fn new(engine: Arc<RwLock<Engine>>) -> Self {
        // Use a bounded channel to prevent out-of-memory errors during high-throughput loads
        let (sender, receiver) = bounded(10_000);
        Self { sender, receiver, engine }
    }

    /// Fire a memory spike into the neural bus
    pub fn fire(&self, spike: Spike) {
        if let Err(e) = self.sender.send(spike) {
            eprintln!("⚡ Neural Bus Error: Failed to fire spike: {}", e);
        }
    }

    /// Start listening to spikes in a dedicated lock-free thread
    pub fn start_listening(&self) {
        let receiver = self.receiver.clone();
        let engine = self.engine.clone(); // Clone Arc for the thread
        
        thread::spawn(move || {
            // println!("⚡ Neural Bus is listening for spikes on a dedicated thread.");
            for spike in receiver {
                match spike {
                    Spike::PerceptionSpike(_p) => {
                        // TODO: Instantly route perception to short-term memory
                    }
                    Spike::EventSpike(e) => {
                        // Lock the engine and ingest the event directly into the vector DB
                        if let Ok(mut eng) = engine.write() {
                            if let Ok(collection) = eng.get_collection_mut("events") {
                                // Ingest event embedding with no metadata for now
                                let _ = collection.insert(e.embedding.clone(), 0, None);
                            }
                        }
                    }   
                }
            }
        });
    }
}
