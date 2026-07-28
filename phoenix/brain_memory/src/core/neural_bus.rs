use crate::models::{Event, Perception};
use crossbeam_channel::{bounded, Sender, Receiver};
use std::thread;

/// Neural Spiking Event Bus for ultra-fast, lock-free memory ingestion.
pub struct NeuralBus {
    sender: Sender<Spike>,
    receiver: Receiver<Spike>,
}

#[derive(Debug, Clone)]
pub enum Spike {
    PerceptionSpike(Perception),
    EventSpike(Event),
    // Can add more spike types like ReflectionSpike, ErrorSpike
}

impl NeuralBus {
    pub fn new() -> Self {
        // Use a bounded channel to prevent out-of-memory errors during high-throughput loads
        let (sender, receiver) = bounded(10_000);
        Self { sender, receiver }
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
        thread::spawn(move || {
            // println!("⚡ Neural Bus is listening for spikes on a dedicated thread.");
            for spike in receiver {
                match spike {
                    Spike::PerceptionSpike(_p) => {
                        // TODO: Instantly route perception to short-term memory
                        // println!("⚡ [Neural Bus]: Perception Spike received: {}", p.id);
                    }
                    Spike::EventSpike(_e) => {
                        // TODO: Check similarity with existing events and increase 'importance' weight
                        // println!("⚡ [Neural Bus]: Event Spike received: {} (Importance: {})", e.id, e.importance);
                    }   
                }
            }
        });
    }
}
