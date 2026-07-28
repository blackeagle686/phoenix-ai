use crate::models::{Event, Knowledge};
use chrono::{Utc, Duration};

/// Compresses short-term, highly-detailed memories into abstract rules over time.
pub struct FractalCompressor {
    decay_rate: f32, // How fast memories lose their high-definition detail
}

impl FractalCompressor {
    pub fn new(decay_rate: f32) -> Self {
        Self { decay_rate }
    }

    /// Evaluates a list of events and determines if they should be compressed
    pub fn compress_timeline(&self, events: &mut Vec<Event>) -> Vec<Knowledge> {
        let now = Utc::now();
        let mut extracted_wisdom = Vec::new();

        for event in events.iter_mut() {
            let age = now.signed_duration_since(event.timestamp).num_seconds() as f32;
            
            // If the event is older than a certain threshold, apply decay
            if age > 300.0 { // Older than 5 minutes
                // Reduce importance based on age and decay rate
                event.importance -= self.decay_rate * (age / 3600.0);
                
                // If the event drops below a threshold but still holds structural value,
                // we fractalize it (extract the core concept)
                if event.importance < 0.1 && event.importance > 0.0 {
                    println!("🌀 [Fractal Compressor]: Compressing Event {} into abstract Knowledge.", event.id);
                    // TODO: Actually create Knowledge node
                    // extracted_wisdom.push(Knowledge::new(...));
                    
                    // Mark event for deletion or deep storage by dropping importance to 0
                    event.importance = 0.0; 
                }
            }
        }
        
        extracted_wisdom
    }
}

/// Helper function to calculate cosine similarity between two vectors
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }
    
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();
    
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    }
}
