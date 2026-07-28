use criterion::{black_box, criterion_group, criterion_main, Criterion};
use brain_memory::core::{NeuralBus, Spike, FractalCompressor};
use brain_memory::models::{Event, Perception};
use chrono::Utc;
use serde_json::json;

fn create_dummy_event(id_num: usize) -> Event {
    Event {
        id: format!("event_{}", id_num),
        event_type: "benchmark_event".to_string(),
        perception_id: "perception_0".to_string(),
        payload: json!({"data": "test"}),
        embedding: vec![0.1; 1536], // Typical OpenAI embedding size
        timestamp: Utc::now(),
        importance: 0.8,
    }
}

fn bench_neural_bus(c: &mut Criterion) {
    let mut group = c.benchmark_group("BrainMemory Neural Bus");
    
    let bus = NeuralBus::new();
    bus.start_listening(); // Starts the crossbeam thread
    
    group.bench_function("fire_1_event", |b| {
        b.iter(|| {
            let event = create_dummy_event(1);
            bus.fire(Spike::EventSpike(black_box(event)));
        })
    });
    
    group.finish();
}

fn bench_fractal_compressor(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fractal Compressor");
    
    let compressor = FractalCompressor::new(0.5);
    
    // Create 1000 events
    let mut events: Vec<Event> = (0..1000).map(create_dummy_event).collect();
    
    // Artificially age them so they trigger compression
    let past = Utc::now() - chrono::Duration::hours(1);
    for e in &mut events {
        e.timestamp = past;
    }
    
    group.bench_function("compress_1000_events", |b| {
        b.iter(|| {
            // We clone to preserve the original vector for the next iteration
            let mut test_events = events.clone();
            compressor.compress_timeline(black_box(&mut test_events));
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_neural_bus, bench_fractal_compressor);
criterion_main!(benches);
