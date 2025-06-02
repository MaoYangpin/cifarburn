use crate::{
    data::{Cifar10Batcher, Cifar10Item},
    model::Cifar10ModelConfig,
};
use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
    record::{CompactRecorder, Recorder},
};

#[allow(unused)]
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: Cifar10Item) {
    let config = Cifar10ModelConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = Cifar10Batcher;
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
