mod data;

use burn::backend::Cuda;
use burn::{backend::Autodiff, data::dataset::Dataset};
// use burn::{data::dataloader::batcher::Batcher, prelude::*};

use crate::data::Cifar10Dataset;
// use crate::{model::ModelConfig, training::TrainingConfig};
// use burn::backend::Wgpu;

#[allow(unused)]
type MyBackend = Cuda<f32, i32>;
#[allow(unused)]
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() -> Result<(), std::io::Error> {
    println!("Loading CIFAR-10 datasets...");

    // Load the training dataset using the default path
    let train_dataset = Cifar10Dataset::train()?;
    println!(
        "Training dataset loaded with {} samples.",
        train_dataset.len()
    );

    // Load the test dataset using the default path
    let test_dataset = Cifar10Dataset::test()?;
    println!("Test dataset loaded with {} samples.", test_dataset.len());
    Ok(())
}
