mod data;
mod inference;
mod model;
mod training;

use crate::data::Cifar10Dataset;
use crate::model::Cifar10ModelConfig;
use crate::training::TrainingConfig;
use burn::backend::Autodiff;
use burn::backend::Cuda;
use burn::data::dataloader::Dataset;
use burn::optim::AdamConfig;
use burn_cuda::CudaDevice;

type MyBackend = Cuda<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    // Create a default Wgpu device
    // let device = burn::backend::wgpu::WgpuDevice::default();
    let device = CudaDevice::default();

    // All the training artifacts will be saved in this directory
    let artifact_dir = "/tmp/guide";

    // Train the model
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(Cifar10ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    let test_dataset = Cifar10Dataset::test().unwrap();
    println!("Test dataset loaded with {} samples.", test_dataset.len());

    if let Some(test_item) = test_dataset.get(0) {
        let artifact_dir = "artifacts";
        let device = Default::default();
        inference::infer::<MyBackend>(artifact_dir, device, test_item);
    }
}
