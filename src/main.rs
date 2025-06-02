#![recursion_limit = "256"]
mod data;
mod inference;
mod model;
mod training;

use crate::data::Cifar10Dataset;
use crate::model::Cifar10ModelConfig;
use crate::training::TrainingConfig;
use burn::backend::{Autodiff, Wgpu};
// use burn::backend::Cuda;
use burn::data::dataloader::Dataset;
use burn::optim::AdamConfig;
// use burn_cuda::CudaDevice;

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    // Create a default Wgpu device
    let device = burn::backend::wgpu::WgpuDevice::default();
    // let device = CudaDevice::default();
    // All the training artifacts will be saved in this directory
    let artifact_dir = "/tmp/cifar10";

    // Train the model
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(Cifar10ModelConfig::new(), AdamConfig::new()),
        device.clone(),
    );

    inference::infer::<MyBackend>(
        artifact_dir,
        device,
        Cifar10Dataset::test().get(42).unwrap(),
    );
}
