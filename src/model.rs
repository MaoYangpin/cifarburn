use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Cifar10Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct Cifar10ModelConfig {
    #[config(default = "10")] // Output classes
    num_classes: usize,
    #[config(default = "512")] // Hidden size of the first linear layer
    hidden_size: usize,
}

impl Cifar10ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Cifar10Model<B> {
        Cifar10Model {
            // Conv1: 3 input channels, 32 output channels, 5x5 kernel, no padding
            conv1: Conv2dConfig::new([3, 32], [5, 5]).init(device),
            // Conv2: 32 input channels, 64 output channels, 5x5 kernel, no padding
            conv2: Conv2dConfig::new([32, 64], [5, 5]).init(device),
            activation: Relu::new(),
            // Linear1: Input size 64 * 24 * 24 (flattened output from conv2), output size 512
            linear1: LinearConfig::new(64 * 24 * 24, self.hidden_size).init(device),
            // Linear2: Input size 512, output size num_classes
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> Cifar10Model<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, ..] = images.dims();

        // Expected Input: [batch_size, 3, 32, 32]
        // Conv1: Input [batch_size, 3, 32, 32] -> Output [batch_size, 32, 28, 28] (32 - 5 + 1 = 28)
        let x = self.conv1.forward(images);
        let x = self.activation.forward(x);

        // Conv2: Input [batch_size, 32, 28, 28] -> Output [batch_size, 64, 24, 24] (28 - 5 + 1 = 24)
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);

        // Reshape: Flatten the tensor from [batch_size, 64, 24, 24] to [batch_size, 64 * 24 * 24]
        let x = x.reshape([batch_size, 64 * 24 * 24]);

        // Linear1: Input [batch_size, 36864] -> Output [batch_size, 512]
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);

        // Linear2: Input [batch_size, 512] -> Output [batch_size, num_classes]
        self.linear2.forward(x)
    }
}
