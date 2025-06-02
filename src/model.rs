use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Cifar10Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct Cifar10ModelConfig {
    num_classes: usize, // 10 for CIFAR-10
    hidden_size: usize, // e.g. 512
    #[config(default = "0.3")]
    dropout: f64,
}

impl Cifar10ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Cifar10Model<B> {
        Cifar10Model {
            conv1: Conv2dConfig::new([3, 32], [3, 3]).init(device), // Input channels: 3 (RGB)
            conv2: Conv2dConfig::new([32, 64], [3, 3]).init(device),
            conv3: Conv2dConfig::new([64, 128], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([4, 4]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(128 * 4 * 4, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Cifar10Model<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, ..] = images.dims();

        let x = self.conv1.forward(images); // [batch_size, 32, height, width]
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv2.forward(x); // [batch_size, 64, height, width]
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv3.forward(x); // [batch_size, 128, height, width]
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.pool.forward(x); // [batch_size, 128, 4, 4]
        let x = x.reshape([batch_size, 128 * 4 * 4]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
