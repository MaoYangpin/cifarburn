use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Cifar10Model<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    pool1: MaxPool2d,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    pool2: MaxPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct Cifar10ModelConfig {
    #[config(default = "10")]
    num_classes: usize,
    #[config(default = "512")]
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl Cifar10ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Cifar10Model<B> {
        Cifar10Model {
            conv1: Conv2dConfig::new([3, 32], [5, 5]).init(device),
            bn1: BatchNormConfig::new(32).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).init(),
            conv2: Conv2dConfig::new([32, 64], [5, 5]).init(device),
            bn2: BatchNormConfig::new(64).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(64 * 22 * 22, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> Cifar10Model<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        // Keep this dbg! to confirm initial input is still 32x32
        dbg!(&images.dims(), "Input to model.forward");

        let [batch_size, _channels, _height, _width] = images.dims();

        let x = self.conv1.forward(images);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool1.forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool2.forward(x);
        let x = self.dropout.forward(x);

        // ADD THIS dbg! TO SEE SHAPE JUST BEFORE RESHAPE
        dbg!(&x.dims(), "Shape before final reshape");

        let x = x.reshape([batch_size, 64 * 22 * 22]); // This is the line that panics

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.linear2.forward(x)
    }
}
