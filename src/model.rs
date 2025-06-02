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
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    pool: MaxPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct Cifar10ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.3")]
    dropout: f64,
}

impl Cifar10ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Cifar10Model<B> {
        Cifar10Model {
            conv1: Conv2dConfig::new([3, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            conv2: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn2: BatchNormConfig::new(128).init(device),
            conv3: Conv2dConfig::new([128, 256], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn3: BatchNormConfig::new(256).init(device),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(256 * 4 * 4, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Cifar10Model<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, ..] = images.dims();

        let x = self.conv1.forward(images);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        let x = self.dropout.forward(x);

        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        let x = self.dropout.forward(x);

        let x = x.reshape([batch_size, 256 * 4 * 4]);
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        self.linear2.forward(x)
    }
}
