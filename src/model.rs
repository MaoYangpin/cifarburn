use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Cifar10Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct Cifar10ModelConfig {
    #[config(default = "10")]
    num_classes: usize,
    #[config(default = "128")]
    hidden_size: usize,
}

impl Cifar10ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Cifar10Model<B> {
        Cifar10Model {
            linear1: LinearConfig::new(3 * 32 * 32, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Cifar10Model<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, ..] = images.dims();
        let x = images.reshape([batch_size, 3 * 32 * 32]);
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}
