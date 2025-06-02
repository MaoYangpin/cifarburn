use burn::data::dataset::Dataset;
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use std::fs::File;
use std::io::{self, ErrorKind, Read};
use std::path::{Path, PathBuf};

// Constants for CIFAR-10 data format
const IMAGE_BYTE_SIZE: usize = 3072; // 32 * 32 * 3 (channels first: RRR...GGG...BBB...)
const LABEL_BYTE_SIZE: usize = 1;
const RECORD_BYTE_SIZE: usize = IMAGE_BYTE_SIZE + LABEL_BYTE_SIZE; // 1 label byte + 3072 image bytes

#[derive(Clone, Debug)]
pub struct Cifar10Item {
    pub image: [u8; 3072], // 32x32x3 (RGB)
    pub label: u8,
}

#[allow(unused)]
#[derive(Clone, Debug)]
pub struct Cifar10Batch<B: Backend> {
    pub images: Tensor<B, 4>, // [batch_size, 3, 32, 32]
    pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone, Default)]
pub struct Cifar10Batcher;

impl<B: Backend> Batcher<B, Cifar10Item, Cifar10Batch<B>> for Cifar10Batcher {
    fn batch(&self, items: Vec<Cifar10Item>, device: &B::Device) -> Cifar10Batch<B> {
        // Process images
        let images = items
            .iter()
            .map(|item| {
                // Convert to float32 and normalize to [0,1]
                let float_data: Vec<f32> = item.image.iter().map(|x| *x as f32 / 255.0).collect();

                // Create tensor with shape [3072] then reshape to [1, 3, 32, 32]
                Tensor::<B, 1>::from_data(TensorData::new(float_data, [3072]), device)
                    .reshape([1, 3, 32, 32])
            })
            .collect();

        // Process labels
        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::new(vec![item.label as i64], [1]),
                    device,
                )
            })
            .collect();

        Cifar10Batch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cifar10Dataset {
    items: Vec<Cifar10Item>,
}
impl Cifar10Dataset {
    /// Creates a new `Cifar10Dataset` by reading all binary files from the specified directory.
    /// This function loads *all* training and test batches into a single dataset.
    /// It's mainly a utility to get the full dataset, but `train` and `test` methods
    /// below are more commonly used.
    fn default_data_dir() -> PathBuf {
        dirs::home_dir()
            .expect("Failed to find home directory")
            .join(".cache")
            .join("burn-dataset")
            .join("cifar10")
    }

    /// Creates a `Cifar10Dataset` containing only the training samples.
    pub fn train() -> Self {
        let mut train_items: Vec<Cifar10Item> = Vec::new();
        let data_dir = Self::default_data_dir(); // Use default
        let train_batches = [
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
        ];
        for batch_file in &train_batches {
            let path = data_dir.join(batch_file);
            eprintln!("Attempting to load training batch: {:?}", path);
            train_items.extend(Self::read_binary_file(&path).expect("Read train_items failed."));
        }
        Cifar10Dataset { items: train_items }
    }

    /// Creates a `Cifar10Dataset` containing only the test samples.
    pub fn test() -> Self {
        let mut test_items: Vec<Cifar10Item> = Vec::new();
        let data_dir = Self::default_data_dir(); // Use default
        let test_batch_name = "test_batch.bin";
        let path = data_dir.join(test_batch_name);
        eprintln!("Attempting to load test batch: {:?}", path);
        test_items.extend(Self::read_binary_file(&path).expect("Read test_items failed."));
        Cifar10Dataset { items: test_items }
    }

    /// Reads a single CIFAR-10 binary file and parses its contents into `Cifar10Item`s.
    fn read_binary_file(path: &Path) -> Result<Vec<Cifar10Item>, io::Error> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() % RECORD_BYTE_SIZE != 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "CIFAR-10 binary file {:?} has incorrect size. Expected a multiple of {}, got {}",
                    path,
                    RECORD_BYTE_SIZE,
                    buffer.len()
                ),
            ));
        }

        let mut items = Vec::new();
        for chunk in buffer.chunks_exact(RECORD_BYTE_SIZE) {
            let label = chunk[0];
            let mut image = [0u8; IMAGE_BYTE_SIZE];
            image.copy_from_slice(&chunk[LABEL_BYTE_SIZE..RECORD_BYTE_SIZE]);
            items.push(Cifar10Item { image, label });
        }
        Ok(items)
    }
}

impl Dataset<Cifar10Item> for Cifar10Dataset {
    fn get(&self, index: usize) -> Option<Cifar10Item> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
