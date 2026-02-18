use std::fs;

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};

const NEURON_SIZE: usize = 128;
const EPOCH: usize = 10;
const LEARNING_RATE: f64 = 0.01;
const BIAS_1: f64 = 0.0;
const BIAS_2: f64 = 0.0;
const NUM_IMAGES: usize = 10;

fn main() {
    let training_labels_path = "./dataset/train-labels.idx1-ubyte";
    let training_images_path = "./dataset/train-images.idx3-ubyte";
    let training_labels = load_labels(training_labels_path);
    let training_images = load_images(training_images_path);

    // generate the bias values
    // 1 x 128
    let mut b1 = Array1::from_vec(vec![BIAS_1; NEURON_SIZE]);
    // 1 x 10
    let mut b2 = Array1::from_vec(vec![BIAS_2; 10]);

    // generate the weight 1 values
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    let w1_data: Vec<f64> = (0..784 * NEURON_SIZE)
        .map(|_| normal.sample(&mut rng))
        .collect();
    // 784 x 128
    let mut w1 = Array2::from_shape_vec((784, NEURON_SIZE), w1_data).unwrap();

    // generate the weight 2 values
    let w2_data: Vec<f64> = (0..NEURON_SIZE * 10)
        .map(|_| normal.sample(&mut rng))
        .collect();
    // 128 x 10
    let mut w2 = Array2::from_shape_vec((NEURON_SIZE, 10), w2_data).unwrap();

    // row = 1 x 784
    for _ in 0..EPOCH {
        for (i, row) in training_images.rows().into_iter().enumerate() {
            println!("{} ", i);
            // First pass = 1 x 128
            let z1 = row2d(row.dot(&w1) + &b1);
            // ReLU activation function
            let a1 = z1.clone().mapv_into(|x| x.max(0.0));
            // 1 x 10
            let z2 = a1.dot(&w2) + &b2;

            let output = softmax(z2);

            // Loss
            // dZ2 = output - y
            let dz2 = &output - &training_labels.row(i);

            // dW2 = A1ᵀ · dZ2
            // blame matrix dW2 = a1 transposed . loss = [128;1] . [1;10] = [128;10]
            let dw2 = a1.t().dot(&dz2);

            // dA1 = dZ2 · W2ᵀ
            let da1 = dz2.dot(&w2.t());
            // dZ1 = dA1 * (Z1 > 0 ? 1 : 0)
            let t = z1.clone().mapv(|v| if v > 0.0 { v } else { 0.0 });
            let dz1 = da1 * t;
            // dW1 = inputᵀ · dZ1
            let dw1 = row.insert_axis(Axis(0)).t().dot(&dz1);

            let db2 = dz2.sum();
            let db1 = dz1.sum();

            // W1 = W1 - lr * dW1
            w1 = w1 - LEARNING_RATE * dw1;
            // W2 = W2 - lr * dW2
            w2 = w2 - LEARNING_RATE * dw2;
            // b1 = b1 - lr * db1
            b1 -= LEARNING_RATE * db1;
            // b2 = b2 - lr * db2
            b2 -= LEARNING_RATE * db2;
        }
    }
    let t = 5;
    let z1 = (training_images.row(t).dot(&w1) + &b1).insert_axis(Axis(0));
    let a1 = z1.clone().mapv_into(|x| x.max(0.0));
    let z2 = a1.dot(&w2) + &b2;
    let output = softmax(z2.clone());
    println!("{:?}", output);
    println!();
    println!("{:?}", training_labels.row(t));
}

fn row2d(row: Array1<f64>) -> Array2<f64> {
    row.insert_axis(Axis(0))
}

fn softmax(vec: Array2<f64>) -> Array2<f64> {
    let mut result = vec.clone();

    let sum: f64 = result.iter().map(|v| v.exp()).sum();
    result.mapv_inplace(|v| v.exp() / sum);

    result
}

fn load_images(path: &str) -> Array2<f64> {
    let bytes = fs::read(path).expect("Failed to read file");

    let count_header: [u8; 4] = bytes[4..8].try_into().unwrap();
    let _count = u32::from_be_bytes(count_header);

    let mut images = Vec::new();

    for chunk in &bytes[16..(16 + (784 * NUM_IMAGES))] {
        let label = u8::from_be_bytes([*chunk]);
        images.push(label as f64 / 255.0);
    }

    let file_name: Vec<&str> = path.split("/").collect();
    println!("Found {} images from {}", images.len() / 784, file_name[2]);

    Array2::from_shape_vec((images.len() / 784, 784), images)
        .expect("failed to convert images to matrix")
}

fn load_labels(path: &str) -> Array2<f64> {
    let bytes = fs::read(path).expect("Failed to read file");

    let count_header: [u8; 4] = bytes[4..8].try_into().unwrap();
    let _count = u32::from_be_bytes(count_header);

    let mut labels = Vec::new();

    for chunk in &bytes[8..(8 + NUM_IMAGES)] {
        let mut one_hot = [0.0; 10];
        let label = u8::from_be_bytes([*chunk]);
        one_hot[label as usize] = 1.0;
        labels.extend(one_hot);
    }

    let file_name: Vec<&str> = path.split("/").collect();
    println!("Found {} labels from {}", NUM_IMAGES, file_name[2]);

    Array2::from_shape_vec((NUM_IMAGES, 10), labels).expect("failed to convert labels to matrix")
}
