use std::fs;

use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};

const NEURON_SIZE: usize = 128;
// const EPOCH: usize = 2;
const BIAS_1: f64 = 0.0;
const BIAS_2: f64 = 0.0;

fn main() {
    let training_labels_path = "./dataset/train-labels.idx1-ubyte";
    let training_images_path = "./dataset/train-images.idx3-ubyte";
    let training_labels = load_labels(training_labels_path);
    let training_images = load_images(training_images_path);

    // generate the bias values
    let b1 = Array1::from_vec(vec![BIAS_1; NEURON_SIZE]);
    let b2 = Array1::from_vec(vec![BIAS_2; 10]);

    // generate the weight 1 values
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    let w1_data: Vec<f64> = (0..784 * NEURON_SIZE)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let w1 = Array2::from_shape_vec((784, NEURON_SIZE), w1_data).unwrap();

    // generate the weight 2 values
    let w2_data: Vec<f64> = (0..NEURON_SIZE * 10)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let w2 = Array2::from_shape_vec((NEURON_SIZE, 10), w2_data).unwrap();

    // First pass
    let z1 = training_images.dot(&w1) + b1;
    // ReLU activation function
    let a1 = z1.mapv_into(|x| x.max(0.0));
    let z2 = a1.dot(&w2) + b2;

    let output = softmax(z2.clone());

    println!();
    println!("{:?}", output);

    println!();
    println!("{:?}", training_labels.row(0));

    println!();
    println!("sum {:?}", output.row(0).sum());
}

// fn softmax(matrix: Array2<f64>) -> Array2<f64> {
//     let mut result = matrix.clone();
//
//     for mut row in result.axis_iter_mut(Axis(0)) {
//         let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
//         let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
//         row.mapv_inplace(|v| (v - max).exp() / sum);
//     }
//
//     result
// }

fn softmax(matrix: Array2<f64>) -> Array2<f64> {
    let mut result = matrix.clone();

    for mut row in result.axis_iter_mut(Axis(0)) {
        let sum: f64 = row.iter().map(|v| v.exp()).sum();
        row.mapv_inplace(|v| v.exp() / sum);
    }

    result
}

fn load_images(path: &str) -> Array2<f64> {
    let bytes = fs::read(path).expect("Failed to read file");

    let count_header: [u8; 4] = bytes[4..8].try_into().unwrap();
    let _count = u32::from_be_bytes(count_header);

    let mut images = Vec::new();

    for chunk in &bytes[16..(16 + (784 * 1))] {
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
    let count = u32::from_be_bytes(count_header);

    let mut labels = Vec::new();

    for chunk in &bytes[8..] {
        let mut one_hot = [0.0; 10];
        let label = u8::from_be_bytes([*chunk]);
        one_hot[label as usize] = 1.0;
        labels.extend(one_hot);
    }

    let file_name: Vec<&str> = path.split("/").collect();
    println!("Found {} labels from {}", count, file_name[2]);

    Array2::from_shape_vec((count as usize, 10), labels)
        .expect("failed to convert labels to matrix")
}
