use clap::{Parser, Subcommand};
use colored::Colorize;
use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
};

const NEURON_SIZE: usize = 128;
const EPOCH: usize = 40;
const LEARNING_RATE: f64 = 0.01;
const BIAS_1: f64 = 0.0;
const BIAS_2: f64 = 0.0;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train,
    Test,
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Train => train_model(),
        Command::Test => test_model(),
    }
}

/// Tests the trained model against the test dataset and reports accuracy metrics.
/// Loads test samples, runs forward propagation, and compares predictions to ground truth.
fn test_model() {
    let testing_labels_path = "./dataset/t10k-labels.idx1-ubyte";
    let testing_images_path = "./dataset/t10k-images.idx3-ubyte";

    let sample_count = 1000;

    println!(
        "{} Loading test dataset ({} samples)...",
        "[*]".cyan(),
        sample_count
    );
    let testing_labels = load_labels(testing_labels_path, sample_count);
    let testing_images = load_images(testing_images_path, sample_count);
    println!("{} Dataset loaded successfully", "[+]".green());

    // Loading model weights and biases
    println!("{} Loading model weights...", "[*]".cyan());
    let (b1, b2, w1, w2) = load_model().expect("Failed to load model from disk");
    println!("{} Model loaded successfully", "[+]".green());

    println!("{} Running inference...\n", "[*]".cyan());

    let mut failures = 0;
    let mut correct = 0;

    for (i, row) in testing_images.rows().into_iter().enumerate() {
        // Hidden layer: linear transform + ReLU activation
        let z1 = row2d(row.dot(&w1) + &b1);
        let a1 = z1.mapv(|x| x.max(0.0)); // ReLU

        // Output layer: linear transform + softmax for class probabilities
        let z2 = a1.dot(&w2) + &b2;
        let output = softmax(z2);

        // Pick the class with the highest probability as the prediction
        let predicted = output
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, val)| (index, *val))
            .unwrap();

        // Decode one-hot encoded label to the actual digit
        let real_number = testing_labels
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0;

        let is_correct = real_number == predicted.0;

        // Color-coded result indicator: [1] for correct, [0] for incorrect
        let result_marker = if is_correct {
            correct += 1;
            "[1]".green()
        } else {
            failures += 1;
            "[0]".red()
        };

        println!(
            "  sample {:>4} | real: {} | predicted: {} | prob: {:.2} |{}",
            i + 1,
            real_number,
            predicted.0,
            predicted.1,
            result_marker
        );
    }

    let accuracy = (correct as f32 / sample_count as f32) * 100.0;

    println!();
    println!("─────────────────────────────────────────");
    println!("{} Test Results", "[-]".yellow());
    println!("─────────────────────────────────────────");
    println!("  Samples tested : {}", sample_count);
    println!("  Correct        : {}", correct.to_string().green());
    println!("  Incorrect      : {}", failures.to_string().red());
    println!(
        "  Accuracy       : {:.2}%  {}",
        accuracy,
        if accuracy >= 90.0 {
            "[PASS]".green()
        } else {
            "[FAIL]".red()
        }
    );
    println!("─────────────────────────────────────────");
}

fn train_model() {
    println!("{} Starting training pipeline", "[*]".cyan());

    let training_labels_path = "./dataset/train-labels.idx1-ubyte";
    let training_images_path = "./dataset/train-images.idx3-ubyte";

    let sample_count = 6000;

    println!(
        "{} Loading labels from: {}",
        "[*]".cyan(),
        training_labels_path
    );
    let training_labels = load_labels(training_labels_path, sample_count);

    println!(
        "{} Loading images from: {}",
        "[*]".cyan(),
        training_images_path
    );
    let training_images = load_images(training_images_path, sample_count);

    println!(
        "{} Dataset loaded: {} samples, {} classes",
        "[+]".green(),
        sample_count,
        10
    );

    // -------------------------------------------------------------------------
    // Bias initialization
    // All neurons in layer 1 start with the same constant bias (BIAS_1),
    // and all output neurons start with BIAS_2. These are small constants
    // that shift the activation threshold.
    // -------------------------------------------------------------------------

    // Bias for hidden layer: shape [128] (1 per neuron in layer 1)
    let mut b1 = Array1::from_vec(vec![BIAS_1; NEURON_SIZE]);
    // Bias for output layer: shape [10] (1 per class digit 0-9)
    let mut b2 = Array1::from_vec(vec![BIAS_2; 10]);

    println!(
        "{} Biases initialized | b1: [{};{}] b2: [{};10]",
        "[*]".cyan(),
        BIAS_1,
        NEURON_SIZE,
        BIAS_2
    );

    // -------------------------------------------------------------------------
    // Weight initialization via He (Kaiming) initialization
    //
    // He init sets weights from a normal distribution with:
    //   mean = 0, std = sqrt(2 / fan_in)
    //
    // This keeps variance stable through ReLU activations, preventing
    // vanishing or exploding gradients early in training.
    // -------------------------------------------------------------------------
    let normal_w1 = Normal::new(0.0, (2.0 / 784.0_f64).sqrt()).unwrap();
    let normal_w2 = Normal::new(0.0, (2.0 / NEURON_SIZE as f64).sqrt()).unwrap();

    let mut rng = rand::rng();

    println!(
        "{} Initializing weights with He initialization",
        "[*]".cyan()
    );

    // W1: maps input layer (784 pixels) to hidden layer (NEURON_SIZE neurons)
    // Shape: [784 x 128]
    let w1_data: Vec<f64> = (0..784 * NEURON_SIZE)
        .map(|_| normal_w1.sample(&mut rng))
        .collect();
    // 784 x 128
    let mut w1 = Array2::from_shape_vec((784, NEURON_SIZE), w1_data).unwrap();

    // W2: maps hidden layer (NEURON_SIZE neurons) to output layer (10 classes)
    // Shape: [128 x 10]
    let w2_data: Vec<f64> = (0..NEURON_SIZE * 10)
        .map(|_| normal_w2.sample(&mut rng))
        .collect();
    // 128 x 10
    let mut w2 = Array2::from_shape_vec((NEURON_SIZE, 10), w2_data).unwrap();

    println!(
        "{} Weights initialized | w1: [784x{}] w2: [{}x10]",
        "[+]".green(),
        NEURON_SIZE,
        NEURON_SIZE
    );

    // -------------------------------------------------------------------------
    // Training loop
    //
    // For each epoch, we iterate over every sample and perform:
    //   1. Forward pass  -- compute predictions
    //   2. Loss          -- measure how wrong we are
    //   3. Backward pass -- compute gradients via backpropagation
    //   4. Update        -- nudge weights/biases in the right direction
    // -------------------------------------------------------------------------
    println!(
        "{} Beginning training | epochs: {} | learning rate: {} | samples: {}",
        "[*]".cyan(),
        EPOCH,
        LEARNING_RATE,
        sample_count
    );

    // row = 1 x 784
    for j in 0..EPOCH {
        println!("{} Epoch {}/{} starting", "[*]".cyan(), j + 1, EPOCH);

        let mut correct = 0;
        let mut total_loss = 0.0_f64;

        for (i, row) in training_images.rows().into_iter().enumerate() {
            // -----------------------------------------------------------------
            // Forward pass
            // -----------------------------------------------------------------

            // Z1 = input · W1 + b1
            // Linear transformation from input space [784] to hidden space [128]
            // Shape: [1 x 128]
            let z1 = row2d(row.dot(&w1) + &b1);

            // A1 = ReLU(Z1)
            // ReLU (Rectified Linear Unit) zeroes out negative activations,
            // introducing non-linearity so the network can learn complex patterns.
            // Shape: [1 x 128]
            let a1 = z1.mapv(|x| x.max(0.0));

            // Z2 = A1 · W2 + b2
            // Linear transformation from hidden space [128] to output space [10]
            // Shape: [1 x 10]
            let z2 = a1.dot(&w2) + &b2;

            // output = softmax(Z2)
            // Converts raw logits into a probability distribution over 10 classes.
            // All values are in (0, 1) and sum to 1.
            // Shape: [1 x 10]
            let output = softmax(z2);

            // -----------------------------------------------------------------
            // Loss computation (Cross-entropy, implicit via gradient)
            //
            // The explicit loss value is not computed here, but the gradient
            // dZ2 = output - y is the analytical derivative of cross-entropy
            // loss with respect to Z2 when softmax is the final activation.
            // -----------------------------------------------------------------

            // dL/dZ2 = predicted probability - one-hot true label
            // This is the error signal for the output layer.
            // Shape: [1 x 10]
            let dz2 = &output - &training_labels.row(i);

            // Track cross-entropy loss for logging: -sum(y * log(p))
            let label_row = training_labels.row(i);
            let loss: f64 = label_row
                .iter()
                .zip(output.iter())
                .map(|(y, p)| -y * p.ln().max(-100.0))
                .sum();
            total_loss += loss;

            // Track accuracy for logging
            let pred = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let actual = label_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            if pred == actual {
                correct += 1;
            }

            // -----------------------------------------------------------------
            // Backward pass (Backpropagation)
            //
            // We propagate the error signal backwards through the network
            // to compute how much each weight and bias contributed to the loss.
            // -----------------------------------------------------------------

            // dL/dW2 = A1ᵀ · dZ2
            // How much each weight in W2 is responsible for the output error.
            // Shape: [128 x 10]  ([128x1] . [1x10])
            let dw2 = a1.t().dot(&dz2);

            // dL/dA1 = dZ2 · W2ᵀ
            // Propagate error back through W2 into the hidden layer.
            // Shape: [1 x 128]
            let da1 = dz2.dot(&w2.t());

            // dA1/dZ1 = ReLU'(Z1) = 1 if Z1 > 0, else 0
            // The derivative of ReLU: passes gradient only where Z1 was positive.
            // Shape: [1 x 128]
            let t = z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

            // dL/dZ1 = dL/dA1 * ReLU'(Z1)
            // Chain rule: combine hidden layer error with ReLU gate.
            // Shape: [1 x 128]
            let dz1 = da1 * t;

            // dL/dW1 = inputᵀ · dZ1
            // How much each weight in W1 is responsible for the hidden layer error.
            // Shape: [784 x 128]  ([784x1] . [1x128])
            let dw1 = row.insert_axis(Axis(0)).t().dot(&dz1);

            // dL/db2 = sum(dZ2)
            // Bias gradient for output layer (scalar collapsed from [1x10]).
            let db2 = dz2.sum();

            // dL/db1 = sum(dZ1)
            // Bias gradient for hidden layer (scalar collapsed from [1x128]).
            let db1 = dz1.sum();

            // -----------------------------------------------------------------
            // Gradient descent parameter update
            //
            // theta = theta - learning_rate * d(theta)
            // We step each parameter opposite to its gradient to reduce loss.
            // -----------------------------------------------------------------
            w1 = w1 - LEARNING_RATE * dw1;
            w2 = w2 - LEARNING_RATE * dw2;
            b1 -= LEARNING_RATE * db1;
            b2 -= LEARNING_RATE * db2;
        }

        // Epoch summary log
        let accuracy = correct as f64 / 6000.0 * 100.0;
        let avg_loss = total_loss / 6000.0;
        println!(
            "{} Epoch {}/{} complete | avg loss: {:.4} | accuracy: {:.2}%",
            "[+]".green(),
            j + 1,
            EPOCH,
            avg_loss,
            accuracy
        );
    }

    println!("{} Training complete. Saving model...", "[*]".cyan());
    save_model(&b1, &b2, &w1, &w2).expect("failed to save model");
    println!("{} Model saved successfully", "[+]".green());
}

/// save the model to file le
/// header 0  b1 size - u8
/// header 1  b2 size - u8
/// header 2  w1 rows - u32
/// header 3  w1 cols - u32
/// header 4  w2 rows - u32
/// header 5  w3 cols - u32
fn save_model(
    b1: &Array1<f64>,
    b2: &Array1<f64>,
    w1: &Array2<f64>,
    w2: &Array2<f64>,
) -> Result<(), String> {
    let file = File::create("model.nn").map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(file);

    // header
    writer
        .write_all(&(b1.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(b2.len() as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(w1.shape()[0] as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(w1.shape()[1] as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(w2.shape()[0] as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(w2.shape()[1] as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;

    // biases and weights written separately
    for val in b1.iter().chain(b2.iter()) {
        writer
            .write_all(&val.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    for val in w1.iter().chain(w2.iter()) {
        writer
            .write_all(&val.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn load_model() -> Result<(Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>), String> {
    let file = File::open("model.nn").map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file);

    // read header
    let mut u32_buf = [0u8; 4];

    reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let b1_size = u32::from_le_bytes(u32_buf) as usize;

    reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let b2_size = u32::from_le_bytes(u32_buf) as usize;

    reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let w1_rows = u32::from_le_bytes(u32_buf) as usize;

    reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let w1_cols = u32::from_le_bytes(u32_buf) as usize;

    reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let w2_rows = u32::from_le_bytes(u32_buf) as usize;

    reader.read_exact(&mut u32_buf).map_err(|e| e.to_string())?;
    let w2_cols = u32::from_le_bytes(u32_buf) as usize;

    // helper to read n f64 values
    let mut read_f64s = |n: usize| -> Result<Vec<f64>, String> {
        let mut buf = [0u8; 8];
        (0..n)
            .map(|_| {
                reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
                Ok(f64::from_le_bytes(buf))
            })
            .collect()
    };

    let b1 = Array1::from_vec(read_f64s(b1_size)?);
    let b2 = Array1::from_vec(read_f64s(b2_size)?);
    let w1 = Array2::from_shape_vec((w1_rows, w1_cols), read_f64s(w1_rows * w1_cols)?)
        .map_err(|e| e.to_string())?;
    let w2 = Array2::from_shape_vec((w2_rows, w2_cols), read_f64s(w2_rows * w2_cols)?)
        .map_err(|e| e.to_string())?;

    Ok((b1, b2, w1, w2))
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

fn load_images(path: &str, num: usize) -> Array2<f64> {
    let bytes = fs::read(path).expect("Failed to read file");

    let count_header: [u8; 4] = bytes[4..8].try_into().unwrap();
    let _count = u32::from_be_bytes(count_header);

    let mut images = Vec::new();

    for chunk in &bytes[16..(16 + (784 * num))] {
        let label = u8::from_be_bytes([*chunk]);
        images.push(label as f64 / 255.0);
    }

    let file_name: Vec<&str> = path.split("/").collect();
    println!("Found {} images from {}", images.len() / 784, file_name[2]);

    Array2::from_shape_vec((images.len() / 784, 784), images)
        .expect("failed to convert images to matrix")
}

fn load_labels(path: &str, num: usize) -> Array2<f64> {
    let bytes = fs::read(path).expect("Failed to read file");

    let count_header: [u8; 4] = bytes[4..8].try_into().unwrap();
    let _count = u32::from_be_bytes(count_header);

    let mut labels = Vec::new();

    for chunk in &bytes[8..(8 + num)] {
        let mut one_hot = [0.0; 10];
        let label = u8::from_be_bytes([*chunk]);
        one_hot[label as usize] = 1.0;
        labels.extend(one_hot);
    }

    let file_name: Vec<&str> = path.split("/").collect();
    println!("Found {} labels from {}", num, file_name[2]);

    Array2::from_shape_vec((num, 10), labels).expect("failed to convert labels to matrix")
}
