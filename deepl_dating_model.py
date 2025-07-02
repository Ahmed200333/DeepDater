import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# --- 1. Loading dataset ---
data = np.load("final_dataset.npz")
X = data["X"]
y_complete = data["y_complete"]
y_partial = data["y_partial"]

print(f" .npz keys : {list(data.keys())}")
print(f" X : {X.shape}, y_complete : {y_complete.shape}, y_partial : {y_partial.shape}")

# --- 2. Normalization ---

# Normalize X so the mean of positive branch lengths = 1
X_norm = []
for branches in X:
    mean_len = branches[branches > 0].mean() if np.any(branches > 0) else 1
    X_norm.append(branches / mean_len)
X_norm = np.array(X_norm)

# Normalize dates so that the most recent node has value 1
y_complete_norm, y_partial_norm, y_complete_norm_masked, max_dates = [], [], [], []
for yc, yp in zip(y_complete, y_partial):
    max_d = np.max(yc)
    max_dates.append(max_d)
    yc_norm = 1 - (max_d - yc)
    yp_norm = 1 - (max_d - yp)
    # Masking: 1 for nodes to predict (not leaves), 0 otherwise
    mask = ((yp == 0) & (yc != 0)).astype(float)
    y_complete_norm.append(yc_norm)
    y_partial_norm.append(yp_norm)
    y_complete_norm_masked.append(yc_norm * mask)

y_complete_norm = np.array(y_complete_norm)
y_partial_norm = np.array(y_partial_norm)
y_complete_norm_masked = np.array(y_complete_norm_masked)

# Shuffle data
from sklearn.utils import shuffle
X_norm, y_complete_norm_masked, y_partial_norm, max_dates = shuffle(
    X_norm, y_complete_norm_masked, y_partial_norm, max_dates, random_state=42
)

# --- 3. Train/test split ---
X_train, X_test, y_train, y_test, y_partial_train, y_partial_test, max_train, max_test = train_test_split(
    X_norm, y_complete_norm_masked, y_partial_norm, max_dates, test_size=0.3, random_state=42
)

# --- 4. Data Generator ---
class TreeDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for training on tree vectors and partial date vectors.

    Args:
        X (np.ndarray): Input tree vectors.
        y_partial (np.ndarray): Partial date vectors (with leaf dates only).
        y_complete (np.ndarray): Full target date vectors (masked).
        max_dates (List[float]): Max date per sample for denormalization.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle data at each epoch.
    """
    def __init__(self, X, y_partial, y_complete, max_dates, batch_size=32, shuffle=True):
        self.X = X
        self.y_partial = y_partial
        self.y_complete = y_complete
        self.max_dates = max_dates
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index: int):
        batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_idx]
        y_part_batch = self.y_partial[batch_idx]
        y_full_batch = self.y_complete[batch_idx]
        X_input = np.concatenate([X_batch, y_part_batch], axis=1)
        return X_input, y_full_batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --- 5. Masked loss ---
def masked_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes MSE loss ignoring positions where y_true is 0 (i.e. not to predict).

    Args:
        y_true (tf.Tensor): Ground truth tensor with masked positions.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: Masked mean squared error.
    """
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.square(y_true - y_pred) * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

# --- 6. Model definition ---
input_dim = X_train.shape[1] + y_partial_train.shape[1]
output_dim = y_train.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(output_dim)
])
model.compile(optimizer='adam', loss=masked_mse)

# --- 7. Generators ---
train_gen = TreeDataGenerator(X_train, y_partial_train, y_train, max_train, batch_size=32)
val_gen = TreeDataGenerator(X_test, y_partial_test, y_test, max_test, batch_size=32, shuffle=False)

# --- 8. Training ---
print("\nTraining model...")
model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

# --- 9. Predictions ---
X_input_test = np.concatenate([X_test, y_partial_test], axis=1)
y_pred_norm = model.predict(X_input_test)

# Denormalize predictions
y_pred, y_real = [], []
for pred, true_masked, yp, m in zip(y_pred_norm, y_test, y_partial_test, max_test):
    pred_denorm = m - (1 - pred)
    true_denorm = m - (1 - true_masked)
    yp_denorm = m - (1 - yp)
    leaf_mask = (yp > 0).astype(float)
    pred_final = pred_denorm * (1 - leaf_mask) + yp_denorm * leaf_mask
    true_final = true_denorm * (1 - leaf_mask) + yp_denorm * leaf_mask
    y_pred.append(pred_final)
    y_real.append(true_final)

y_pred = np.array(y_pred)
y_real = np.array(y_real)

# --- Print one sample ---
i = random.randint(1, 100)
print("\n--- Tree example", i, "---")
print("True vector:")
print(np.round(y_real[i], 4))
print("\nPredicted vector:")
print(np.round(y_pred[i], 4))

# --- 10. Custom metrics ---
def compute_custom_metrics(true_mat: np.ndarray, pred_mat: np.ndarray):
    """
    Computes RRE (Relative Root Error) and NRMSE for internal nodes.

    Args:
        true_mat (np.ndarray): Ground truth date vectors.
        pred_mat (np.ndarray): Predicted date vectors.

    Returns:
        Tuple[float, float]: Mean RRE and mean NRMSE_internal.
    """
    rres = []
    nrmse_vals = []
    for true_vec, pred_vec in zip(true_mat, pred_mat):
        t_root_true = true_vec[-1]
        t_root_pred = pred_vec[-1]
        H = np.max(true_vec) - np.min(true_vec)
        RRE = abs(t_root_pred - t_root_true) / H if H > 0 else 0
        rres.append(RRE)

        sum_error = 0
        for internal_idx in range(len(true_vec) - 1):  # root excluded
            if true_vec[internal_idx] != 0 and true_vec[internal_idx] != pred_vec[internal_idx]:
                sum_error += abs(pred_vec[internal_idx] - true_vec[internal_idx]) / H
        NRMSE = sum_error / (len(true_vec) - 1)
        nrmse_vals.append(NRMSE)

    return np.mean(rres), np.mean(nrmse_vals)

# Apply custom metrics
rre_mean, nrmse_internal_mean = compute_custom_metrics(y_real, y_pred)
print(f"Mean RRE (root) : {rre_mean:.6f}")
print(f"Mean NRMSE internal nodes : {nrmse_internal_mean:.6f}")

# Save model
model.save("deepl_dating_model.keras")
print("Model saved to deepl_dating_model.keras")

