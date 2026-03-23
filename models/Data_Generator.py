import os
import torch
from tqdm import tqdm

from config import DATA_DIR, DATA_CONFIG
from radon_transform import TheoreticalDataGenerator


def generate_and_save_dataset(filename, num_samples, generator, desc="Generating", seed_offset=0):
    """生成数据并保存为 .pt 文件，包含 coeff_true / g_observed / coeff_initial。"""
    print(f"\nStarted {desc}...")
    print(f"Target Samples: {num_samples}")

    coeff_true_list = []
    g_observed_list = []
    coeff_initial_list = []

    for i in tqdm(range(num_samples), desc=desc):
        seed = seed_offset + i
        c_true, _, g_obs, c_init = generator.generate_training_sample(random_seed=seed)
        coeff_true_list.append(c_true.detach().cpu())
        g_observed_list.append(g_obs.detach().cpu())
        coeff_initial_list.append(c_init.detach().cpu())

    data_dict = {
        "coeff_true": torch.stack(coeff_true_list).unsqueeze(1),
        "g_observed": torch.stack(g_observed_list),
        "coeff_initial": torch.stack(coeff_initial_list).unsqueeze(1),
    }

    save_path = os.path.join(DATA_DIR, filename)
    torch.save(data_dict, save_path)
    print(f"Saved {num_samples} samples to {save_path}")
    print(f"Final Tensor shape (GT): {data_dict['coeff_true'].shape}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_source = str(
        DATA_CONFIG.get("train_data_source", DATA_CONFIG.get("data_source", "random_ellipses"))
    ).strip().lower()
    val_source = str(
        DATA_CONFIG.get("val_data_source", DATA_CONFIG.get("data_source", "shepp_logan"))
    ).strip().lower()

    train_samples = int(os.environ.get("TRAIN_SAMPLES_OVERRIDE", "20000"))
    val_samples = int(os.environ.get("VAL_SAMPLES_OVERRIDE", "2000"))

    print(f"Train data source: {train_source}")
    print(f"Val data source: {val_source}")
    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")

    train_generator = TheoreticalDataGenerator(data_source=train_source)
    val_generator = TheoreticalDataGenerator(data_source=val_source)

    generate_and_save_dataset(
        "train_dataset.pt",
        train_samples,
        train_generator,
        desc=f"Train ({train_source})",
        seed_offset=0,
    )

    generate_and_save_dataset(
        "val_dataset.pt",
        val_samples,
        val_generator,
        desc=f"Val ({val_source})",
        seed_offset=1_000_000,
    )

    print("\nAll data generated successfully!")
    print("Now run: python train_offline.py")


if __name__ == "__main__":
    main()
