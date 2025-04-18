import json
import os
import re
import hydra
from omegaconf import DictConfig
import sacrebleu


def parse_bleu(pred_file, ref_file):
    with open(pred_file, "r", encoding="utf-8") as f:
        predictions = [line.strip() for line in f.readlines()]

    with open(ref_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f.readlines()]

    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="none")
    return bleu.score


def parse_val_loss_from_log(log_dir="lightning_logs"):
    log_path = None
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".txt"):
                log_path = os.path.join(root, file)
                break
    if not log_path or not os.path.isfile(log_path):
        return None

    with open(log_path, "r") as f:
        lines = f.readlines()

    losses = []
    for line in lines:
        match = re.search(r"val_loss=([\d\.]+)", line)
        if match:
            losses.append(float(match.group(1)))

    return losses[-1] if losses else None


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    ref_file = cfg.data.test_tgt
    pred_file = "outputs/predictions.txt"

    bleu = parse_bleu(pred_file, ref_file)
    val_loss = parse_val_loss_from_log()

    metrics = {
        "bleu": round(bleu, 2),
    }

    if val_loss is not None:
        metrics["val_loss"] = round(val_loss, 4)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
