import re
import sys
import argparse
from pathlib import Path


def parse_val_loss(log_file: str) -> list[tuple[int, float]]:
    """Parse val loss entries from MLX training log file."""
    pattern = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")
    results = []

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                val_loss = float(match.group(2))
                results.append((iteration, val_loss))

    return results


def print_table(results: list[tuple[int, float]]) -> None:
    """Print val loss as a formatted table."""
    print(f"\n{'Iter':>8} {'Val Loss':>10} {'Change':>10}")
    print("-" * 32)

    prev_loss = None
    best_iter, best_loss = None, float("inf")

    for iter_num, val_loss in results:
        change = ""
        if prev_loss is not None:
            delta = val_loss - prev_loss
            arrow = "↑" if delta > 0 else "↓"
            change = f"{arrow} {abs(delta):.4f}"

        marker = " ◀ best" if val_loss < best_loss else ""
        if val_loss < best_loss:
            best_loss = val_loss
            best_iter = iter_num

        print(f"{iter_num:>8} {val_loss:>10.4f} {change:>10}{marker}")
        prev_loss = val_loss

    print("-" * 32)
    print(f"\nBest checkpoint: iter {best_iter} with val loss {best_loss:.4f}")
    print(f"Recommendation: load adapter from iter {best_iter}, not the final checkpoint")


def plot_ascii(results: list[tuple[int, float]]) -> None:
    """Simple ASCII plot of val loss curve."""
    if not results:
        return

    losses = [r[1] for r in results]
    iters = [r[0] for r in results]
    min_loss = min(losses)
    max_loss = max(losses)
    height = 20
    width = min(len(results), 60)

    print(f"\nVal Loss Curve ({min_loss:.3f} - {max_loss:.3f})")
    print("=" * (width + 10))

    # Sample evenly if too many points
    step = max(1, len(results) // width)
    sampled = results[::step]

    for row in range(height, -1, -1):
        threshold = min_loss + (max_loss - min_loss) * row / height
        line = ""
        for _, loss in sampled:
            line += "█" if loss >= threshold else " "
        loss_label = f"{threshold:6.3f} |" if row % 4 == 0 else "       |"
        print(f"{loss_label}{line}")

    print("       +" + "-" * len(sampled))
    print(f"        iter {iters[0]} {'':>{len(sampled)-20}} iter {iters[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and display MLX val loss from log file")
    parser.add_argument("log_file", help="Path to MLX training log file")
    parser.add_argument("--no-plot", action="store_true", help="Skip ASCII plot")
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(Path.cwd())
        print(f"Error: log file '{args.log_file}' not found")
        sys.exit(1)

    results = parse_val_loss(args.log_file)

    if not results:
        print("No val loss entries found. Make sure you piped MLX training output to the log file.")
        print("Example: mlx_lm.lora --config config.yaml 2>&1 | tee training.log")
        sys.exit(1)

    print(f"Found {len(results)} val loss checkpoints")
    print_table(results)

    if not args.no_plot:
        plot_ascii(results)