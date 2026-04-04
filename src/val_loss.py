import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_val_loss(log_file: str) -> list[tuple[int, float]]:
    """Parse val loss entries from MLX training log file."""
    pattern = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")
    results = []

    with Path(log_file).open() as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iteration = int(match.group(1))
                val_loss = float(match.group(2))
                results.append((iteration, val_loss))

    return results


def print_table(results: list[tuple[int, float]]) -> None:
    """Print val loss as a formatted table."""
    logger.info("\n%8s %10s %10s", "Iter", "Val Loss", "Change")
    logger.info("-" * 32)

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

        logger.info("%8d %10.4f %10s%s", iter_num, val_loss, change, marker)
        prev_loss = val_loss

    logger.info("-" * 32)
    logger.info("\nBest checkpoint: iter %d with val loss %.4f", best_iter, best_loss)
    logger.info(
        "Recommendation: load adapter from iter %d, not the final checkpoint", best_iter
    )


def plot_ascii(results: list[tuple[int, float]]) -> None:
    """Plot a simple ASCII val loss curve."""
    if not results:
        return

    losses = [r[1] for r in results]
    iters = [r[0] for r in results]
    min_loss = min(losses)
    max_loss = max(losses)
    height = 20
    width = min(len(results), 60)

    logger.info("\nVal Loss Curve (%.3f - %.3f)", min_loss, max_loss)
    logger.info("=" * (width + 10))

    # Sample evenly if too many points
    step = max(1, len(results) // width)
    sampled = results[::step]

    for row in range(height, -1, -1):
        threshold = min_loss + (max_loss - min_loss) * row / height
        line = ""
        for _, loss in sampled:
            line += "█" if loss >= threshold else " "
        loss_label = f"{threshold:6.3f} |" if row % 4 == 0 else "       |"
        logger.info("%s%s", loss_label, line)

    logger.info("       +%s", "-" * len(sampled))
    logger.info(
        "        iter %d %s iter %d",
        iters[0],
        " " * (len(sampled) - 20),
        iters[-1],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse and display MLX val loss from log file"
    )
    parser.add_argument("log_file", help="Path to MLX training log file")
    parser.add_argument("--no-plot", action="store_true", help="Skip ASCII plot")
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        logger.info(Path.cwd())
        logger.info("Error: log file '%s' not found", args.log_file)
        sys.exit(1)

    results = parse_val_loss(args.log_file)

    if not results:
        logger.info(
            "No val loss entries found. Make sure you piped MLX training output to the log file."
        )
        logger.info("Example: mlx_lm.lora --config config.yaml 2>&1 | tee training.log")
        sys.exit(1)

    logger.info("Found %d val loss checkpoints", len(results))
    print_table(results)

    if not args.no_plot:
        plot_ascii(results)
