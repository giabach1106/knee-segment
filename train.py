import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

import torch
import matplotlib.pyplot as plt

from src.utils import setup_logging, ensure_reproducibility
from src.training import ExperimentConfig, run_experiment, create_standard_experiments
from src.visualization import plot_metrics_comparison, create_experiment_comparison_plot


def run_single_experiment(
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    disable_wandb: bool = False
) -> Dict:
    """Run a single experiment.
    
    Args:
        config_path: Path to experiment configuration JSON file.
        experiment_name: Name of standard experiment to run.
        device: Device to use ('cuda', 'cpu', or None for auto).
        
    Returns:
        Experiment results dictionary.
    """
    # Load or create config
    if config_path:
        config = ExperimentConfig.load(config_path)
    elif experiment_name:
        # Get from standard experiments
        standard_configs = {c.name: c for c in create_standard_experiments()}
        if experiment_name not in standard_configs:
            raise ValueError(
                f"Unknown experiment: {experiment_name}. "
                f"Available: {list(standard_configs.keys())}"
            )
        config = standard_configs[experiment_name]
    else:
        raise ValueError("Must provide either config_path or experiment_name")
        
    # Set device
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Update wandb config if specified
    if wandb_project:
        config.wandb_config["project"] = wandb_project
    if disable_wandb:
        config.wandb_config["use_wandb"] = False
        
    # Run experiment
    results = run_experiment(config, device=device)
    
    return results


def run_all_experiments(
    experiments: Optional[List[str]] = None,
    device: Optional[str] = None,
    skip_existing: bool = True,
    wandb_project: Optional[str] = None,
    disable_wandb: bool = False
) -> Dict[str, Dict]:
    """Run all or selected standard experiments.
    
    Args:
        experiments: List of experiment names to run. If None, runs all.
        device: Device to use.
        skip_existing: Whether to skip experiments that already have results.
        
    Returns:
        Dictionary mapping experiment names to results.
    """
    # Get standard experiments
    all_configs = create_standard_experiments()
    
    # Filter if specific experiments requested
    if experiments:
        configs_to_run = [c for c in all_configs if c.name in experiments]
    else:
        configs_to_run = all_configs
        
    # Set device
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    results = {}
    
    for config in configs_to_run:
        experiment_dir = Path("experiments") / config.name
        results_file = experiment_dir / "results_summary.json"
        
        # Skip if exists and skip_existing is True
        if skip_existing and results_file.exists():
            logging.info(f"Skipping {config.name} - results already exist")
            with open(results_file, 'r') as f:
                results[config.name] = json.load(f)
            continue
            
        logging.info(f"\n{'='*60}")
        logging.info(f"Running experiment: {config.name}")
        logging.info(f"{'='*60}\n")
        
        try:
            # Update wandb config if specified
            if wandb_project:
                config.wandb_config["project"] = wandb_project
            if disable_wandb:
                config.wandb_config["use_wandb"] = False
                
            result = run_experiment(config, device=device)
            results[config.name] = result
        except Exception as e:
            logging.error(f"Experiment {config.name} failed: {str(e)}")
            
    return results


def compare_experiments(
    experiment_names: Optional[List[str]] = None,
    output_dir: str = "results"
) -> None:
    """Compare results across experiments.
    
    Args:
        experiment_names: List of experiment names to compare. If None, uses all.
        output_dir: Directory to save comparison plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Collect results
    experiments_dir = Path("experiments")
    all_results = {}
    
    # Get list of experiments
    if experiment_names:
        exp_dirs = [experiments_dir / name for name in experiment_names]
    else:
        exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
        
    # Load results
    for exp_dir in exp_dirs:
        metrics_file = exp_dir / "test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_results[exp_dir.name] = metrics
        else:
            logging.warning(f"No test metrics found for {exp_dir.name}")
            
    if not all_results:
        logging.error("No experiment results found to compare")
        return
        
    # Create comparison plots
    logging.info(f"Comparing {len(all_results)} experiments")
    
    # Basic metrics comparison
    fig = plot_metrics_comparison(
        all_results,
        title="Preprocessing Methods Comparison",
        save_path=output_dir / "metrics_comparison.png"
    )
    plt.close(fig)
    
    # Detailed comparison with baseline
    if "baseline" in all_results:
        baseline_metrics = all_results["baseline"]
        other_experiments = {k: v for k, v in all_results.items() if k != "baseline"}
        
        fig = create_experiment_comparison_plot(
            baseline_metrics,
            other_experiments,
            save_path=output_dir / "detailed_comparison.png"
        )
        plt.close(fig)
        
    # Save results table
    results_table = []
    for exp_name, metrics in all_results.items():
        results_table.append({
            "Experiment": exp_name,
            "Dice (%)": f"{metrics.get('dice_percent', 0):.2f}",
            "IoU (%)": f"{metrics.get('iou_percent', 0):.2f}",
            "Pixel Accuracy (%)": f"{metrics.get('pixel_accuracy_percent', 0):.2f}"
        })
        
    # Sort by Dice score
    results_table.sort(key=lambda x: float(x["Dice (%)"].rstrip('%')), reverse=True)
    
    # Save as JSON
    with open(output_dir / "results_table.json", 'w') as f:
        json.dump(results_table, f, indent=2)
        
    # Print results
    print("\nExperiment Results Summary:")
    print("-" * 60)
    print(f"{'Experiment':<25} {'Dice (%)':<10} {'IoU (%)':<10} {'PA (%)':<10}")
    print("-" * 60)
    for row in results_table:
        print(f"{row['Experiment']:<25} {row['Dice (%)']:<10} "
              f"{row['IoU (%)']:<10} {row['Pixel Accuracy (%)']:<10}")
        
    logging.info(f"Comparison results saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train U-Net models for knee ultrasound segmentation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single experiment
    single_parser = subparsers.add_parser(
        "single", help="Run a single experiment"
    )
    single_parser.add_argument(
        "--config", type=str, help="Path to experiment config JSON"
    )
    single_parser.add_argument(
        "--name", type=str, help="Name of standard experiment"
    )
    single_parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    single_parser.add_argument(
        "--wandb-project", type=str,
        help="W&B project name (default: knee-ultrasound-segmentation)"
    )
    single_parser.add_argument(
        "--disable-wandb", action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # All experiments
    all_parser = subparsers.add_parser(
        "all", help="Run all standard experiments"
    )
    all_parser.add_argument(
        "--experiments", nargs="+", help="Specific experiments to run"
    )
    all_parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    all_parser.add_argument(
        "--no-skip", action="store_true",
        help="Don't skip existing results"
    )
    all_parser.add_argument(
        "--wandb-project", type=str,
        help="W&B project name (default: knee-ultrasound-segmentation)"
    )
    all_parser.add_argument(
        "--disable-wandb", action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # Compare
    compare_parser = subparsers.add_parser(
        "compare", help="Compare experiment results"
    )
    compare_parser.add_argument(
        "--experiments", nargs="+", help="Specific experiments to compare"
    )
    compare_parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for comparison plots"
    )
    
    # List experiments
    list_parser = subparsers.add_parser(
        "list", help="List available standard experiments"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir="logs", console=True)
    
    # Ensure reproducibility
    ensure_reproducibility(seed=42)
    
    # Execute command
    if args.command == "single":
        if not args.config and not args.name:
            parser.error("Must provide either --config or --name")
        run_single_experiment(
            config_path=args.config,
            experiment_name=args.name,
            device=args.device,
            wandb_project=args.wandb_project,
            disable_wandb=args.disable_wandb
        )
        
    elif args.command == "all":
        run_all_experiments(
            experiments=args.experiments,
            device=args.device,
            skip_existing=not args.no_skip,
            wandb_project=args.wandb_project,
            disable_wandb=args.disable_wandb
        )
        
    elif args.command == "compare":
        compare_experiments(
            experiment_names=args.experiments,
            output_dir=args.output
        )
        
    elif args.command == "list":
        configs = create_standard_experiments()
        print("\nAvailable experiments:")
        print("-" * 50)
        for config in configs:
            preproc = ", ".join([p["type"] for p in config.preprocessing]) or "None"
            aug = "Yes" if config.enable_augmentation else "No"
            print(f"{config.name:<25} Preprocessing: {preproc}, Augmentation: {aug}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 