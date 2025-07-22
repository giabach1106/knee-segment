import argparse
import json
from pathlib import Path
import logging
import torch
import matplotlib.pyplot as plt

from src.utils import setup_logging, ensure_reproducibility
from src.training import ExperimentConfig, run_experiment, create_standard_experiments
from src.visualization import plot_metrics_comparison, create_experiment_comparison_plot


def run_single_experiment(config_path=None, experiment_name=None, device=None):
    """Run 1 experiment"""
    # Load config
    if config_path:
        config = ExperimentConfig.load(config_path)
    elif experiment_name:
        standard_configs = {c.name: c for c in create_standard_experiments()}
        config = standard_configs[experiment_name]
    else:
        raise ValueError("config_path or experiment_name")
        
    # Set device
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Run experiment
    results = run_experiment(config, device=device)
    return results


def run_all_experiments(experiments=None, device=None):
    """Run all experiments"""
    all_configs = create_standard_experiments()
    
    # Filter experiments
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
        
        # Skip nếu đã có results
        if results_file.exists():
            logging.info(f"Skipping {config.name} - already exists")
            with open(results_file, 'r') as f:
                results[config.name] = json.load(f)
            continue
            
        logging.info(f"Running experiment: {config.name}")
        
        try:
            result = run_experiment(config, device=device)
            results[config.name] = result
        except Exception as e:
            logging.error(f"Experiment {config.name} failed: {e}")
            
    return results


def compare_experiments(experiment_names=None, output_dir="results"):
    """So sánh results across experiments"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Collect results
    experiments_dir = Path("experiments")
    all_results = {}
    
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
    
    if not all_results:
        logging.error("No experiment results found")
        return
        
    # Create comparison plots
    logging.info(f"Comparing {len(all_results)} experiments")
    
    # Basic comparison
    fig = plot_metrics_comparison(
        all_results,
        title="Preprocessing Methods Comparison",
        save_path=output_dir / "metrics_comparison.png"
    )
    plt.close(fig)
    
    # Detailed comparison với baseline
    if "baseline" in all_results:
        baseline_metrics = all_results["baseline"]
        other_experiments = {k: v for k, v in all_results.items() if k != "baseline"}
        
        fig = create_experiment_comparison_plot(
            baseline_metrics,
            other_experiments,
            save_path=output_dir / "detailed_comparison.png"
        )
        plt.close(fig)
        
    # Results table
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
    
    # Save results
    with open(output_dir / "results_table.json", 'w') as f:
        json.dump(results_table, f, indent=2)
        
    # Print results
    print("\nExperiment Results:")
    print("-" * 60)
    print(f"{'Experiment':<25} {'Dice (%)':<10} {'IoU (%)':<10} {'PA (%)':<10}")
    print("-" * 60)
    for row in results_table:
        print(f"{row['Experiment']:<25} {row['Dice (%)']:<10} "
              f"{row['IoU (%)']:<10} {row['Pixel Accuracy (%)']:<10}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train UNet cho knee segmentation")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Single experiment
    single_parser = subparsers.add_parser("single", help="Run single experiment")
    single_parser.add_argument("--config", type=str, help="Config JSON path")
    single_parser.add_argument("--name", type=str, help="Standard experiment name")
    single_parser.add_argument("--device", type=str, choices=["cuda", "cpu"])
    
    # all experiments
    all_parser = subparsers.add_parser("all", help="Run all experiments")
    all_parser.add_argument("--experiments", nargs="+", help="Specific experiments")
    all_parser.add_argument("--device", type=str, choices=["cuda", "cpu"])
    
    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare results")
    compare_parser.add_argument("--experiments", nargs="+", help="Experiments to compare")
    compare_parser.add_argument("--output", type=str, default="results", help="Output dir")
    
    # list
    list_parser = subparsers.add_parser("list", help="List experiments")
    
    args = parser.parse_args()
    
    # setup
    setup_logging(log_dir="logs", console=True)
    ensure_reproducibility(seed=42)
    
    # execute
    if args.command == "single":
        run_single_experiment(
            config_path=args.config,
            experiment_name=args.name,
            device=args.device
        )
        
    elif args.command == "all":
        run_all_experiments(
            experiments=args.experiments,
            device=args.device
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
            print(f"{config.name:<25} Preproc: {preproc}, Aug: {aug}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 