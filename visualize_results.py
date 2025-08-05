#!/usr/bin/env python3
"""
Visualization script for LLM benchmark results
Creates charts and analysis from benchmark_results.json
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BenchmarkVisualizer:
    def __init__(self, results_file: str = "benchmark_results.json"):
        self.results_file = results_file
        self.data = None
        self.df = None
        self.load_data()

    def load_data(self):
        """Load benchmark results from JSON file"""
        try:
            with open(self.results_file, "r") as f:
                self.data = json.load(f)

            # Convert to pandas DataFrame
            results = self.data.get("results", [])
            self.df = pd.DataFrame(results)

            print(f"✅ Loaded {len(self.df)} benchmark results")

        except FileNotFoundError:
            print(f"❌ Results file not found: {self.results_file}")
            print("Run the benchmark first with: python main.py")
            return
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {self.results_file}")
            return

    def create_performance_dashboard(self, save_path: str = "benchmark_dashboard.png"):
        """Create a comprehensive performance dashboard"""
        if self.df is None or self.df.empty:
            print("❌ No data to visualize")
            return

        # Set up the plot style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "LLM Benchmark Performance Dashboard", fontsize=16, fontweight="bold"
        )

        # 1. Latency comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(
            self.df["model_name"],
            self.df["latency_ms"],
            color=sns.color_palette("husl", len(self.df)),
        )
        ax1.set_title("Latency (ms) - Lower is Better")
        ax1.set_ylabel("Latency (ms)")
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # 2. Throughput comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(
            self.df["model_name"],
            self.df["throughput_tokens_per_sec"],
            color=sns.color_palette("husl", len(self.df)),
        )
        ax2.set_title("Throughput (tokens/sec) - Higher is Better")
        ax2.set_ylabel("Throughput (tokens/sec)")
        ax2.tick_params(axis="x", rotation=45)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        # 3. Accuracy comparison
        ax3 = axes[0, 2]
        bars3 = ax3.bar(
            self.df["model_name"],
            self.df["accuracy_score"],
            color=sns.color_palette("husl", len(self.df)),
        )
        ax3.set_title("Accuracy Score - Higher is Better")
        ax3.set_ylabel("Accuracy Score")
        ax3.tick_params(axis="x", rotation=45)

        for bar in bars3:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        # 4. Perplexity comparison
        ax4 = axes[1, 0]
        bars4 = ax4.bar(
            self.df["model_name"],
            self.df["perplexity"],
            color=sns.color_palette("husl", len(self.df)),
        )
        ax4.set_title("Perplexity - Lower is Better")
        ax4.set_ylabel("Perplexity")
        ax4.tick_params(axis="x", rotation=45)

        for bar in bars4:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

        # 5. Memory usage comparison
        ax5 = axes[1, 1]
        bars5 = ax5.bar(
            self.df["model_name"],
            self.df["memory_usage_mb"],
            color=sns.color_palette("husl", len(self.df)),
        )
        ax5.set_title("Memory Usage (MB) - Lower is Better")
        ax5.set_ylabel("Memory (MB)")
        ax5.tick_params(axis="x", rotation=45)

        for bar in bars5:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )

        # 6. Radar chart for overall performance
        ax6 = axes[1, 2]

        # Normalize metrics for radar chart (0-1 scale)
        metrics = [
            "latency_ms",
            "throughput_tokens_per_sec",
            "accuracy_score",
            "perplexity",
            "memory_usage_mb",
        ]

        # Normalize data (invert latency and perplexity since lower is better)
        normalized_data = self.df.copy()
        normalized_data["latency_norm"] = 1 - (
            normalized_data["latency_ms"] / normalized_data["latency_ms"].max()
        )
        normalized_data["throughput_norm"] = (
            normalized_data["throughput_tokens_per_sec"]
            / normalized_data["throughput_tokens_per_sec"].max()
        )
        normalized_data["accuracy_norm"] = (
            normalized_data["accuracy_score"] / normalized_data["accuracy_score"].max()
        )
        normalized_data["perplexity_norm"] = 1 - (
            normalized_data["perplexity"] / normalized_data["perplexity"].max()
        )
        normalized_data["memory_norm"] = 1 - (
            normalized_data["memory_usage_mb"]
            / normalized_data["memory_usage_mb"].max()
        )

        # Create radar chart
        categories = ["Latency", "Throughput", "Accuracy", "Perplexity", "Memory"]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for idx, model in enumerate(self.df["model_name"]):
            values = [
                normalized_data.iloc[idx]["latency_norm"],
                normalized_data.iloc[idx]["throughput_norm"],
                normalized_data.iloc[idx]["accuracy_norm"],
                normalized_data.iloc[idx]["perplexity_norm"],
                normalized_data.iloc[idx]["memory_norm"],
            ]
            values += values[:1]  # Complete the circle

            ax6.plot(angles, values, "o-", linewidth=2, label=model)
            ax6.fill(angles, values, alpha=0.25)

        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title("Overall Performance Radar Chart")
        ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Dashboard saved to: {save_path}")

        plt.show()

    def create_comparison_table(self):
        """Create a detailed comparison table"""
        if self.df is None or self.df.empty:
            print("❌ No data to analyze")
            return

        print("\n" + "=" * 80)
        print("DETAILED MODEL COMPARISON")
        print("=" * 80)

        # Calculate rankings
        rankings = {}
        metrics = [
            "latency_ms",
            "throughput_tokens_per_sec",
            "accuracy_score",
            "perplexity",
            "memory_usage_mb",
        ]

        for metric in metrics:
            if metric in ["latency_ms", "perplexity", "memory_usage_mb"]:
                # Lower is better
                rankings[metric] = self.df[metric].rank(ascending=True)
            else:
                # Higher is better
                rankings[metric] = self.df[metric].rank(ascending=False)

        # Create comparison table
        comparison_df = self.df.copy()
        for metric in metrics:
            comparison_df[f"{metric}_rank"] = rankings[metric]

        # Display table
        display_cols = [
            "model_name",
            "latency_ms",
            "throughput_tokens_per_sec",
            "accuracy_score",
            "perplexity",
            "memory_usage_mb",
        ]

        print(comparison_df[display_cols].to_string(index=False, float_format="%.2f"))

        # Show rankings
        print("\n" + "=" * 50)
        print("RANKINGS (1 = Best)")
        print("=" * 50)

        rank_cols = ["model_name"] + [f"{metric}_rank" for metric in metrics]
        rank_df = comparison_df[rank_cols].copy()
        rank_df.columns = [
            "Model",
            "Latency Rank",
            "Throughput Rank",
            "Accuracy Rank",
            "Perplexity Rank",
            "Memory Rank",
        ]

        print(rank_df.to_string(index=False))

        # Calculate overall score
        rank_df["Overall Score"] = rank_df.iloc[:, 1:].sum(axis=1)
        rank_df = rank_df.sort_values("Overall Score")

        print("\n🏆 OVERALL RANKING (Lower Score = Better)")
        print("=" * 50)
        for idx, row in rank_df.iterrows():
            print(f"{row['Overall Score']:.1f} - {row['Model']}")

    def generate_insights(self):
        """Generate insights from the benchmark data"""
        if self.df is None or self.df.empty:
            print("❌ No data to analyze")
            return

        print("\n" + "=" * 60)
        print("BENCHMARK INSIGHTS")
        print("=" * 60)

        # Best performers
        fastest = self.df.loc[self.df["latency_ms"].idxmin()]
        highest_throughput = self.df.loc[self.df["throughput_tokens_per_sec"].idxmax()]
        most_accurate = self.df.loc[self.df["accuracy_score"].idxmax()]
        lowest_perplexity = self.df.loc[self.df["perplexity"].idxmin()]
        most_efficient = self.df.loc[self.df["memory_usage_mb"].idxmin()]

        print(
            f"🏃 Fastest Model: {fastest['model_name']} ({fastest['latency_ms']:.1f}ms)"
        )
        print(
            f"⚡ Highest Throughput: {highest_throughput['model_name']} ({highest_throughput['throughput_tokens_per_sec']:.2f} tokens/sec)"
        )
        print(
            f"🎯 Most Accurate: {most_accurate['model_name']} ({most_accurate['accuracy_score']:.3f})"
        )
        print(
            f"🧠 Lowest Perplexity: {lowest_perplexity['model_name']} ({lowest_perplexity['perplexity']:.2f})"
        )
        print(
            f"💾 Most Memory Efficient: {most_efficient['model_name']} ({most_efficient['memory_usage_mb']:.0f}MB)"
        )

        # Performance correlations
        print("\n📈 PERFORMANCE CORRELATIONS:")

        # Latency vs Throughput
        latency_throughput_corr = self.df["latency_ms"].corr(
            self.df["throughput_tokens_per_sec"]
        )
        print(f"   Latency vs Throughput: {latency_throughput_corr:.3f}")

        # Accuracy vs Perplexity
        accuracy_perplexity_corr = self.df["accuracy_score"].corr(self.df["perplexity"])
        print(f"   Accuracy vs Perplexity: {accuracy_perplexity_corr:.3f}")

        # Memory vs Performance
        memory_latency_corr = self.df["memory_usage_mb"].corr(self.df["latency_ms"])
        print(f"   Memory vs Latency: {memory_latency_corr:.3f}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")

        if len(self.df) >= 3:
            # Find balanced performer (closest to median in all metrics)
            median_latency = self.df["latency_ms"].median()
            median_throughput = self.df["throughput_tokens_per_sec"].median()
            median_accuracy = self.df["accuracy_score"].median()

            balanced_scores = []
            for idx, row in self.df.iterrows():
                score = (
                    abs(row["latency_ms"] - median_latency) / median_latency
                    + abs(row["throughput_tokens_per_sec"] - median_throughput)
                    / median_throughput
                    + abs(row["accuracy_score"] - median_accuracy) / median_accuracy
                )
                balanced_scores.append((row["model_name"], score))

            balanced_scores.sort(key=lambda x: x[1])
            most_balanced = balanced_scores[0][0]
            print(f"   Most Balanced Performer: {most_balanced}")

        # Cost-performance analysis
        print("\n💰 COST-PERFORMANCE ANALYSIS:")
        for idx, row in self.df.iterrows():
            efficiency_score = (
                row["throughput_tokens_per_sec"] * row["accuracy_score"]
            ) / row["memory_usage_mb"]
            print(
                f"   {row['model_name']}: {efficiency_score:.4f} (tokens/sec/accuracy/MB)"
            )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visualize LLM benchmark results")
    parser.add_argument(
        "--file",
        default="benchmark_results.json",
        help="Path to benchmark results JSON file",
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Generate performance dashboard"
    )
    parser.add_argument("--table", action="store_true", help="Show comparison table")
    parser.add_argument(
        "--insights", action="store_true", help="Generate insights and recommendations"
    )
    parser.add_argument("--all", action="store_true", help="Run all visualizations")

    args = parser.parse_args()

    visualizer = BenchmarkVisualizer(args.file)

    if args.all or args.dashboard:
        visualizer.create_performance_dashboard()

    if args.all or args.table:
        visualizer.create_comparison_table()

    if args.all or args.insights:
        visualizer.generate_insights()

    if not any([args.dashboard, args.table, args.insights, args.all]):
        # Default: run all
        visualizer.create_performance_dashboard()
        visualizer.create_comparison_table()
        visualizer.generate_insights()


if __name__ == "__main__":
    main()
