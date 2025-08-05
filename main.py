import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    accuracy_score: float
    perplexity: float
    timestamp: str
    model_size_gb: Optional[float] = None
    parameters: Optional[int] = None


@dataclass
class BenchmarkConfig:
    models: List[str]
    prompt_length: int = 512
    max_new_tokens: int = 128
    temperature: float = 0.7
    batch_size: int = 1
    num_runs: int = 5
    warmup_runs: int = 2


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []

    def generate(
        self, model: str, prompt: str, max_tokens: int = 128, temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using Ollama"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        start_time = time.time()
        response = self.session.post(f"{self.base_url}/api/generate", json=payload)
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            return {
                "response": data.get("response", ""),
                "latency": (end_time - start_time) * 1000,  # Convert to milliseconds
                "tokens_generated": len(data.get("response", "").split()),
                "success": True,
            }
        else:
            return {
                "response": "",
                "latency": 0,
                "tokens_generated": 0,
                "success": False,
                "error": response.text,
            }


class LLMBenchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.ollama_client = OllamaClient()
        self.results: List[BenchmarkResult] = []

        # Test prompts for different tasks
        self.test_prompts = {
            "text_generation": "The future of artificial intelligence holds great promise for",
            "code_generation": "def fibonacci(n):",
            "question_answering": "What is the capital of France?",
            "summarization": "Summarize the following text: Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit programming.",
            "translation": "Translate the following English text to French: Hello, how are you today?",
        }

    def calculate_perplexity(self, model_name: str, text: str) -> float:
        """Calculate perplexity for a given text (simplified implementation)"""
        try:
            # This is a simplified perplexity calculation
            # In practice, you'd want to use the model's actual tokenizer and logits
            words = text.split()
            if len(words) < 2:
                return 1.0

            # Simple n-gram based perplexity approximation
            bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
            unique_bigrams = len(set(bigrams))
            total_bigrams = len(bigrams)

            if total_bigrams == 0:
                return 1.0

            # Simplified perplexity calculation
            perplexity = np.exp(-np.log(unique_bigrams / total_bigrams))
            return min(perplexity, 100.0)  # Cap at reasonable value
        except:
            return 50.0  # Default fallback

    def calculate_accuracy_score(self, response: str, task_type: str) -> float:
        """Calculate accuracy score based on response quality"""
        if not response.strip():
            return 0.0

        # Simple heuristics for different task types
        if task_type == "text_generation":
            # Check for coherent continuation
            return min(1.0, len(response.split()) / 10.0)
        elif task_type == "code_generation":
            # Check for valid Python syntax indicators
            if "def" in response or "return" in response or ":" in response:
                return 0.8
            return 0.3
        elif task_type == "question_answering":
            # Check for relevant keywords
            keywords = ["paris", "france", "capital"]
            response_lower = response.lower()
            matches = sum(1 for keyword in keywords if keyword in response_lower)
            return min(1.0, matches / len(keywords))
        elif task_type == "summarization":
            # Check for summary-like characteristics
            if len(response.split()) < 50 and any(
                word in response.lower() for word in ["machine", "learning", "ai"]
            ):
                return 0.7
            return 0.4
        else:
            return 0.5

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0

    async def benchmark_model(self, model_name: str) -> Optional[BenchmarkResult]:
        """Benchmark a single model"""
        logger.info(f"Benchmarking model: {model_name}")

        latencies = []
        throughputs = []
        accuracies = []
        perplexities = []

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            for task_type, prompt in self.test_prompts.items():
                result = self.ollama_client.generate(
                    model_name,
                    prompt,
                    self.config.max_new_tokens,
                    self.config.temperature,
                )
                if not result["success"]:
                    logger.warning(
                        f"Warmup failed for {model_name}: {result.get('error', 'Unknown error')}"
                    )

        # Actual benchmark runs
        for run in range(self.config.num_runs):
            run_latencies = []
            run_throughputs = []
            run_accuracies = []
            run_perplexities = []

            for task_type, prompt in self.test_prompts.items():
                result = self.ollama_client.generate(
                    model_name,
                    prompt,
                    self.config.max_new_tokens,
                    self.config.temperature,
                )

                if result["success"]:
                    latency = result["latency"]
                    tokens_generated = result["tokens_generated"]
                    response = result["response"]

                    # Calculate metrics
                    throughput = (
                        (tokens_generated / (latency / 1000)) if latency > 0 else 0
                    )
                    accuracy = self.calculate_accuracy_score(response, task_type)
                    perplexity = self.calculate_perplexity(model_name, response)

                    run_latencies.append(latency)
                    run_throughputs.append(throughput)
                    run_accuracies.append(accuracy)
                    run_perplexities.append(perplexity)
                else:
                    logger.error(
                        f"Generation failed for {model_name}: {result.get('error', 'Unknown error')}"
                    )
                    return None

            # Average metrics for this run
            if run_latencies:
                latencies.append(statistics.mean(run_latencies))
                throughputs.append(statistics.mean(run_throughputs))
                accuracies.append(statistics.mean(run_accuracies))
                perplexities.append(statistics.mean(run_perplexities))

        if not latencies:
            logger.error(f"No successful runs for model {model_name}")
            return None

        # Calculate final metrics
        avg_latency = statistics.mean(latencies)
        avg_throughput = statistics.mean(throughputs)
        avg_accuracy = statistics.mean(accuracies)
        avg_perplexity = statistics.mean(perplexities)
        memory_usage = self.get_memory_usage()

        return BenchmarkResult(
            model_name=model_name,
            latency_ms=avg_latency,
            throughput_tokens_per_sec=avg_throughput,
            memory_usage_mb=memory_usage,
            accuracy_score=avg_accuracy,
            perplexity=avg_perplexity,
            timestamp=datetime.now().isoformat(),
        )

    async def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run benchmarks for all models"""
        if not self.ollama_client.is_available():
            logger.error("Ollama is not available. Please start Ollama first.")
            return []

        available_models = self.ollama_client.list_models()
        logger.info(f"Available models: {available_models}")

        # Filter models based on config
        models_to_test = [
            model for model in self.config.models if model in available_models
        ]

        if not models_to_test:
            logger.error(
                f"No configured models found in available models: {available_models}"
            )
            return []

        logger.info(f"Benchmarking models: {models_to_test}")

        # Run benchmarks sequentially (could be parallelized)
        for model in models_to_test:
            result = await self.benchmark_model(model)
            if result:
                self.results.append(result)
                logger.info(f"Completed benchmark for {model}")

        return self.results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file"""
        results_dict = [asdict(result) for result in self.results]

        with open(filename, "w") as f:
            json.dump(
                {
                    "benchmark_config": asdict(self.config),
                    "results": results_dict,
                    "summary": self.generate_summary(),
                },
                f,
                indent=2,
            )

        logger.info(f"Results saved to {filename}")

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.results:
            return {}

        latencies = [r.latency_ms for r in self.results]
        throughputs = [r.throughput_tokens_per_sec for r in self.results]
        accuracies = [r.accuracy_score for r in self.results]
        perplexities = [r.perplexity for r in self.results]

        return {
            "fastest_model": min(self.results, key=lambda x: x.latency_ms).model_name,
            "highest_throughput": max(
                self.results, key=lambda x: x.throughput_tokens_per_sec
            ).model_name,
            "highest_accuracy": max(
                self.results, key=lambda x: x.accuracy_score
            ).model_name,
            "lowest_perplexity": min(
                self.results, key=lambda x: x.perplexity
            ).model_name,
            "avg_latency_ms": statistics.mean(latencies),
            "avg_throughput": statistics.mean(throughputs),
            "avg_accuracy": statistics.mean(accuracies),
            "avg_perplexity": statistics.mean(perplexities),
        }

    def print_results(self):
        """Print benchmark results in a formatted table"""
        if not self.results:
            print("No benchmark results available.")
            return

        print("\n" + "=" * 80)
        print("LLM BENCHMARK RESULTS")
        print("=" * 80)

        # Print summary
        summary = self.generate_summary()
        print("\nSUMMARY:")
        print(f"  Fastest Model: {summary['fastest_model']}")
        print(f"  Highest Throughput: {summary['highest_throughput']}")
        print(f"  Highest Accuracy: {summary['highest_accuracy']}")
        print(f"  Lowest Perplexity: {summary['lowest_perplexity']}")

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print(
            f"{'Model':<20} {'Latency(ms)':<12} {'Throughput':<12} {'Accuracy':<10} {'Perplexity':<12} {'Memory(MB)':<12}"
        )
        print("-" * 80)

        for result in sorted(self.results, key=lambda x: x.latency_ms):
            print(
                f"{result.model_name:<20} {result.latency_ms:<12.2f} {result.throughput_tokens_per_sec:<12.2f} "
                f"{result.accuracy_score:<10.3f} {result.perplexity:<12.2f} {result.memory_usage_mb:<12.2f}"
            )


async def main():
    """Main function to run LLM benchmarks"""
    print("Starting LLM Benchmark Research")
    print("=" * 50)

    # Check if PyTorch is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    # Configure benchmark
    config = BenchmarkConfig(
        models=[
            "llama2:7b",
            "llama2:13b",
            "mistral:7b",
            "codellama:7b",
            "neural-chat:7b",
        ],
        prompt_length=512,
        max_new_tokens=128,
        temperature=0.7,
        num_runs=3,
        warmup_runs=1,
    )

    # Create benchmarker and run tests
    benchmarker = LLMBenchmarker(config)

    print(f"\n📊 Benchmarking {len(config.models)} models...")
    print(f"   Models: {', '.join(config.models)}")
    print(f"   Runs per model: {config.num_runs}")
    print(f"   Max tokens: {config.max_new_tokens}")

    results = await benchmarker.run_benchmarks()

    if results:
        benchmarker.print_results()
        benchmarker.save_results()

        print("\n Benchmark completed successfully!")
        print("   Results saved to: benchmark_results.json")
        print(f"   Total models tested: {len(results)}")
    else:
        print(
            "\n No benchmark results generated. Check if Ollama is running and models are available."
        )


if __name__ == "__main__":
    asyncio.run(main())
