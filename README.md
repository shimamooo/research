# LLM Benchmark Research

A comprehensive PyTorch-based research tool for benchmarking Large Language Model (LLM) performance across multiple models using Ollama.

## Features

- **Multi-Model Benchmarking**: Test performance across various LLM models
- **Comprehensive Metrics**: Latency, throughput, accuracy, perplexity, and memory usage
- **Multiple Task Types**: Text generation, code generation, Q&A, summarization, and translation
- **Statistical Analysis**: Mean, standard deviation, and ranking of models
- **JSON Export**: Save detailed results for further analysis
- **Real-time Monitoring**: Progress tracking and logging

## Prerequisites

### 1. Install Ollama

First, install Ollama on your system:

**Windows:**

```bash
# Download from https://ollama.ai/download
# Or use winget:
winget install Ollama.Ollama
```

**macOS:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Pull Models

Pull the models you want to benchmark:

```bash
# Popular models to test
ollama pull llama2:7b
ollama pull llama2:13b
ollama pull mistral:7b
ollama pull codellama:7b
ollama pull neural-chat:7b
ollama pull qwen:7b
ollama pull phi:2.7b
```

## Installation

1. **Install Python dependencies:**

```bash
uv sync
```

2. **Verify installation:**

```bash
python main.py --help
```

## Usage

### Basic Benchmarking

Run the default benchmark with pre-configured models:

```bash
python main.py
```

### Custom Configuration

You can modify the benchmark configuration in `main.py`:

```python
config = BenchmarkConfig(
    models=[
        "llama2:7b",
        "mistral:7b",
        "codellama:7b"
    ],
    prompt_length=512,
    max_new_tokens=128,
    temperature=0.7,
    num_runs=5,
    warmup_runs=2
)
```

### Available Models

Common Ollama models you can benchmark:

- `llama2:7b` - Meta's Llama 2 7B parameter model
- `llama2:13b` - Meta's Llama 2 13B parameter model
- `mistral:7b` - Mistral AI's 7B model
- `codellama:7b` - Code-optimized Llama model
- `neural-chat:7b` - Intel's neural chat model
- `qwen:7b` - Alibaba's Qwen model
- `phi:2.7b` - Microsoft's Phi-2 model
- `gemma:2b` - Google's Gemma 2B model
- `gemma:7b` - Google's Gemma 7B model

## Benchmark Metrics

The system measures the following metrics:

### 1. **Latency (ms)**

- Time taken to generate response
- Lower is better

### 2. **Throughput (tokens/sec)**

- Tokens generated per second
- Higher is better

### 3. **Accuracy Score (0-1)**

- Task-specific quality assessment
- Higher is better

### 4. **Perplexity**

- Language model quality metric
- Lower is better

### 5. **Memory Usage (MB)**

- RAM consumption during inference
- Lower is better

## Task Types

The benchmark tests models on multiple tasks:

1. **Text Generation**: Continuation of prompts
2. **Code Generation**: Python function completion
3. **Question Answering**: Factual queries
4. **Summarization**: Text summarization
5. **Translation**: Language translation

## Output

### Console Output

```
🚀 Starting LLM Benchmark Research
==================================================
✅ CUDA available: NVIDIA GeForce RTX 4090

📊 Benchmarking 5 models...
   Models: llama2:7b, llama2:13b, mistral:7b, codellama:7b, neural-chat:7b
   Runs per model: 3
   Max tokens: 128

================================================================================
LLM BENCHMARK RESULTS
================================================================================

SUMMARY:
  Fastest Model: llama2:7b
  Highest Throughput: codellama:7b
  Highest Accuracy: neural-chat:7b
  Lowest Perplexity: llama2:13b

DETAILED RESULTS:
Model                Latency(ms)  Throughput   Accuracy   Perplexity   Memory(MB)
--------------------------------------------------------------------------------
llama2:7b           245.67       12.34        0.723     15.67        2048.45
mistral:7b          312.89       9.87         0.789     12.34        1876.23
codellama:7b        298.45       15.67        0.634     18.90        2156.78
neural-chat:7b      356.78       8.90         0.823     11.23        1987.45
llama2:13b          445.23       6.78         0.756     9.87         3456.78

✅ Benchmark completed successfully!
   Results saved to: benchmark_results.json
   Total models tested: 5
```

### JSON Results

Results are saved to `benchmark_results.json` with detailed metrics:

```json
{
  "benchmark_config": {
    "models": ["llama2:7b", "mistral:7b", "codellama:7b"],
    "prompt_length": 512,
    "max_new_tokens": 128,
    "temperature": 0.7,
    "num_runs": 3,
    "warmup_runs": 1
  },
  "results": [
    {
      "model_name": "llama2:7b",
      "latency_ms": 245.67,
      "throughput_tokens_per_sec": 12.34,
      "memory_usage_mb": 2048.45,
      "accuracy_score": 0.723,
      "perplexity": 15.67,
      "timestamp": "2024-01-15T10:30:45.123456"
    }
  ],
  "summary": {
    "fastest_model": "llama2:7b",
    "highest_throughput": "codellama:7b",
    "highest_accuracy": "neural-chat:7b",
    "lowest_perplexity": "llama2:13b"
  }
}
```

## Advanced Usage

### Custom Prompts

Modify the test prompts in the `LLMBenchmarker` class:

```python
self.test_prompts = {
    "custom_task": "Your custom prompt here",
    "another_task": "Another custom prompt"
}
```

### Parallel Benchmarking

For faster benchmarking, you can modify the code to run models in parallel:

```python
# In run_benchmarks method
tasks = [self.benchmark_model(model) for model in models_to_test]
results = await asyncio.gather(*tasks)
```

### Memory Monitoring

The system automatically tracks memory usage. For more detailed monitoring:

```python
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## Troubleshooting

### Common Issues

1. **Ollama not running:**

   ```bash
   # Start Ollama
   ollama serve
   ```

2. **Model not found:**

   ```bash
   # Pull the model
   ollama pull model_name
   ```

3. **CUDA out of memory:**

   - Reduce batch size
   - Use smaller models
   - Close other applications

4. **Slow performance:**
   - Check if CUDA is available
   - Ensure sufficient RAM
   - Close background processes

### Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster inference
2. **Model Selection**: Start with smaller models (7B parameters)
3. **Memory Management**: Close unnecessary applications
4. **Network**: Use local models for consistent performance

## Research Applications

This benchmarking system is useful for:

- **Model Comparison**: Compare performance across different architectures
- **Hardware Optimization**: Test performance on different hardware configurations
- **Deployment Planning**: Choose optimal models for production
- **Research Validation**: Verify model performance claims
- **Cost Analysis**: Balance performance vs. resource requirements

## Contributing

To extend the benchmarking system:

1. Add new metrics in `BenchmarkResult`
2. Implement new task types in `test_prompts`
3. Add custom evaluation functions
4. Extend the `OllamaClient` for additional features

## License

This project is open source and available under the MIT License.
