"""
Raspberry Pi 4 Inference Benchmark for LiteRT Seed Germination Model

This script benchmarks the LiteRT model on Raspberry Pi 4, measuring:
- Warm-up performance
- CPU/Memory usage
- Average inference time
- Throughput (images/second)

Usage:
    python rpi4_benchmark.py --model mobilenetv3_unet_dynamic.tflite --num_warmup 10 --num_runs 100
"""

import argparse
import time
import numpy as np
import psutil
import os
from typing import Dict, List, Tuple
from ai_edge_litert.interpreter import Interpreter
import matplotlib.pyplot as plt
from datetime import datetime

class RPi4Benchmark:
    """
    Benchmark LiteRT model on Raspberry Pi 4
    """
    
    def __init__(
        self, 
        model_path: str,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        num_threads: int = 4,
        verbose: bool = True
    ):
        """
        Initialize benchmark
        
        Args:
            model_path: Path to LiteRT model
            input_shape: Input image shape (H, W, C)
            num_threads: Number of CPU threads to use
            verbose: Print detailed information
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_threads = num_threads
        self.verbose = verbose
        
        # Load model
        if self.verbose:
            print("="*70)
            print("RASPBERRY PI 4 - LiteRT MODEL BENCHMARK")
            print("="*70)
            print(f"\n[1/3] Loading model: {model_path}")
        
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        if self.verbose:
            self._print_model_info()
        
        # Generate dummy input
        self.dummy_input = self._generate_dummy_input()
        
        if self.verbose:
            print(f"\n[2/3] Configuration:")
            print(f"  - Input shape: {self.input_shape}")
            print(f"  - CPU threads: {num_threads}")
            print(f"  - Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    def _print_model_info(self):
        """Print model information"""
        print(f"\n  Model Details:")
        print(f"    Input:")
        print(f"      - Shape: {self.input_details[0]['shape']}")
        print(f"      - Type: {self.input_details[0]['dtype']}")
        print(f"      - Quantization: {self.input_details[0]['quantization']}")
        print(f"    Output:")
        print(f"      - Shape: {self.output_details[0]['shape']}")
        print(f"      - Type: {self.output_details[0]['dtype']}")
        print(f"      - Quantization: {self.output_details[0]['quantization']}")
    
    def _generate_dummy_input(self) -> np.ndarray:
        """
        Generate random input for benchmarking
        
        Returns:
            Random input array with correct shape and dtype
        """
        input_dtype = self.input_details[0]['dtype']
        input_shape = self.input_details[0]['shape']
        
        if input_dtype == np.float32:
            # For dynamic quantization (FP32 input)
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
        elif input_dtype == np.int8:
            # For INT8 quantization
            dummy_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
        elif input_dtype == np.uint8:
            # For UINT8 quantization
            dummy_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype}")
        
        return dummy_input
    
    def _run_single_inference(self) -> float:
        """
        Run single inference and return time
        
        Returns:
            Inference time in seconds
        """
        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], self.dummy_input)
        self.interpreter.invoke()
        _ = self.interpreter.get_tensor(self.output_details[0]['index'])
        inference_time = time.time() - start_time
        
        return inference_time
    
    def warmup(self, num_warmup: int = 10) -> Dict[str, float]:
        """
        Warm-up phase to stabilize performance
        
        Args:
            num_warmup: Number of warm-up iterations
            
        Returns:
            Dictionary with warm-up statistics
        """
        if self.verbose:
            print(f"\n[3/3] Warm-up phase ({num_warmup} iterations)...")
        
        warmup_times = []
        
        for i in range(num_warmup):
            inference_time = self._run_single_inference()
            warmup_times.append(inference_time)
            
            if self.verbose and (i + 1) % 5 == 0:
                print(f"  Warm-up {i+1}/{num_warmup}: {inference_time*1000:.2f} ms")
        
        warmup_stats = {
            'mean': np.mean(warmup_times),
            'std': np.std(warmup_times),
            'min': np.min(warmup_times),
            'max': np.max(warmup_times),
            'times': warmup_times
        }
        
        if self.verbose:
            print(f"\n  Warm-up complete:")
            print(f"    Mean: {warmup_stats['mean']*1000:.2f} ms")
            print(f"    Std:  {warmup_stats['std']*1000:.2f} ms")
            print(f"    Min:  {warmup_stats['min']*1000:.2f} ms")
            print(f"    Max:  {warmup_stats['max']*1000:.2f} ms")
        
        return warmup_stats
    
    def benchmark(
        self, 
        num_runs: int = 100,
        monitor_resources: bool = True
    ) -> Dict:
        """
        Run benchmark with resource monitoring
        
        Args:
            num_runs: Number of benchmark iterations
            monitor_resources: Monitor CPU/Memory usage
            
        Returns:
            Dictionary with benchmark results
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"BENCHMARK ({num_runs} iterations)")
            print(f"{'='*70}")
        
        inference_times = []
        cpu_usage = []
        memory_usage = []
        cpu_temp = []
        
        # Get process for monitoring
        process = psutil.Process(os.getpid())
        
        for i in range(num_runs):
            # Monitor resources before inference
            if monitor_resources:
                cpu_percent = psutil.cpu_percent(interval=None)
                mem_info = process.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)
                
                # Try to get CPU temperature (RPi specific)
                try:
                    with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                        temp = float(f.read()) / 1000.0
                        cpu_temp.append(temp)
                except:
                    pass
                
                cpu_usage.append(cpu_percent)
                memory_usage.append(memory_mb)
            
            # Run inference
            inference_time = self._run_single_inference()
            inference_times.append(inference_time)
            
            # Progress update
            if self.verbose and (i + 1) % 20 == 0:
                avg_time = np.mean(inference_times[-20:])
                print(f"  Progress: {i+1}/{num_runs} | "
                      f"Avg (last 20): {avg_time*1000:.2f} ms | "
                      f"FPS: {1/avg_time:.2f}")
        
        # Calculate statistics
        results = {
            'inference_times': inference_times,
            'mean_time': np.mean(inference_times),
            'std_time': np.std(inference_times),
            'min_time': np.min(inference_times),
            'max_time': np.max(inference_times),
            'median_time': np.median(inference_times),
            'p95_time': np.percentile(inference_times, 95),
            'p99_time': np.percentile(inference_times, 99),
            'throughput': 1.0 / np.mean(inference_times),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'cpu_temp': cpu_temp,
            'mean_cpu': np.mean(cpu_usage) if cpu_usage else None,
            'mean_memory': np.mean(memory_usage) if memory_usage else None,
            'mean_temp': np.mean(cpu_temp) if cpu_temp else None,
            'num_runs': num_runs,
            'num_threads': self.num_threads,
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print benchmark results"""
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}")
        
        print(f"\n--- INFERENCE TIME ---")
        print(f"  Mean:      {results['mean_time']*1000:.2f} ms")
        print(f"  Std:       {results['std_time']*1000:.2f} ms")
        print(f"  Min:       {results['min_time']*1000:.2f} ms")
        print(f"  Max:       {results['max_time']*1000:.2f} ms")
        print(f"  Median:    {results['median_time']*1000:.2f} ms")
        print(f"  95th %ile: {results['p95_time']*1000:.2f} ms")
        print(f"  99th %ile: {results['p99_time']*1000:.2f} ms")
        
        print(f"\n--- THROUGHPUT ---")
        print(f"  Images/sec: {results['throughput']:.2f}")
        print(f"  FPS:        {results['throughput']:.2f}")
        
        if results['mean_cpu'] is not None:
            print(f"\n--- RESOURCE USAGE ---")
            print(f"  CPU Usage:    {results['mean_cpu']:.1f}%")
            print(f"  Memory:       {results['mean_memory']:.1f} MB")
            if results['mean_temp'] is not None:
                print(f"  CPU Temp:     {results['mean_temp']:.1f}°C")
        
        print(f"\n{'='*70}")
    
    def plot_results(
        self, 
        results: Dict, 
        warmup_stats: Dict = None,
        save_prefix: str = "benchmark"
    ):
        """
        Plot benchmark results as individual PDF files at 300 DPI
        
        Args:
            results: Benchmark results dictionary
            warmup_stats: Warm-up statistics (optional)
            save_prefix: Prefix for output PDF files
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for better PDF rendering
        
        inference_times_ms = np.array(results['inference_times']) * 1000
        saved_files = []
        
        # ========================================================================
        # PLOT 1: Inference Times Over Iterations
        # ========================================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if warmup_stats is not None:
            warmup_times_ms = np.array(warmup_stats['times']) * 1000
            ax.plot(range(-len(warmup_times_ms), 0), warmup_times_ms, 
                   'o-', color='orange', alpha=0.6, label='Warm-up', markersize=4)
        
        ax.plot(inference_times_ms, 'o-', color='blue', alpha=0.6, 
               label='Benchmark', markersize=3)
        ax.axhline(results['mean_time']*1000, color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {results["mean_time"]*1000:.2f} ms')
        ax.axhline(results['p95_time']*1000, color='green', linestyle='--', 
                  linewidth=2, label=f'95th percentile: {results["p95_time"]*1000:.2f} ms')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inference time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Inference time per iteration', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        filename = f"{save_prefix}_inference_times.pdf"
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
        
        # ========================================================================
        # PLOT 2: Histogram of Inference Times
        # ========================================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        n, bins, patches = ax.hist(inference_times_ms, bins=50, color='blue', 
                                    alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(results['mean_time']*1000, color='red', linestyle='--', 
                  linewidth=2.5, label=f'Mean: {results["mean_time"]*1000:.2f} ms')
        ax.axvline(results['median_time']*1000, color='green', linestyle='--', 
                  linewidth=2.5, label=f'Median: {results["median_time"]*1000:.2f} ms')
        ax.axvline(results['p95_time']*1000, color='orange', linestyle='--', 
                  linewidth=2.5, label=f'95th: {results["p95_time"]*1000:.2f} ms')
        
        ax.set_xlabel('Inference time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of inference times', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add statistics text box
        stats_text = (
            f'Statistics:\n'
            f'Mean: {results["mean_time"]*1000:.2f} ms\n'
            f'Std: {results["std_time"]*1000:.2f} ms\n'
            f'Min: {results["min_time"]*1000:.2f} ms\n'
            f'Max: {results["max_time"]*1000:.2f} ms\n'
            f'Median: {results["median_time"]*1000:.2f} ms'
        )
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filename = f"{save_prefix}_histogram.pdf"
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
        
        # ========================================================================
        # PLOT 3: CPU Usage (if available)
        # ========================================================================
        if results['cpu_usage']:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(results['cpu_usage'], color='purple', alpha=0.7, linewidth=1.5)
            ax.axhline(results['mean_cpu'], color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {results["mean_cpu"]:.1f}%')
            
            # Add fill between for visual effect
            ax.fill_between(range(len(results['cpu_usage'])), 
                           results['cpu_usage'], alpha=0.3, color='purple')
            
            ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax.set_ylabel('CPU usage (%)', fontsize=12, fontweight='bold')
            ax.set_title('CPU usage during benchmark', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([0, 100])
            
            # Add statistics text box
            cpu_stats_text = (
                f'CPU statistics:\n'
                f'Mean: {results["mean_cpu"]:.1f}%\n'
                f'Min: {np.min(results["cpu_usage"]):.1f}%\n'
                f'Max: {np.max(results["cpu_usage"]):.1f}%\n'
                f'Std: {np.std(results["cpu_usage"]):.1f}%'
            )
            ax.text(0.98, 0.97, cpu_stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
            
            plt.tight_layout()
            filename = f"{save_prefix}_cpu_usage.pdf"
            plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            saved_files.append(filename)
            plt.close(fig)
        
        # ========================================================================
        # PLOT 4: CPU Temperature (if available)
        # ========================================================================
        if results['cpu_temp']:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.plot(results['cpu_temp'], color='red', alpha=0.7, linewidth=1.5)
            ax.axhline(results['mean_temp'], color='darkred', linestyle='--', 
                      linewidth=2, label=f'Mean: {results["mean_temp"]:.1f}°C')
            
            # Add fill between for visual effect
            ax.fill_between(range(len(results['cpu_temp'])), 
                           results['cpu_temp'], alpha=0.3, color='red')
            
            # Add warning zones
            ax.axhspan(70, 85, alpha=0.1, color='orange', label='Warning zone (70-85°C)')
            ax.axhspan(85, 100, alpha=0.1, color='red', label='Critical zone (>85°C)')
            
            ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax.set_title('CPU temperature during benchmark', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics text box
            temp_stats_text = (
                f'Temperature statistics:\n'
                f'Mean: {results["mean_temp"]:.1f}°C\n'
                f'Min: {np.min(results["cpu_temp"]):.1f}°C\n'
                f'Max: {np.max(results["cpu_temp"]):.1f}°C\n'
                f'Std: {np.std(results["cpu_temp"]):.1f}°C'
            )
            ax.text(0.98, 0.97, temp_stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.5))
            
            plt.tight_layout()
            filename = f"{save_prefix}_cpu_temperature.pdf"
            plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
            saved_files.append(filename)
            plt.close(fig)
        
        # ========================================================================
        # PLOT 5: Summary Dashboard (Combined Overview)
        # ========================================================================
        fig = plt.figure(figsize=(12, 12))  # Increased height from 10 to 12
        gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)  # Increased hspace from 0.3 to 0.45
        
        # Subplot 1: Inference time line plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(inference_times_ms, 'o-', color='blue', alpha=0.6, markersize=2)
        ax1.axhline(results['mean_time']*1000, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=10)
        ax1.set_ylabel('Time (ms)', fontsize=10)
        ax1.set_title('Inference time per iteration', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Histogram
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(inference_times_ms, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax2.axvline(results['mean_time']*1000, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Time (ms)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Subplot 3: Statistics table
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        stats_data = [
            ['Metric', 'Value'],
            ['Mean', f'{results["mean_time"]*1000:.2f} ms'],
            ['Std', f'{results["std_time"]*1000:.2f} ms'],
            ['Min', f'{results["min_time"]*1000:.2f} ms'],
            ['Max', f'{results["max_time"]*1000:.2f} ms'],
            ['Median', f'{results["median_time"]*1000:.2f} ms'],
            ['95th %ile', f'{results["p95_time"]*1000:.2f} ms'],
            ['99th %ile', f'{results["p99_time"]*1000:.2f} ms'],
            ['FPS', f'{results["throughput"]:.2f}'],
        ]
        table = ax3.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        ax3.set_title('Performance statistics', fontsize=11, fontweight='bold', pad=20)
        
        # Subplot 4: CPU usage (if available)
        if results['cpu_usage']:
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.plot(results['cpu_usage'], color='purple', alpha=0.7, linewidth=1)
            ax4.axhline(results['mean_cpu'], color='red', linestyle='--', linewidth=2)
            ax4.fill_between(range(len(results['cpu_usage'])), 
                            results['cpu_usage'], alpha=0.3, color='purple')
            ax4.set_xlabel('Iteration', fontsize=10)
            ax4.set_ylabel('CPU usage (%)', fontsize=10)
            ax4.set_title('CPU usage', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 100])
        
        # Subplot 5: CPU temperature (if available)
        if results['cpu_temp']:
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.plot(results['cpu_temp'], color='red', alpha=0.7, linewidth=1)
            ax5.axhline(results['mean_temp'], color='darkred', linestyle='--', linewidth=2)
            ax5.fill_between(range(len(results['cpu_temp'])), 
                            results['cpu_temp'], alpha=0.3, color='red')
            ax5.set_xlabel('Iteration', fontsize=10)
            ax5.set_ylabel('Temperature (°C)', fontsize=10)
            ax5.set_title('CPU temperature', fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Benchmark summary dashboard', fontsize=14, fontweight='bold', y=0.995)
        filename = f"{save_prefix}_summary_dashboard.pdf"
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
        saved_files.append(filename)
        plt.close(fig)
        
        # ========================================================================
        # Print summary
        # ========================================================================
        print(f"\n{'='*70}")
        print(" PDF PLOTS GENERATED")
        print(f"{'='*70}")
        for i, file in enumerate(saved_files, 1):
            print(f"  {i}. {file}")
        print(f"{'='*70}\n")
    
    def save_results(self, results: Dict, save_path: str = "benchmark_results.txt"):
        """
        Save benchmark results to text file
        
        Args:
            results: Benchmark results dictionary
            save_path: Path to save results
        """
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RASPBERRY PI 4 - LITERT BENCHMARK RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Number of runs: {results['num_runs']}\n")
            f.write(f"CPU threads: {results['num_threads']}\n\n")
            
            f.write("--- INFERENCE TIME ---\n")
            f.write(f"Mean:      {results['mean_time']*1000:.2f} ms\n")
            f.write(f"Std:       {results['std_time']*1000:.2f} ms\n")
            f.write(f"Min:       {results['min_time']*1000:.2f} ms\n")
            f.write(f"Max:       {results['max_time']*1000:.2f} ms\n")
            f.write(f"Median:    {results['median_time']*1000:.2f} ms\n")
            f.write(f"95th %ile: {results['p95_time']*1000:.2f} ms\n")
            f.write(f"99th %ile: {results['p99_time']*1000:.2f} ms\n\n")
            
            f.write("--- THROUGHPUT ---\n")
            f.write(f"Images/sec: {results['throughput']:.2f}\n")
            f.write(f"FPS:        {results['throughput']:.2f}\n\n")
            
            if results['mean_cpu'] is not None:
                f.write("--- RESOURCE USAGE ---\n")
                f.write(f"CPU Usage:    {results['mean_cpu']:.1f}%\n")
                f.write(f"Memory:       {results['mean_memory']:.1f} MB\n")
                if results['mean_temp'] is not None:
                    f.write(f"CPU Temp:     {results['mean_temp']:.1f}°C\n")
        
        print(f" Results saved to: {save_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Benchmark LiteRT model on Raspberry Pi 4'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to LiteRT model file'
    )
    parser.add_argument(
        '--num_warmup', 
        type=int, 
        default=10,
        help='Number of warm-up iterations (default: 10)'
    )
    parser.add_argument(
        '--num_runs', 
        type=int, 
        default=100,
        help='Number of benchmark iterations (default: 100)'
    )
    parser.add_argument(
        '--num_threads', 
        type=int, 
        default=4,
        help='Number of CPU threads (default: 4 for RPi4)'
    )
    parser.add_argument(
        '--input_height', 
        type=int, 
        default=256,
        help='Input image height (default: 256)'
    )
    parser.add_argument(
        '--input_width', 
        type=int, 
        default=256,
        help='Input image width (default: 256)'
    )
    parser.add_argument(
        '--input_channels', 
        type=int, 
        default=3,
        help='Input image channels (default: 3)'
    )
    parser.add_argument(
        '--no_plot', 
        action='store_true',
        help='Disable plotting'
    )
    parser.add_argument(
        '--output_prefix', 
        type=str, 
        default='benchmark',
        help='Prefix for output files (default: benchmark)'
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    input_shape = (args.input_height, args.input_width, args.input_channels)
    benchmark = RPi4Benchmark(
        model_path=args.model,
        input_shape=input_shape,
        num_threads=args.num_threads,
        verbose=True
    )
    
    # Run warm-up
    warmup_stats = benchmark.warmup(num_warmup=args.num_warmup)
    
    # Run benchmark
    results = benchmark.benchmark(
        num_runs=args.num_runs,
        monitor_resources=True
    )
    
    # Save results
    results_file = f"{args.output_prefix}_results.txt"
    benchmark.save_results(results, save_path=results_file)
    
    # Plot results
    if not args.no_plot:
        benchmark.plot_results(
            results, 
            warmup_stats, 
            save_prefix=args.output_prefix  # Changed from save_path
        )
    
    print("\n Benchmark complete!")


if __name__ == "__main__":
    main()