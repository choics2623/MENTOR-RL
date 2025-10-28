#!/usr/bin/env python3
"""
Automated evaluation script for RAG benchmarks
"""

import os
import sys
import time
import signal
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# Configuration
# Add your models here in the format:
# MODELS = {
#     "model-name": "/path/to/your/model",
# }
MODELS = {
    # Example model configurations (commented out)
    # "qwen3-8B-base-30step": "/path/to/your/model",
}

BENCHMARKS = {
    "bamboogle": "test",
    # "2wikimultihopqa": "dev", 
    # "hotpotqa": "dev",
}

SGLANG_PORT = 8002
RETRIEVER_URL = "http://0.0.0.0:7777"
SANDBOX_URL = "http://0.0.0.0:2623"
DATA_DIR = "./data/eval_dataset"  # Update this path to your evaluation dataset directory
SAVE_DIR = "./saved_eval"
SCRIPTS_DIR = "./"

# Server wait time after starting (in seconds)
SERVER_WAIT_TIME = 60

class SGLangServerManager:
    def __init__(self):
        self.server_process = None
        
    def start_server(self, model_path, model_name):
        """Start SGLang server with specified model"""
        logging.info(f"Starting SGLang server with model: {model_name}")

        # Kill any existing server
        self.stop_server()

        # Determine TP and context-length based on model series
        if "qwen2.5" in model_name.lower() or "Qwen2.5" in model_name:
            tp = "4"
            context_length = "32768"
        elif "qwen3" in model_name.lower() or "Qwen3" in model_name:
            tp = "8"
            context_length = "40960"
        else:
            # Default fallback
            tp = "4"
            context_length = "32768"

        logging.info(f"Model series config: TP={tp}, Context Length={context_length}")

        # Start new server
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--served-model-name", "Qwen3",
            "--model-path", model_path,
            "--tp", tp,
            "--context-length", context_length,
            "--enable-metrics",
            "--dtype", "bfloat16",
            "--mem-fraction-static", "0.85",
            "--host", "0.0.0.0",
            "--port", str(SGLANG_PORT),
            "--trust-remote-code",
            "--disable-overlap",
            "--disable-radix-cache"
        ]
        
        logging.info(f"Command: {' '.join(cmd)}")

        # Create log file for this server
        log_file = f"/tmp/sglang_server_{model_name}.log"
        with open(log_file, 'w') as f:
            f.write(f"Starting server at {datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")

        # Start server in background, redirecting output to log file
        # Use only GPUs 4,5,6,7
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

        self.server_process = subprocess.Popen(
            cmd,
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            env=env
        )

        logging.info(f"Server started with PID: {self.server_process.pid}")
        logging.info(f"Server logs: {log_file}")
        logging.info(f"Waiting {SERVER_WAIT_TIME} seconds for server to be ready...")

        # Wait and check periodically
        for i in range(SERVER_WAIT_TIME):
            time.sleep(1)
            if self.server_process.poll() is not None:
                logging.error(f"Server process died after {i} seconds!")
                logging.error(f"Check logs at: {log_file}")
                with open(log_file, 'r') as f:
                    logging.error(f"Last 50 lines of log:\n{f.read()[-5000:]}")
                return False

        logging.info("Server should be ready now")
        return True
        
    def stop_server(self):
        """Stop the current SGLang server"""
        if self.server_process:
            logging.info(f"Stopping server process {self.server_process.pid}")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.warning("Server didn't terminate gracefully, killing...")
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None
        
        # Also kill any orphaned sglang processes
        try:
            subprocess.run(["pkill", "-f", "sglang.launch_server"], check=False)
            time.sleep(5)  # Give time for cleanup
        except Exception as e:
            logging.warning(f"Failed to kill sglang processes: {e}")

def run_evaluation(model_name, model_path, benchmark, split):
    """Run evaluation for a specific model and benchmark"""
    
    save_note = f"{model_name}_{benchmark}_{split}"
    
    cmd = [
        "conda", "run", "-n", "mentor",
        "python", "run_eval.py",
        "--config_path", "eval_config.yaml",
        "--method_name", "mentor-qwen3",
        "--data_dir", DATA_DIR,
        "--dataset_name", benchmark,
        "--split", split,
        "--save_dir", SAVE_DIR,
        "--save_note", save_note,
        "--sgl_remote_url", f"http://0.0.0.0:{SGLANG_PORT}",
        "--remote_retriever_url", RETRIEVER_URL,
        "--sandbox_url", SANDBOX_URL,
        "--generator_model", model_path
    ]
    
    logging.info(f"Running evaluation: {model_name} on {benchmark} ({split})")
    logging.info(f"Command: {' '.join(cmd)}")
    
    # Change to scripts directory
    original_cwd = os.getcwd()
    os.chdir(SCRIPTS_DIR)
    
    try:
        # Run evaluation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per evaluation
        )
        
        if result.returncode == 0:
            logging.info(f"✅ Successfully evaluated {model_name} on {benchmark}")
            return True
        else:
            logging.error(f"❌ Evaluation failed for {model_name} on {benchmark}")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"❌ Evaluation timed out for {model_name} on {benchmark}")
        return False
    except Exception as e:
        logging.error(f"❌ Exception during evaluation: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    logging.info("=" * 60)
    logging.info("Starting automated evaluation")
    logging.info(f"Models: {list(MODELS.keys())}")
    logging.info(f"Benchmarks: {list(BENCHMARKS.keys())}")
    logging.info("=" * 60)

    # Create necessary directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SCRIPTS_DIR):
        logging.warning(f"Scripts directory does not exist: {SCRIPTS_DIR}")
        logging.info(f"Creating scripts directory: {SCRIPTS_DIR}")
        os.makedirs(SCRIPTS_DIR, exist_ok=True)
    
    # Initialize server manager
    server_manager = SGLangServerManager()
    
    total_evaluations = len(MODELS) * len(BENCHMARKS)
    completed_evaluations = 0
    failed_evaluations = []
    
    try:
        # Iterate through each model
        for model_name, model_path in MODELS.items():
            logging.info(f"\n🔄 Starting evaluations for model: {model_name}")
            
            # Start server for this model
            if not server_manager.start_server(model_path, model_name):
                logging.error(f"Failed to start server for {model_name}, skipping...")
                failed_evaluations.extend([(model_name, benchmark) for benchmark in BENCHMARKS.keys()])
                continue
            
            # Run evaluations for all benchmarks with this model
            for benchmark, split in BENCHMARKS.items():
                success = run_evaluation(model_name, model_path, benchmark, split)
                
                completed_evaluations += 1
                
                if not success:
                    failed_evaluations.append((model_name, benchmark))
                
                logging.info(f"Progress: {completed_evaluations}/{total_evaluations} evaluations completed")
            
            logging.info(f"✅ Completed all evaluations for {model_name}")
            
        # Final summary
        logging.info("\n" + "=" * 60)
        logging.info("EVALUATION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total evaluations: {total_evaluations}")
        logging.info(f"Successful: {total_evaluations - len(failed_evaluations)}")
        logging.info(f"Failed: {len(failed_evaluations)}")
        
        if failed_evaluations:
            logging.info("\nFailed evaluations:")
            for model, benchmark in failed_evaluations:
                logging.info(f"  - {model} on {benchmark}")
        else:
            logging.info("\n🎉 All evaluations completed successfully!")
            
    except KeyboardInterrupt:
        logging.info("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        logging.error(f"\n❌ Unexpected error: {e}")
    finally:
        # Always clean up the server
        logging.info("\n🔧 Cleaning up...")
        server_manager.stop_server()
        logging.info("✅ Cleanup completed")

if __name__ == "__main__":
    main()