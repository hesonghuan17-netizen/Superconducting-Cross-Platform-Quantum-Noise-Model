import subprocess
import sys
import os
import json
from pathlib import Path
import threading
import queue
import time


# ==================== Configuration Section ====================
# Specify the target data folder name here
TARGET_DATA_FOLDER = ""  # Modify this to specify the folder to process

# Initial parameter configuration file path
INITIAL_CONFIG_FILE = r""  # Modify this to specify your initial configuration file path

# Average optimization phase configuration (fine-grained comprehensive)
AVERAGE_CMA_CONFIG = {
    'ecr': {
        'sigma': 0.01,
        'popsize': 20,
        'maxiter': 200,
        'tolfun': 1e-8,
        'tolx': 1e-8,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    },
    'leakage': {
        'sigma': 0.1,
        'popsize': 20,
        'maxiter': 200,
        'tolfun': 1e-8,
        'tolx': 1e-8,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    },
    'mea': {
        'sigma': 0.01,
        'popsize': 20,
        'maxiter': 200,
        'tolfun': 1e-8,
        'tolx': 1e-8,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    },
    't1t2': {
        'sigma': 0.1,
        'popsize': 20,
        'maxiter': 200,
        'tolfun': 1e-8,
        'tolx': 1e-8,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    }
}

# ==================== Path Configuration Center ====================
class AveragePathConfig:
    """Path configuration class for Average correlation matrix optimization - same file structure as batch optimization"""

    def __init__(self):
        # Get current script directory and project root directory
        self.current_dir = Path(__file__).parent.absolute()
        self.project_root = self.current_dir.parent

        # Extract layout name
        self.layout_name = self._extract_layout_name(TARGET_DATA_FOLDER)

        # ==================== Input File Paths ====================
        # Your provided initial parameter file
        self.initial_params_file = Path(INITIAL_CONFIG_FILE)

        # Average training data file path
        self.training_data_file = self.project_root / 'Experiment' / 'data' / 'mtx' / TARGET_DATA_FOLDER / 'average_correlation_matrix.npy'

        # ==================== Use same file structure as batch optimization ====================
        # Create results under optimization_results/{layout_name}/average_correlation_matrix/
        training_file_name = "average_correlation_matrix"
        base_output_dir = self.current_dir / 'optimization_results' / self.layout_name / training_file_name

        # Shared parameter file path (same structure as batch optimization)
        self.shared_params_file = base_output_dir / 'Paramaeter_Loading' / 'shared_parameters.json'

        # Log file output directory (same structure as batch optimization)
        self.log_output_dir = base_output_dir / 'optimization_logs'

        # Generation log files for each optimizer (same naming as batch optimization)
        log_prefix = f'optimization_log_{training_file_name}'
        self.ecr_log_file = self.log_output_dir / f'{log_prefix}_ecr_cma_generations.txt'
        self.leakage_log_file = self.log_output_dir / f'{log_prefix}_leakage_cma_generations.txt'
        self.t1t2_log_file = self.log_output_dir / f'{log_prefix}_t1t2_cma_generations.txt'
        self.mea_log_file = self.log_output_dir / f'{log_prefix}_mea_cma_generations.txt'

        # Final result files (same structure as batch optimization)
        self.final_results_dir = base_output_dir
        timestamp = self._get_timestamp()
        self.final_params_backup = self.final_results_dir / f'optimized_parameters_{training_file_name}_{timestamp}.json'

        # ==================== Optimizer Script Paths ====================
        self.optimizer_scripts = {
            'ecr': self.current_dir / 'optimize_ecr_parallel.py',
            'leakage': self.current_dir / 'optimize_leakage_parallel.py',
            't1t2': self.current_dir / 'optimize_t1t2_parallel.py',
            'mea': self.current_dir / 'optimize_mea_parallel.py'
        }

    def _extract_layout_name(self, folder_name):
        """Extract layout name from folder name"""
        if folder_name.startswith('correlation_matrix_'):
            return folder_name[19:]  # Remove 'correlation_matrix_' prefix
        return folder_name

    def _get_timestamp(self):
        """Get current timestamp string"""
        return time.strftime("%Y%m%d_%H%M%S")

    def validate_input_files(self):
        """Validate if input files exist"""
        required_files = [
            (self.initial_params_file, "Initial configuration file"),
            (self.training_data_file, "Average training data file")
        ]

        missing_files = []
        for file_path, description in required_files:
            if not file_path.exists():
                missing_files.append(f"{description}: {file_path}")

        if missing_files:
            print("❌ The following required files do not exist:")
            for file_info in missing_files:
                print(f"   {file_info}")
            print("\n💡 Please ensure:")
            print(f"   1. Initial configuration file exists: {INITIAL_CONFIG_FILE}")
            print(f"   2. Training data file exists: {self.training_data_file}")
            return False

        print("✅ All input files validated successfully")
        return True

    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.shared_params_file.parent,
            self.log_output_dir,
            self.final_results_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Ensure directory exists: {directory}")

    def initialize_shared_params(self):
        """Create shared parameter file from initial configuration file"""
        try:
            # Read your provided initial configuration
            with open(self.initial_params_file, 'r', encoding='utf-8') as f:
                initial_params = json.load(f)

            # Add optimization metadata
            initial_params['_experiment_meta'] = {
                'experiment_name': 'average_optimization',
                'layout_id': self.layout_name,
                'data_variant': 'average_correlation_matrix',
                'created_at': self._get_timestamp(),
                'training_data_path': str(self.training_data_file)
            }

            # Write to shared parameter file
            with open(self.shared_params_file, 'w', encoding='utf-8') as f:
                json.dump(initial_params, f, indent=4, ensure_ascii=False)

            print(f"✅ Shared parameter file created: {self.shared_params_file}")
            return True

        except Exception as e:
            print(f"❌ Failed to create shared parameter file: {e}")
            return False

    def get_environment_variables(self):
        """Get environment variables passed to optimizer scripts"""
        return {
            # Basic paths
            'INITIAL_PARAMS_FILE': str(self.initial_params_file),
            'SHARED_PARAMS_FILE': str(self.shared_params_file),
            'TRAINING_DATA_FILE': str(self.training_data_file),
            'PROJECT_ROOT': str(self.project_root),

            # Experiment information
            'EXPERIMENT_NAME': 'average_optimization',
            'LAYOUT_ID': self.layout_name,
            'DATA_VARIANT': 'average_correlation_matrix',

            # Log file paths
            'ECR_LOG_FILE': str(self.ecr_log_file),
            'LEAKAGE_LOG_FILE': str(self.leakage_log_file),
            'T1T2_LOG_FILE': str(self.t1t2_log_file),
            'MEA_LOG_FILE': str(self.mea_log_file),

            # CMA-ES configuration - ECR
            'CMA_ECR_SIGMA': str(AVERAGE_CMA_CONFIG['ecr']['sigma']),
            'CMA_ECR_POPSIZE': str(AVERAGE_CMA_CONFIG['ecr']['popsize']),
            'CMA_ECR_MAXITER': str(AVERAGE_CMA_CONFIG['ecr']['maxiter']),
            'CMA_ECR_TOLFUN': str(AVERAGE_CMA_CONFIG['ecr']['tolfun']),
            'CMA_ECR_TOLX': str(AVERAGE_CMA_CONFIG['ecr']['tolx']),

            # CMA-ES configuration - LEAKAGE
            'CMA_LEAKAGE_SIGMA': str(AVERAGE_CMA_CONFIG['leakage']['sigma']),
            'CMA_LEAKAGE_POPSIZE': str(AVERAGE_CMA_CONFIG['leakage']['popsize']),
            'CMA_LEAKAGE_MAXITER': str(AVERAGE_CMA_CONFIG['leakage']['maxiter']),
            'CMA_LEAKAGE_TOLFUN': str(AVERAGE_CMA_CONFIG['leakage']['tolfun']),
            'CMA_LEAKAGE_TOLX': str(AVERAGE_CMA_CONFIG['leakage']['tolx']),

            # CMA-ES configuration - T1T2
            'CMA_T1T2_SIGMA': str(AVERAGE_CMA_CONFIG['t1t2']['sigma']),
            'CMA_T1T2_POPSIZE': str(AVERAGE_CMA_CONFIG['t1t2']['popsize']),
            'CMA_T1T2_MAXITER': str(AVERAGE_CMA_CONFIG['t1t2']['maxiter']),
            'CMA_T1T2_TOLFUN': str(AVERAGE_CMA_CONFIG['t1t2']['tolfun']),
            'CMA_T1T2_TOLX': str(AVERAGE_CMA_CONFIG['t1t2']['tolx']),

            # CMA-ES configuration - MEA
            'CMA_MEA_SIGMA': str(AVERAGE_CMA_CONFIG['mea']['sigma']),
            'CMA_MEA_POPSIZE': str(AVERAGE_CMA_CONFIG['mea']['popsize']),
            'CMA_MEA_MAXITER': str(AVERAGE_CMA_CONFIG['mea']['maxiter']),
            'CMA_MEA_TOLFUN': str(AVERAGE_CMA_CONFIG['mea']['tolfun']),
            'CMA_MEA_TOLX': str(AVERAGE_CMA_CONFIG['mea']['tolx']),

            # System configuration
            'PYTHONIOENCODING': 'utf-8'
        }

    def print_configuration(self, selected_optimizers=None):
        """Print current path configuration"""
        print("\n" + "=" * 70)
        print("Average Correlation Matrix Pre-optimization Configuration")
        print("=" * 70)
        print(f"Target data folder: {TARGET_DATA_FOLDER}")
        print(f"Layout name: {self.layout_name}")
        print(f"Project root directory: {self.project_root}")
        print(f"Current script directory: {self.current_dir}")

        print("\nInput files:")
        print(f"  Initial configuration file: {self.initial_params_file}")
        print(f"  Average training data: {self.training_data_file}")

        print("\nShared files:")
        print(f"  Shared parameter file: {self.shared_params_file}")

        print("\nOutput directories:")
        print(f"  Log output directory: {self.log_output_dir}")
        print(f"  Final result directory: {self.final_results_dir}")

        print(f"\nAverage optimization CMA-ES configuration:")
        for optimizer_name in ['ecr', 'leakage', 't1t2', 'mea']:
            config = AVERAGE_CMA_CONFIG[optimizer_name]
            print(f"  {optimizer_name.upper()}:")
            print(f"    Population size: {config['popsize']}, Max generations: {config['maxiter']}")
            print(f"    Initial step size: {config['sigma']}, Tolerance: {config['tolfun']}")


        if selected_optimizers:
            print(f"\nSelected optimizers:")
            for name in selected_optimizers:
                print(f"  ✅ {name.upper()} optimizer: {self.optimizer_scripts[name]}")

        print("=" * 70)


# ==================== Optimizer Selection Function ====================
def select_optimizers():
    """Select optimizers to run"""
    available_optimizers = {
        'ecr': 'ECR fidelity optimizer',
        'leakage': 'LP/SP leakage parameter optimizer',
        't1t2': 'T1/T2 time constant optimizer',
        'mea': 'Measurement error optimizer'
    }

    print("\n🚀 Select Average optimizers to run")
    print("Available options:")
    for key, name in available_optimizers.items():
        print(f"  {key} - {name}")

    print("\nEnter optimizer codes to run, separated by spaces")
    print("Examples: ecr mea  or  ecr leakage  or  all (all)")
    print("Enter exit to quit program")

    while True:
        user_input = input("\nPlease enter: ").strip().lower()

        if user_input == 'exit':
            print("Exiting program")
            return None

        if user_input == 'all':
            selected = list(available_optimizers.keys())
            print(f"✅ Selected all: {', '.join([opt.upper() for opt in selected])}")
            return selected

        if not user_input:
            print("❌ Please enter at least one optimizer")
            continue

        selected = []
        invalid = []

        for opt in user_input.split():
            if opt in available_optimizers:
                if opt not in selected:  # Avoid duplicates
                    selected.append(opt)
            else:
                invalid.append(opt)

        if invalid:
            print(f"❌ Invalid optimizers: {', '.join(invalid)}")
            print(f"Available options: {', '.join(available_optimizers.keys())} or all")
            continue

        if not selected:
            print("❌ Please select at least one valid optimizer")
            continue

        print(f"✅ Selected: {', '.join([opt.upper() for opt in selected])}")
        return selected


# ==================== Average Optimization Manager ====================
class AverageOptimizationManager:
    """Average correlation matrix optimization manager"""

    def __init__(self, selected_optimizers):
        self.selected_optimizers = selected_optimizers
        self.config = AveragePathConfig()
        self.processes = {}
        self.start_time = None

    def run_optimization(self):
        """Run Average optimization"""
        print(f"\n🎯 Starting Average correlation matrix pre-optimization")
        print(f"Target: {TARGET_DATA_FOLDER}/average_correlation_matrix.npy")

        # Print configuration information
        self.config.print_configuration(self.selected_optimizers)

        # Create necessary directories
        self.config.create_directories()

        # Validate input files
        if not self.config.validate_input_files():
            return False

        # Initialize shared parameter file
        if not self.config.initialize_shared_params():
            return False

        # Validate optimizer scripts
        if not self._validate_optimizer_scripts():
            return False

        # Start optimizers
        if not self._start_optimizers():
            return False

        # Monitor progress
        self._monitor_progress()

        # Create final backup
        self._create_final_backup()

        return True

    def _validate_optimizer_scripts(self):
        """Validate if optimizer scripts exist"""
        missing_scripts = []
        for optimizer_key in self.selected_optimizers:
            script_path = self.config.optimizer_scripts.get(optimizer_key)
            if not script_path or not script_path.exists():
                missing_scripts.append(f"{optimizer_key}: {script_path}")

        if missing_scripts:
            print("❌ The following optimizer scripts do not exist:")
            for script_info in missing_scripts:
                print(f"   {script_info}")
            return False

        print(f"✅ Optimizer script validation passed ({len(self.selected_optimizers)} scripts)")
        return True

    def _start_optimizers(self):
        """Start selected optimizers and display output in current console"""
        print("\n" + "=" * 60)
        print("Starting Average optimizers")
        print("=" * 60)

        # Get environment variables
        env_vars = self.config.get_environment_variables()
        env = os.environ.copy()
        env.update(env_vars)

        # Optimizer information
        optimizer_names = {
            'ecr': 'ECR fidelity optimizer',
            'leakage': 'LP/SP leakage parameter optimizer',
            't1t2': 'T1/T2 time constant optimizer',
            'mea': 'Measurement error optimizer'
        }

        self.start_time = time.time()
        self.processes = {}
        output_queue = queue.Queue()

        def run_optimizer(optimizer_key, script_path, env_vars):
            """Run single optimizer in thread"""
            try:
                # Start subprocess
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    env=env,
                    encoding='utf-8',
                    errors='replace'
                )

                self.processes[optimizer_key] = {
                    'process': process,
                    'description': optimizer_names.get(optimizer_key, f"{optimizer_key} optimizer"),
                    'start_time': time.time()
                }

                # Read output in real-time
                for line in process.stdout:
                    if line.strip():  # Only show non-empty lines
                        output_queue.put(f"[{optimizer_key.upper()}] {line.rstrip()}")

                # Wait for process to end
                return_code = process.wait()
                if return_code == 0:
                    output_queue.put(f"[{optimizer_key.upper()}] ✅ Optimization completed")
                else:
                    output_queue.put(f"[{optimizer_key.upper()}] ❌ Optimization failed (return code: {return_code})")

            except Exception as e:
                output_queue.put(f"[{optimizer_key.upper()}] ❌ Launch failed: {e}")

        # Start all optimizer threads
        threads = []
        for optimizer_key in self.selected_optimizers:
            script_path = self.config.optimizer_scripts[optimizer_key]
            description = optimizer_names.get(optimizer_key, f"{optimizer_key} optimizer")

            print(f"Starting {description}...")

            thread = threading.Thread(
                target=run_optimizer,
                args=(optimizer_key, script_path, env_vars),
                daemon=True
            )
            thread.start()
            threads.append((optimizer_key, thread))

            time.sleep(0.5)  # Stagger startup times

        print(f"\n🎯 {len(self.selected_optimizers)} optimizers started")
        print("=" * 60)
        print("Real-time output monitoring:")
        print("=" * 60)

        # Monitor all output
        active_optimizers = set(self.selected_optimizers)

        try:
            while active_optimizers:
                try:
                    # Get output from queue (timeout 1 second)
                    output = output_queue.get(timeout=1)
                    print(output)

                    # Check if any optimizer completed
                    if "optimization completed" in output.lower() or "optimization failed" in output.lower() or "launch failed" in output.lower():
                        # Extract optimizer name from output
                        for opt_key in list(active_optimizers):
                            if opt_key.upper() in output:
                                active_optimizers.discard(opt_key)
                                break

                except queue.Empty:
                    # Check if threads are still alive
                    living_threads = [opt_key for opt_key, thread in threads if thread.is_alive()]
                    if not living_threads:
                        break

                except KeyboardInterrupt:
                    print("\n⚠️ User interrupted monitoring, but optimizers will continue running in background")
                    print("Can check python process status through task manager")
                    break

        except Exception as e:
            print(f"❌ Error during monitoring process: {e}")

        print("\n" + "=" * 60)
        print("🎉 All optimizer monitoring completed!")
        print("=" * 60)
        return True

    def _monitor_progress(self):
        """Monitor optimization progress"""
        print("\n" + "=" * 60)
        print("Monitoring Average optimization progress")
        print("=" * 60)
        print("Press Ctrl+C to stop monitoring (optimizers will continue running)")

        try:
            while True:
                # Check process status
                active_processes = []
                completed_processes = []

                for optimizer_key, process_info in self.processes.items():
                    process = process_info['process']
                    poll_result = process.poll()
                    if poll_result is None:  # Process still running
                        active_processes.append(optimizer_key)
                    else:
                        completed_processes.append((optimizer_key, poll_result))

                # Print status update
                current_time = time.time()
                elapsed_time = current_time - self.start_time

                print(f"\r⏰ Runtime: {elapsed_time / 60:.1f}min | "
                      f"🏃 Active: {len(active_processes)} | "
                      f"✅ Completed: {len(completed_processes)}", end="", flush=True)

                # If all processes completed, exit monitoring
                if len(active_processes) == 0:  # Modified here
                    print(f"\n🎉 All Average optimizers completed!")
                    break

                # Wait 10 seconds before checking again
                time.sleep(10)

        except KeyboardInterrupt:
            print(f"\n⚠️ Monitoring stopped, but optimizers still running in background")

    def _create_final_backup(self):
        """Create final result backup"""
        try:
            if self.config.shared_params_file.exists():
                # Read final parameters
                with open(self.config.shared_params_file, 'r', encoding='utf-8') as f:
                    final_params = json.load(f)

                # Save backup
                with open(self.config.final_params_backup, 'w', encoding='utf-8') as f:
                    json.dump(final_params, f, indent=4, ensure_ascii=False)

                print(f"✅ Average optimization final parameters saved: {self.config.final_params_backup}")

                # Show output file structure
                self._print_output_summary()

                return True
            else:
                print("❌ Shared parameter file does not exist, cannot create backup")
                return False

        except Exception as e:
            print(f"❌ Failed to create final backup: {e}")
            return False

    def _print_output_summary(self):
        """Print output file structure summary"""
        print(f"\n📁 Average optimization output file structure:")
        print(f"optimization_results/")
        print(f"└── {self.config.layout_name}/")
        print(f"    └── average_correlation_matrix/")
        print(f"        ├── Paramaeter_Loading/")
        print(f"        │   └── shared_parameters.json")
        print(f"        ├── optimization_logs/")
        print(f"        │   ├── optimization_log_average_correlation_matrix_ecr_cma_generations.txt")
        print(f"        │   ├── optimization_log_average_correlation_matrix_leakage_cma_generations.txt")
        print(f"        │   ├── optimization_log_average_correlation_matrix_t1t2_cma_generations.txt")
        print(f"        │   └── optimization_log_average_correlation_matrix_mea_cma_generations.txt")
        print(f"        └── optimized_parameters_average_correlation_matrix_*.json")


# ==================== Main Program Entry ====================
def main():
    """Main program entry"""
    try:
        print("🔬 Average Correlation Matrix Pre-optimization Program")
        print(f"Target layout: {TARGET_DATA_FOLDER}")
        print(f"Initial configuration: {INITIAL_CONFIG_FILE}")

        # Select optimizers
        selected_optimizers = select_optimizers()
        if not selected_optimizers:
            return

        # Create and run Average optimization manager
        manager = AverageOptimizationManager(selected_optimizers)

        if manager.run_optimization():
            print("\n🎉 Average correlation matrix pre-optimization completed!")
            print("💡 Tip: Optimized parameters have been saved and can be used as initial parameters for batch optimization")
        else:
            print("\n❌ Average optimization failed")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Program error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()