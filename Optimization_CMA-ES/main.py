import subprocess
import sys
import os
import json
import time
from pathlib import Path
import threading
import queue


# ==================== Configuration Section ====================
# Specify the target data folder name here
TARGET_DATA_FOLDER = ""  # Modify this to specify the folder to process

# Individual data optimization phase configuration (lightweight fine-tuning)
INDIVIDUAL_CMA_CONFIG = {
    'ecr': {
        'sigma': 0.01,
        'popsize': 20,
        'maxiter': 100,
        'tolfun': 1e-6,
        'tolx': 1e-6,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    },
    'leakage': {
        'sigma': 0.1,
        'popsize': 20,
        'maxiter': 100,
        'tolfun': 1e-6,
        'tolx': 1e-6,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    },
    'mea': {
        'sigma': 0.01,
        'popsize': 20,
        'maxiter': 100,
        'tolfun': 1e-6,
        'tolx': 1e-6,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    },
    't1t2': {
        'sigma': 0.1,
        'popsize': 20,
        'maxiter': 100,
        'tolfun': 1e-6,
        'tolx': 1e-6,
        'verb_log': 0,
        'verb_disp': 0,
        'seed': 42
    }
}
# ==================== Path Configuration Center ====================
class PathConfig:
    """Configuration class for unified management of all file paths"""

    def __init__(self, target_data_folder=None, current_training_file=None):
        # Get current script directory and project root directory
        self.current_dir = Path(__file__).parent.absolute()
        self.project_root = self.current_dir.parent

        # Batch processing related parameters
        self.target_data_folder = target_data_folder  # e.g., "correlation_matrix_brisbane_layout_1"
        self.current_training_file = current_training_file  # Current processing npy file path

        # ==================== Input File Paths ====================
        # Initial parameter file path
        self.initial_params_file = self.project_root / 'Optimization_CMA-ES' / '' / ''/''

        # Training data file path
        if current_training_file:
            self.training_data_file = current_training_file
        else:
            # Default path (compatible with original logic)
            self.training_data_file = self.project_root / 'Experiment' / 'data' / 'mtx' / '' / ''

        # ==================== Shared File Paths ====================
        # Parameter file shared by all four optimizers
        if target_data_folder and current_training_file:
            # Batch processing mode: in independent folder for each training data
            layout_name = self._extract_layout_name(target_data_folder)
            training_file_name = current_training_file.stem
            shared_params_dir = self.current_dir / 'optimization_results' / layout_name / training_file_name / 'Paramaeter_Loading'
        else:
            # Default mode
            shared_params_dir = self.project_root / 'Optimization_CMA-ES' / 'Paramaeter_Loading'

        self.shared_params_file = shared_params_dir / 'shared_parameters.json'

        # ==================== Output File Paths ====================
        if target_data_folder and current_training_file:
            # Batch processing mode: each training data has independent output directory
            layout_name = self._extract_layout_name(target_data_folder)
            training_file_name = current_training_file.stem
            base_output_dir = self.current_dir / 'optimization_results' / layout_name / training_file_name

            # Log file output directory
            self.log_output_dir = base_output_dir / 'optimization_logs'

            # Generation log files for each optimizer
            self.ecr_log_file = self.log_output_dir / f'ecr_cma_generations.txt'
            self.leakage_log_file = self.log_output_dir / f'leakage_cma_generations.txt'
            self.t1t2_log_file = self.log_output_dir / f't1t2_cma_generations.txt'
            self.mea_log_file = self.log_output_dir / f'mea_cma_generations.txt'

            # Final result backup files
            self.final_results_dir = base_output_dir
            self.final_params_backup = self.final_results_dir / f'optimized_parameters{self._get_timestamp()}.json'
        else:
            # Default mode (compatible with original logic)
            self.log_output_dir = self.current_dir / 'optimization_logs'

            self.ecr_log_file = self.log_output_dir / 'optimization_log_ecr_cma_generations.txt'
            self.leakage_log_file = self.log_output_dir / 'optimization_log_leakage_cma_generations.txt'
            self.t1t2_log_file = self.log_output_dir / 'optimization_log_t1t2_cma_generations.txt'
            self.mea_log_file = self.log_output_dir / 'optimization_log_mea_cma_generations.txt'

            self.final_results_dir = self.current_dir / 'final_results'
            self.final_params_backup = self.final_results_dir / f'optimized_parameters_{self._get_timestamp()}.json'

        # ==================== Optimizer Script Paths ====================
        self.optimizer_scripts = {
            'ecr': self.current_dir / 'optimize_ecr_parallel.py',
            'leakage': self.current_dir / 'optimize_leakage_parallel.py',
            't1t2': self.current_dir / 'optimize_t1t2_parallel.py',
            'mea': self.current_dir / 'optimize_mea_parallel.py'
        }

    def _extract_layout_name(self, folder_name):
        """Extract layout name from folder name
        Example: correlation_matrix_brisbane_layout_1 -> brisbane_layout_1
        """
        if folder_name.startswith('correlation_matrix_'):
            return folder_name[19:]  # Remove 'correlation_matrix_' prefix
        return folder_name

    def _get_timestamp(self):
        """Get current timestamp string"""
        return time.strftime("%Y%m%d_%H%M%S")

    def get_available_data_folders(self):
        """Get all available data folders"""
        mtx_dir = self.project_root / 'Experiment' / 'data' / 'mtx'
        if not mtx_dir.exists():
            return []

        folders = []
        for item in mtx_dir.iterdir():
            if item.is_dir() and item.name.startswith('correlation_matrix_'):
                folders.append(item.name)

        return sorted(folders)

    def get_training_files_in_folder(self, folder_name, selected_optimizers=None):
        """Get training files to process in specified folder (excluding average_correlation_matrix.npy and completed files)"""
        folder_path = self.project_root / 'Experiment' / 'data' / 'mtx' / folder_name
        if not folder_path.exists():
            return []

        all_files = []
        skipped_files = []

        for npy_file in folder_path.glob('*.npy'):
            if npy_file.name != 'average_correlation_matrix.npy':
                # Pass selected optimizers to check function
                if self._is_file_already_optimized(folder_name, npy_file, selected_optimizers):
                    skipped_files.append(npy_file)
                else:
                    all_files.append(npy_file)

        if skipped_files:
            print(f"\n⏭️  Skipping files already optimized with current settings ({len(skipped_files)} files):")
            for skipped_file in skipped_files:
                print(f"    - {skipped_file.name}")

        return sorted(all_files)

    def _is_file_already_optimized(self, folder_name, npy_file, selected_optimizers=None):
        """Check if specified npy file has been optimized with current settings"""
        layout_name = self._extract_layout_name(folder_name)
        training_file_name = npy_file.stem

        print(f"  🔍 Checking file: {training_file_name}")

        # If no selected optimizers specified, check all by default
        if selected_optimizers is None:
            selected_optimizers = ['ecr', 'leakage', 't1t2', 'mea']

        # Check if result directory exists
        result_dir = self.current_dir / 'optimization_results' / layout_name / training_file_name
        if not result_dir.exists():
            print(f"    ❌ Result directory does not exist: {result_dir.name}")
            return False

        # Check if optimization parameter file exists
        optimized_files = list(result_dir.glob('optimized_parameters*.json'))
        if not optimized_files:
            print(f"    ❌ Optimization parameter file does not exist")
            return False

        # Check if shared parameter file exists
        shared_params_file = result_dir / 'Paramaeter_Loading' / 'shared_parameters.json'
        if not shared_params_file.exists():
            print(f"    ❌ Shared parameter file does not exist")
            return False

        # Check if log files for each optimizer completed specified iterations
        log_dir = result_dir / 'optimization_logs'
        if not log_dir.exists():
            print(f"    ❌ Log directory does not exist")
            return False

        # Define optimizer settings - read from individual optimization CMA config
        all_optimizer_settings = {
            'ecr': {
                'max_iter': INDIVIDUAL_CMA_CONFIG['ecr']['maxiter'],
                'pop_size': INDIVIDUAL_CMA_CONFIG['ecr']['popsize']
            },
            'leakage': {
                'max_iter': INDIVIDUAL_CMA_CONFIG['leakage']['maxiter'],
                'pop_size': INDIVIDUAL_CMA_CONFIG['leakage']['popsize']
            },
            't1t2': {
                'max_iter': INDIVIDUAL_CMA_CONFIG['t1t2']['maxiter'],
                'pop_size': INDIVIDUAL_CMA_CONFIG['t1t2']['popsize']
            },
            'mea': {
                'max_iter': INDIVIDUAL_CMA_CONFIG['mea']['maxiter'],
                'pop_size': INDIVIDUAL_CMA_CONFIG['mea']['popsize']
            }
        }

        # Only check selected optimizers
        optimizer_settings = {name: all_optimizer_settings[name]
                              for name in selected_optimizers
                              if name in all_optimizer_settings}

        print(f"    📋 Optimizers to check: {', '.join([name.upper() for name in optimizer_settings.keys()])}")

        # Check log files for each selected optimizer
        incomplete_optimizers = []

        for optimizer_name, settings in optimizer_settings.items():
            log_file = log_dir / f'{optimizer_name}_cma_generations.txt'

            if not log_file.exists():
                print(f"    ❌ {optimizer_name.upper()} optimizer log file does not exist: {log_file.name}")
                return False

            # Check if log file completed specified iterations
            completion_status = self._check_log_completion(log_file, settings['max_iter'])
            if not completion_status:
                incomplete_optimizers.append(optimizer_name.upper())

        # If there are incomplete optimizers, show specific information
        if incomplete_optimizers:
            print(f"    ❌ Incomplete optimization modules: {', '.join(incomplete_optimizers)}")
            return False

        print(f"    ✅ All selected optimizers completed")
        return True

    def _check_log_completion(self, log_file, required_iterations):
        """Check if log file completed specified iterations"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                print(f"      📝 {log_file.name}: File is empty")
                return False

            # Skip header line
            if len(lines) < 2:
                print(f"      📝 {log_file.name}: Only header line, no optimization data")
                return False

            max_generation = 0

            # Parse from second line (skip CSV header)
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                # CSV format: Generation,BestObjective,BestParameters
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        generation_num = int(parts[0])
                        max_generation = max(max_generation, generation_num)
                    except ValueError:
                        continue

            # Check if required iterations were reached
            completion_achieved = max_generation >= required_iterations

            if not completion_achieved:
                print(f"      📊 {log_file.name}: Progress {max_generation}/{required_iterations} generations (incomplete)")

            return completion_achieved

        except Exception as e:
            print(f"      ⚠️ Error checking log file {log_file.name}: {e}")
            return False

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

    def validate_input_files(self):
        """Validate if input files exist"""
        required_files = [
            (self.initial_params_file, "Initial parameter file"),
            (self.training_data_file, "Training data file")
        ]

        missing_files = []
        for file_path, description in required_files:
            if not file_path.exists():
                missing_files.append(f"{description}: {file_path}")

        if missing_files:
            print("❌ The following required files do not exist:")
            for file_info in missing_files:
                print(f"   {file_info}")
            return False

        print("✅ All input files validated successfully")
        return True

    def initialize_shared_params(self):
        """Create shared parameter file from initial parameter file"""
        try:
            # Read initial parameters
            with open(self.initial_params_file, 'r', encoding='utf-8') as f:
                initial_params = json.load(f)

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
            # Initial parameter file path
            'INITIAL_PARAMS_FILE': str(self.initial_params_file),
            # Shared parameter file path
            'SHARED_PARAMS_FILE': str(self.shared_params_file),

            # Training data file path
            'TRAINING_DATA_FILE': str(self.training_data_file),

            # Log file paths for each optimizer
            'ECR_LOG_FILE': str(self.ecr_log_file),
            'LEAKAGE_LOG_FILE': str(self.leakage_log_file),
            'T1T2_LOG_FILE': str(self.t1t2_log_file),
            'MEA_LOG_FILE': str(self.mea_log_file),

            # CMA-ES configuration - ECR
            'CMA_ECR_SIGMA': str(INDIVIDUAL_CMA_CONFIG['ecr']['sigma']),
            'CMA_ECR_POPSIZE': str(INDIVIDUAL_CMA_CONFIG['ecr']['popsize']),
            'CMA_ECR_MAXITER': str(INDIVIDUAL_CMA_CONFIG['ecr']['maxiter']),
            'CMA_ECR_TOLFUN': str(INDIVIDUAL_CMA_CONFIG['ecr']['tolfun']),
            'CMA_ECR_TOLX': str(INDIVIDUAL_CMA_CONFIG['ecr']['tolx']),

            # CMA-ES configuration - LEAKAGE
            'CMA_LEAKAGE_SIGMA': str(INDIVIDUAL_CMA_CONFIG['leakage']['sigma']),
            'CMA_LEAKAGE_POPSIZE': str(INDIVIDUAL_CMA_CONFIG['leakage']['popsize']),
            'CMA_LEAKAGE_MAXITER': str(INDIVIDUAL_CMA_CONFIG['leakage']['maxiter']),
            'CMA_LEAKAGE_TOLFUN': str(INDIVIDUAL_CMA_CONFIG['leakage']['tolfun']),
            'CMA_LEAKAGE_TOLX': str(INDIVIDUAL_CMA_CONFIG['leakage']['tolx']),

            # CMA-ES configuration - T1T2
            'CMA_T1T2_SIGMA': str(INDIVIDUAL_CMA_CONFIG['t1t2']['sigma']),
            'CMA_T1T2_POPSIZE': str(INDIVIDUAL_CMA_CONFIG['t1t2']['popsize']),
            'CMA_T1T2_MAXITER': str(INDIVIDUAL_CMA_CONFIG['t1t2']['maxiter']),
            'CMA_T1T2_TOLFUN': str(INDIVIDUAL_CMA_CONFIG['t1t2']['tolfun']),
            'CMA_T1T2_TOLX': str(INDIVIDUAL_CMA_CONFIG['t1t2']['tolx']),

            # CMA-ES configuration - MEA
            'CMA_MEA_SIGMA': str(INDIVIDUAL_CMA_CONFIG['mea']['sigma']),
            'CMA_MEA_POPSIZE': str(INDIVIDUAL_CMA_CONFIG['mea']['popsize']),
            'CMA_MEA_MAXITER': str(INDIVIDUAL_CMA_CONFIG['mea']['maxiter']),
            'CMA_MEA_TOLFUN': str(INDIVIDUAL_CMA_CONFIG['mea']['tolfun']),
            'CMA_MEA_TOLX': str(INDIVIDUAL_CMA_CONFIG['mea']['tolx']),

            # Project root directory
            'PROJECT_ROOT': str(self.project_root),

            # Python encoding settings
            'PYTHONIOENCODING': 'utf-8'
        }

    def print_configuration(self, selected_optimizers=None):
        """Print current path configuration"""
        print("\n" + "=" * 60)
        print("Path Configuration Information")
        print("=" * 60)
        print(f"Project root directory: {self.project_root}")
        print(f"Current script directory: {self.current_dir}")

        if self.target_data_folder:
            print(f"Target data folder: {self.target_data_folder}")
        if self.current_training_file:
            print(f"Current training file: {self.current_training_file.name}")

        print("\nInput files:")
        print(f"  Initial parameter file: {self.initial_params_file}")
        print(f"  Training data file: {self.training_data_file}")
        print("\nShared files:")
        print(f"  Shared parameter file: {self.shared_params_file}")
        print("\nOutput directories:")
        print(f"  Log output directory: {self.log_output_dir}")
        print(f"  Final result directory: {self.final_results_dir}")

        print(f"\nAverage optimization CMA-ES configuration:")
        for optimizer_name in ['ecr', 'leakage', 't1t2', 'mea']:
            config = INDIVIDUAL_CMA_CONFIG[optimizer_name]
            print(f"  {optimizer_name.upper()}:")
            print(f"    Population size: {config['popsize']}, Max generations: {config['maxiter']}")
            print(f"    Initial step size: {config['sigma']}, Tolerance: {config['tolfun']}")

        if selected_optimizers:
            print(f"\nSelected optimizer scripts:")
            for name in selected_optimizers:
                print(f"  ✅ {name.upper()} optimizer: {self.optimizer_scripts[name]}")

        print("=" * 60)


# ==================== Optimizer Selection Function ====================
def select_optimizers():
    """Select optimizers to run"""
    available_optimizers = {
        'ecr': 'ECR fidelity optimizer',
        'leakage': 'LP/SP leakage parameter optimizer',
        't1t2': 'T1/T2 time constant optimizer',
        'mea': 'Measurement error optimizer'
    }

    print("\n🚀 Select optimizers to run")
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


# ==================== Main Program Class ====================
class ParallelOptimizationManager:
    """Parallel optimization manager"""

    def __init__(self, selected_optimizers, target_data_folder=None):
        self.selected_optimizers = selected_optimizers
        self.target_data_folder = target_data_folder
        self.processes = {}
        self.start_time = None

    def run_batch_mode(self):
        """Batch processing mode"""
        if not self.target_data_folder:
            print("❌ Target data folder not specified")
            return False

        # Get all training files - pass selected optimizers
        temp_config = PathConfig()
        training_files = temp_config.get_training_files_in_folder(self.target_data_folder, self.selected_optimizers)

        if not training_files:
            print(f"❌ No training files found in folder {self.target_data_folder}")
            return False

        print(f"\n🎯 Starting batch processing of {len(training_files)} training files")

        success_count = 0
        for i, training_file in enumerate(training_files, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing file {i}/{len(training_files)}: {training_file.name}")
            print(f"{'=' * 60}")

            # Create configuration for current training file
            config = PathConfig(self.target_data_folder, training_file)

            # Run optimization
            if self._run_optimization_with_config(config):
                success_count += 1
                print(f"✅ File {training_file.name} processing completed")
            else:
                print(f"❌ File {training_file.name} processing failed")

        print(f"\n🎉 Batch processing completed! Success: {success_count}/{len(training_files)}")
        return success_count > 0

    def _run_optimization_with_config(self, config):
        """Run optimization with specified configuration"""
        # Print configuration information
        config.print_configuration(self.selected_optimizers)

        # Create necessary directories
        config.create_directories()

        # Validate input files
        if not config.validate_input_files():
            return False

        # Initialize shared parameter file
        if not config.initialize_shared_params():
            return False

        # Validate selected optimizer scripts exist
        missing_scripts = []
        for optimizer_key in self.selected_optimizers:
            script_path = config.optimizer_scripts.get(optimizer_key)
            if not script_path or not script_path.exists():
                missing_scripts.append(f"{optimizer_key}: {script_path}")

        if missing_scripts:
            print("❌ The following selected optimizer scripts do not exist:")
            for script_info in missing_scripts:
                print(f"   {script_info}")
            return False

        # Start optimizers (using new real-time output method)
        if not self._start_optimizers(config):
            return False

        # Commented out original monitoring method as new _start_optimizers already includes monitoring
        # self._monitor_progress(config)

        # Create final backup
        self._create_final_backup(config)

        return True

    def _start_optimizers(self, config):
        """Start selected optimizers and display output in current console"""
        print("\n" + "=" * 60)
        print("Starting selected optimizers")
        print("=" * 60)

        # Get environment variables
        env_vars = config.get_environment_variables()
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
            script_path = config.optimizer_scripts[optimizer_key]
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
            print(f"❌ Error during monitoring: {e}")

        print("\n" + "=" * 60)
        print("🎉 All optimizer monitoring completed!")
        print("=" * 60)
        return True

    def _monitor_progress(self, config):
        """Monitor optimization progress - simplified version, only shows basic status"""
        print("\n" + "=" * 60)
        print("Optimizer status monitoring")
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
                if len(active_processes) == 0:
                    print(f"\n🎉 All optimizers completed!")
                    break

                # Wait 10 seconds before checking again
                time.sleep(10)

        except KeyboardInterrupt:
            print(f"\n⚠️ Monitoring stopped, but optimizers still running in background")

    def _create_final_backup(self, config):
        """Create final result backup"""
        try:
            if config.shared_params_file.exists():
                # Read final parameters
                with open(config.shared_params_file, 'r', encoding='utf-8') as f:
                    final_params = json.load(f)

                # Save backup
                with open(config.final_params_backup, 'w', encoding='utf-8') as f:
                    json.dump(final_params, f, indent=4, ensure_ascii=False)

                print(f"✅ Final parameter backup saved: {config.final_params_backup}")
                return True
            else:
                print("❌ Shared parameter file does not exist, cannot create backup")
                return False

        except Exception as e:
            print(f"❌ Failed to create final backup: {e}")
            return False


# ==================== Main Program Entry ====================
def main():
    """Main program entry"""
    try:
        # Select optimizers
        selected_optimizers = select_optimizers()
        if not selected_optimizers:
            return

        # Use specified folder from configuration section for batch processing
        manager = ParallelOptimizationManager(selected_optimizers, TARGET_DATA_FOLDER)

        # Check if specified folder exists
        temp_config = PathConfig()
        available_folders = temp_config.get_available_data_folders()

        if TARGET_DATA_FOLDER not in available_folders:
            print(f"❌ Specified folder '{TARGET_DATA_FOLDER}' does not exist")
            print("Available folders:")
            for folder in available_folders:
                print(f"  - {folder}")
            return

        # Show files to be processed - pass selected optimizers
        training_files = temp_config.get_training_files_in_folder(TARGET_DATA_FOLDER, selected_optimizers)

        if training_files:
            print(f"\n🎯 Will process folder: {TARGET_DATA_FOLDER}")
            print(f"Selected optimizers: {', '.join([opt.upper() for opt in selected_optimizers])}")
            print(f"Files to process ({len(training_files)} files):")
            for i, file_path in enumerate(training_files, 1):
                print(f"  {i}. {file_path.name}")
        else:
            print(f"\n🎯 Folder: {TARGET_DATA_FOLDER}")
            print(f"Selected optimizers: {', '.join([opt.upper() for opt in selected_optimizers])}")
            print("✅ All files have been optimized with current settings, no processing needed!")
            return

        # Run batch processing
        manager.run_batch_mode()

    except Exception as e:
        print(f"❌ Program error: {e}")
        sys.exit(1)

    print("\n🎉 Batch optimization processing completed!")


if __name__ == "__main__":
    main()