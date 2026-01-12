import numpy as np
import cma
import sys
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Any


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import sim.sim_Repetition_X
import sim.sim_Repetition_Z


# -----------------------------------------------------------------------------
# Device configuration (sanitized)
# NOTE: Replace "TargetDevice" and file paths with your own dataset layout.
# -----------------------------------------------------------------------------
REAL_DEVICES = {
    "target_device": {
        "name": "TargetDevice",
        "data_path_X": "./data/results_single_repx/results_extract_RepX/averaged/avg_counts_n{}.json",
        "data_path_Z": "./data/results_single_repz/results_extract_RepZ/averaged/avg_counts_n{}.json",
        "output_json": "optimized_parameters_target.json",
    }
}

# -----------------------------------------------------------------------------
# Template parameter file (sanitized)
# -----------------------------------------------------------------------------
TEMPLATE_JSON_FILE = "./data/template_parameters.json"


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
TIMES = 10
SHOTS_EXP = 1
SHOTS = 4096 * TIMES
NUM_QUBITS_LIST = [5, 9, 13, 17, 21]


# -----------------------------------------------------------------------------
# Parameter bounds
# -----------------------------------------------------------------------------
SPAM_LOWER_BOUND = 1e-3
SPAM_UPPER_BOUND = 1e-1

ECR_LOWER_BOUND = 0.9
ECR_UPPER_BOUND = 0.999

SQG_LOWER_BOUND = 0.9
SQG_UPPER_BOUND = 0.9999

T1_LOWER_BOUND = 1e-6
T1_UPPER_BOUND = 1e-4

RATIO_LOWER_BOUND = 0.1   # Minimum T2/T1 ratio
RATIO_UPPER_BOUND = 1.0   # Maximum T2/T1 ratio


# -----------------------------------------------------------------------------
# Load template parameters
# -----------------------------------------------------------------------------
with open(TEMPLATE_JSON_FILE, "r") as json_file:
    template_parameters = json.load(json_file)

spam_rates = template_parameters.get("spam_rates", [])
spam_rates_initial = template_parameters.get("spam_rates_initial", [])
lp = template_parameters.get("lp", [])
sp = template_parameters.get("sp", [])
ecr_fid = template_parameters.get("ecr_fid", [])
sqg_fid = template_parameters.get("sqg_fid", [])
t1_t2_values = [(d["t1"], d["t2"]) for d in template_parameters.get("t1_t2_values", [])]
ecr_lengths = template_parameters.get("ecr_lengths", [])
sqg_lengths_list = template_parameters.get("sqg_length", [])
sqg_lengths = sqg_lengths_list[0] if sqg_lengths_list else 0
rd_lengths = template_parameters.get("rd_length", [])

print("Template parameters loaded:")
print(f"  spam_rates:         {len(spam_rates)} values")
print(f"  spam_rates_initial: {len(spam_rates_initial)} values")
print(f"  ecr_fid:            {len(ecr_fid)} values")
print(f"  sqg_fid:            {len(sqg_fid)} values")
print(f"  t1_t2_values:       {len(t1_t2_values)} pairs")


def load_experimental_data(file_path: str) -> Dict[str, Any]:
    """Load experimental data from a JSON file. Returns an empty dict if missing."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: invalid JSON in {file_path}: {e}")
        return {}


def load_device_experimental_data(device_key: str) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """Load X- and Z-basis experimental datasets for the given device key."""
    device_cfg = REAL_DEVICES[device_key]
    print(f"Loading experimental data for {device_cfg['name']}...")

    exp_data_X: Dict[int, Any] = {}
    exp_data_Z: Dict[int, Any] = {}

    for n in NUM_QUBITS_LIST:
        x_path = device_cfg["data_path_X"].format(n)
        z_path = device_cfg["data_path_Z"].format(n)

        exp_data_X[n] = load_experimental_data(x_path)
        exp_data_Z[n] = load_experimental_data(z_path)

        print(
            f"  n={n:2d}: X basis ({len(exp_data_X[n])} states), "
            f"Z basis ({len(exp_data_Z[n])} states)"
        )

    return exp_data_X, exp_data_Z


def analyze_results_for_(results_array: np.ndarray) -> Dict[str, int]:
    """Convert raw simulation samples into a bitstring->count dictionary."""
    counts: Dict[str, int] = {}
    for row in results_array:
        bitstring = "".join(str(int(b)) for b in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def calculate_TVD_distance(
    sim_counts: Dict[str, int],
    exp_counts: Dict[str, int],
    total_shots_sim: int,
    total_shots_exp: int,
) -> float:
    """
    Compute TVD distance between two outcome distributions (equivalently 2*TVD if TVD is defined with 1/2 factor).
    Here we use the raw TVD distance: sum_x |p_sim(x) - p_exp(x)|.
    """
    all_states = set(sim_counts.keys()) | set(exp_counts.keys())
    TVD = 0.0
    for state in all_states:
        p_sim = sim_counts.get(state, 0) / total_shots_sim
        p_exp = exp_counts.get(state, 0) / total_shots_exp
        TVD += abs(p_sim - p_exp)
    return TVD


def transform_normalized_to_physical(normalized_params: np.ndarray) -> Dict[str, Any]:
    """Map normalized parameters in [0,1] to physical parameter values."""
    param_idx = 0

    # spam_rates: log mapping from [0,1] to [SPAM_LOWER_BOUND, SPAM_UPPER_BOUND]
    spam_count = len(spam_rates)
    spam_norm = normalized_params[param_idx : param_idx + spam_count]
    log_spam = spam_norm * (np.log10(SPAM_UPPER_BOUND) - np.log10(SPAM_LOWER_BOUND)) + np.log10(SPAM_LOWER_BOUND)
    physical_spam_rates = (10 ** log_spam).tolist()
    param_idx += spam_count

    # spam_rates_initial: log mapping
    spam_init_count = len(spam_rates_initial)
    spam_init_norm = normalized_params[param_idx : param_idx + spam_init_count]
    log_spam_init = spam_init_norm * (np.log10(SPAM_UPPER_BOUND) - np.log10(SPAM_LOWER_BOUND)) + np.log10(
        SPAM_LOWER_BOUND
    )
    physical_spam_rates_initial = (10 ** log_spam_init).tolist()
    param_idx += spam_init_count

    # ecr_fid: linear mapping
    ecr_count = len(ecr_fid)
    ecr_norm = normalized_params[param_idx : param_idx + ecr_count]
    physical_ecr_fid = (ecr_norm * (ECR_UPPER_BOUND - ECR_LOWER_BOUND) + ECR_LOWER_BOUND).tolist()
    param_idx += ecr_count

    # sqg_fid: linear mapping
    sqg_count = len(sqg_fid)
    sqg_norm = normalized_params[param_idx : param_idx + sqg_count]
    physical_sqg_fid = (sqg_norm * (SQG_UPPER_BOUND - SQG_LOWER_BOUND) + SQG_LOWER_BOUND).tolist()
    param_idx += sqg_count

    # T1: log mapping
    t1_count = len(t1_t2_values)
    t1_norm = normalized_params[param_idx : param_idx + t1_count]
    log_t1 = t1_norm * (np.log10(T1_UPPER_BOUND) - np.log10(T1_LOWER_BOUND)) + np.log10(T1_LOWER_BOUND)
    physical_t1 = 10 ** log_t1
    param_idx += t1_count

    # T2/T1 ratio: log mapping
    ratio_norm = normalized_params[param_idx : param_idx + t1_count]
    log_ratio = ratio_norm * (np.log10(RATIO_UPPER_BOUND) - np.log10(RATIO_LOWER_BOUND)) + np.log10(RATIO_LOWER_BOUND)
    physical_ratio = 10 ** log_ratio
    physical_t2 = physical_t1 * physical_ratio

    physical_t1_t2_values = [(float(physical_t1[i]), float(physical_t2[i])) for i in range(t1_count)]

    return {
        "spam_rates": physical_spam_rates,
        "spam_rates_initial": physical_spam_rates_initial,
        "ecr_fid": physical_ecr_fid,
        "sqg_fid": physical_sqg_fid,
        "t1_t2_values": physical_t1_t2_values,
    }


def transform_physical_to_normalized(params_dict: Dict[str, Any]) -> np.ndarray:
    """Map physical parameters to normalized parameters in [0,1]."""
    normalized_params = []

    # spam_rates: log mapping
    for val in params_dict["spam_rates"]:
        val = float(np.clip(val, SPAM_LOWER_BOUND, SPAM_UPPER_BOUND))
        log_val = np.log10(val)
        norm = (log_val - np.log10(SPAM_LOWER_BOUND)) / (np.log10(SPAM_UPPER_BOUND) - np.log10(SPAM_LOWER_BOUND))
        normalized_params.append(norm)

    # spam_rates_initial: log mapping
    for val in params_dict["spam_rates_initial"]:
        val = float(np.clip(val, SPAM_LOWER_BOUND, SPAM_UPPER_BOUND))
        log_val = np.log10(val)
        norm = (log_val - np.log10(SPAM_LOWER_BOUND)) / (np.log10(SPAM_UPPER_BOUND) - np.log10(SPAM_LOWER_BOUND))
        normalized_params.append(norm)

    # ecr_fid: linear mapping
    for val in params_dict["ecr_fid"]:
        val = float(np.clip(val, ECR_LOWER_BOUND, ECR_UPPER_BOUND))
        norm = (val - ECR_LOWER_BOUND) / (ECR_UPPER_BOUND - ECR_LOWER_BOUND)
        normalized_params.append(norm)

    # sqg_fid: linear mapping
    for val in params_dict["sqg_fid"]:
        val = float(np.clip(val, SQG_LOWER_BOUND, SQG_UPPER_BOUND))
        norm = (val - SQG_LOWER_BOUND) / (SQG_UPPER_BOUND - SQG_LOWER_BOUND)
        normalized_params.append(norm)

    # T1: log mapping
    for t1, _t2 in params_dict["t1_t2_values"]:
        t1 = float(np.clip(t1, T1_LOWER_BOUND, T1_UPPER_BOUND))
        log_t1 = np.log10(t1)
        norm_t1 = (log_t1 - np.log10(T1_LOWER_BOUND)) / (np.log10(T1_UPPER_BOUND) - np.log10(T1_LOWER_BOUND))
        normalized_params.append(norm_t1)

    # T2/T1 ratio: log mapping
    for t1, t2 in params_dict["t1_t2_values"]:
        t1 = float(np.clip(t1, T1_LOWER_BOUND, T1_UPPER_BOUND))
        ratio = float(np.clip(t2 / t1, RATIO_LOWER_BOUND, RATIO_UPPER_BOUND))
        log_ratio = np.log10(ratio)
        norm_ratio = (log_ratio - np.log10(RATIO_LOWER_BOUND)) / (np.log10(RATIO_UPPER_BOUND) - np.log10(RATIO_LOWER_BOUND))
        normalized_params.append(norm_ratio)

    return np.array(normalized_params, dtype=float)


def create_objective_function(device_key: str, exp_data_X: Dict[int, Any], exp_data_Z: Dict[int, Any]):
    """Create the CMA-ES objective function for a given device and datasets."""
    device_name = REAL_DEVICES[device_key]["name"]
    eval_counter = {"count": 0}
    all_evaluations = []

    def objective_function(normalized_params: np.ndarray) -> float:
        """Objective: sum of averaged TVD distances across all N and both X/Z bases."""
        eval_counter["count"] += 1
        eval_id = eval_counter["count"]

        physical_params = transform_normalized_to_physical(normalized_params)

        total_TVD = 0.0
        TVD_by_size = {}

        for num_qubits in NUM_QUBITS_LIST:
            TVD_X_avg = 0.0
            TVD_Z_avg = 0.0

            # X basis
            exp_states_X = exp_data_X.get(num_qubits, {})
            if exp_states_X:
                try:
                    sim_X = sim.sim_Repetition_X.main(
                        num_qubits=num_qubits,
                        spam_rates=physical_params["spam_rates"],
                        spam_rates_initial=physical_params["spam_rates_initial"],
                        lp=lp,
                        sp=sp,
                        ecr_fid=physical_params["ecr_fid"],
                        sqg_fid=physical_params["sqg_fid"],
                        t1_t2_values=physical_params["t1_t2_values"],
                        ecr_lengths=ecr_lengths,
                        sqg_length=sqg_lengths,
                        rd_length=rd_lengths,
                        shots=SHOTS,
                        shots_exp=SHOTS_EXP,
                    )
                    sim_counts_X = analyze_results_for_TVD(sim_X)

                    TVD_X = 0.0
                    for _state_key, exp_counts in exp_states_X.items():
                        TVD_X += calculate_TVD_distance(
                            sim_counts_X,
                            exp_counts,
                            total_shots_sim=SHOTS,
                            total_shots_exp=sum(exp_counts.values()),
                        )
                    TVD_X_avg = TVD_X / len(exp_states_X)
                    total_TVD += TVD_X_avg
                except Exception as e:
                    print(f"[{device_name}] X-basis simulation error at n={num_qubits}: {e}")
                    total_TVD += 999.0
                    TVD_X_avg = 999.0

            # Z basis
            exp_states_Z = exp_data_Z.get(num_qubits, {})
            if exp_states_Z:
                try:
                    sim_Z = sim.sim_Repetition_Z.main(
                        num_qubits=num_qubits,
                        spam_rates=physical_params["spam_rates"],
                        spam_rates_initial=physical_params["spam_rates_initial"],
                        lp=lp,
                        sp=sp,
                        ecr_fid=physical_params["ecr_fid"],
                        sqg_fid=physical_params["sqg_fid"],
                        t1_t2_values=physical_params["t1_t2_values"],
                        ecr_lengths=ecr_lengths,
                        sqg_length=sqg_lengths,
                        rd_length=rd_lengths,
                        shots=SHOTS,
                        shots_exp=SHOTS_EXP,
                    )
                    sim_counts_Z = analyze_results_for_TVD(sim_Z)

                    TVD_Z = 0.0
                    for _state_key, exp_counts in exp_states_Z.items():
                        TVD_Z += calculate_TVD_distance(
                            sim_counts_Z,
                            exp_counts,
                            total_shots_sim=SHOTS,
                            total_shots_exp=sum(exp_counts.values()),
                        )
                    TVD_Z_avg = TVD_Z / len(exp_states_Z)
                    total_TVD += TVD_Z_avg
                except Exception as e:
                    print(f"[{device_name}] Z-basis simulation error at n={num_qubits}: {e}")
                    total_TVD += 999.0
                    TVD_Z_avg = 999.0

            TVD_by_size[num_qubits] = {"X": TVD_X_avg, "Z": TVD_Z_avg}

        all_evaluations.append(
            {
                "eval_id": eval_id,
                "total_TVD": float(total_TVD),
                "TVD_by_size": TVD_by_size,
                "parameters": physical_params,
            }
        )

        if eval_id % 10 == 0:
            print(f"[{device_name}] Evaluation {eval_id:4d}: TVD = {total_TVD:.6f}")

        return float(total_TVD)

    return objective_function, eval_counter, all_evaluations


def optimize_single_device(device_key: str, output_dir: str) -> Dict[str, Any]:
    """Optimize parameters for one device using CMA-ES."""
    device_cfg = REAL_DEVICES[device_key]
    device_name = device_cfg["name"]

    print("\n" + "=" * 60)
    print(f"Starting optimization for: {device_name}")
    print("=" * 60)

    exp_data_X, exp_data_Z = load_device_experimental_data(device_key)

    objective_func, eval_counter, all_evaluations = create_objective_function(device_key, exp_data_X, exp_data_Z)

    initial_params = {
        "spam_rates": spam_rates,
        "spam_rates_initial": spam_rates_initial,
        "ecr_fid": ecr_fid,
        "sqg_fid": sqg_fid,
        "t1_t2_values": t1_t2_values,
    }
    x0 = transform_physical_to_normalized(initial_params)

    print(f"[{device_name}] Parameter dimension: {len(x0)}")
    print(f"[{device_name}] Running CMA-ES...")

    sigma0 = 0.2
    opts = {
        "bounds": [0, 1],
        "tolfun": 1e-6,
        "maxiter": 100,
        "verb_disp": 0,
        "verb_log": 0,
        "verbose": -9,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generation_log = os.path.join(output_dir, f"optimization_{device_key}_{timestamp}_generations.txt")
    evaluations_jsonl = os.path.join(output_dir, f"optimization_{device_key}_{timestamp}_all_evaluations.jsonl")

    generation_counter = 0

    while not es.stop():
        generation_counter += 1
        solutions = es.ask()
        fitness_values = [objective_func(x) for x in solutions]
        es.tell(solutions, fitness_values)

        current_best = es.best.f
        with open(generation_log, "a") as f:
            f.write(
                f"Generation {generation_counter}: Best TVD = {current_best:.6f}, "
                f"Evaluations = {eval_counter['count']}\n"
            )

        # Optionally dump evaluation records as JSONL for reproducibility
        with open(evaluations_jsonl, "a") as f:
            for rec in all_evaluations[-len(solutions):]:
                f.write(json.dumps(rec) + "\n")

        if generation_counter % 5 == 0:
            print(f"[{device_name}] Generation {generation_counter:3d}: best TVD = {current_best:.6f} "
                  f"(evals={eval_counter['count']})")

    final_solution = es.best.x
    final_fitness = es.best.f
    final_physical = transform_normalized_to_physical(final_solution)

    print("\n" + "=" * 60)
    print(f"[{device_name}] Optimization completed")
    print("=" * 60)
    print(f"[{device_name}] Best objective (TVD): {final_fitness:.6f}")
    print(f"[{device_name}] Generations: {generation_counter}")
    print(f"[{device_name}] Function evaluations: {eval_counter['count']}")

    optimized_parameters = template_parameters.copy()
    optimized_parameters["spam_rates"] = final_physical["spam_rates"]
    optimized_parameters["spam_rates_initial"] = final_physical["spam_rates_initial"]
    optimized_parameters["ecr_fid"] = final_physical["ecr_fid"]
    optimized_parameters["sqg_fid"] = final_physical["sqg_fid"]
    optimized_parameters["t1_t2_values"] = [{"t1": t1, "t2": t2} for t1, t2 in final_physical["t1_t2_values"]]

    output_json_path = os.path.join(output_dir, device_cfg["output_json"])
    with open(output_json_path, "w") as json_file:
        json.dump(optimized_parameters, json_file, indent=4)

    results_file = os.path.join(output_dir, f"optimization_results_{device_key}_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write(f"Device: {device_name}\n")
        f.write(f"Best objective (TVD): {final_fitness:.6f}\n")
        f.write(f"Generations: {generation_counter}\n")
        f.write(f"Function evaluations: {eval_counter['count']}\n")
        f.write("Final optimized parameters:\n")
        f.write(f"  spam_rates: {final_physical['spam_rates']}\n")
        f.write(f"  spam_rates_initial: {final_physical['spam_rates_initial']}\n")
        f.write(f"  ecr_fid: {final_physical['ecr_fid']}\n")
        f.write(f"  sqg_fid: {final_physical['sqg_fid']}\n")
        f.write(f"  t1_t2_values: {final_physical['t1_t2_values']}\n")

    print(f"[{device_name}] Output saved:")
    print(f"  Parameters JSON: {output_json_path}")
    print(f"  Summary TXT:     {results_file}")
    print(f"  Generation log:  {generation_log}")
    print(f"  Evaluations:     {evaluations_jsonl}")

    return {
        "device": device_name,
        "device_key": device_key,
        "final_fitness": float(final_fitness),
        "generations": generation_counter,
        "evaluations": eval_counter["count"],
        "parameters": final_physical,
        "output_files": {
            "json": output_json_path,
            "results_txt": results_file,
            "generation_log": generation_log,
            "evaluation_log": evaluations_jsonl,
        },
    }


def test_device_data_availability() -> bool:
    """Check whether the required data files and template parameters exist."""
    print("Testing device data availability...")
    print("=" * 60)

    device_key = "target_device"
    device_cfg = REAL_DEVICES[device_key]

    device_ready = True
    missing_files = []
    total_states = 0

    for n in NUM_QUBITS_LIST:
        x_path = device_cfg["data_path_X"].format(n)
        z_path = device_cfg["data_path_Z"].format(n)

        if os.path.exists(x_path):
            try:
                with open(x_path, "r") as f:
                    x_data = json.load(f)
                x_states = len(x_data)
                print(f"  X basis n={n:2d}: {x_states} states")
                total_states += x_states
            except Exception as e:
                print(f"  X basis n={n:2d}: file error - {e}")
                device_ready = False
                missing_files.append(f"X_basis_n{n}(invalid)")
        else:
            print(f"  X basis n={n:2d}: file missing ({x_path})")
            device_ready = False
            missing_files.append(f"X_basis_n{n}")

        if os.path.exists(z_path):
            try:
                with open(z_path, "r") as f:
                    z_data = json.load(f)
                z_states = len(z_data)
                print(f"  Z basis n={n:2d}: {z_states} states")
                total_states += z_states
            except Exception as e:
                print(f"  Z basis n={n:2d}: file error - {e}")
                device_ready = False
                missing_files.append(f"Z_basis_n{n}(invalid)")
        else:
            print(f"  Z basis n={n:2d}: file missing ({z_path})")
            device_ready = False
            missing_files.append(f"Z_basis_n{n}")

    if device_ready:
        print(f"Device '{device_cfg['name']}' is ready (total states: {total_states}).")
    else:
        print(f"Device '{device_cfg['name']}' is not ready. Missing/invalid: {missing_files}")

    if os.path.exists(TEMPLATE_JSON_FILE):
        try:
            with open(TEMPLATE_JSON_FILE, "r") as f:
                _ = json.load(f)
            print(f"Template parameter file OK: {TEMPLATE_JSON_FILE}")
        except Exception as e:
            print(f"Template parameter file invalid: {e}")
            device_ready = False
    else:
        print(f"Template parameter file missing: {TEMPLATE_JSON_FILE}")
        device_ready = False

    return device_ready


def main() -> None:
    """Main entry: optimize parameters for the configured target device."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CURRENT_DIR, f"target_optimization_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("Starting parameter optimization")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")

    if not test_device_data_availability():
        print("Data check failed. Please fix missing/invalid files and retry.")
        return

    print("All required files found. Starting CMA-ES optimization...")

    try:
        result = optimize_single_device("target_device", output_dir)
        print(f"Optimization finished: {result['device']} (best TVD={result['final_fitness']:.6f})")

        summary_file = os.path.join(output_dir, "optimization_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Parameter Optimization Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {result['device']}\n\n")
            f.write("Results:\n")
            f.write("-" * 40 + "\n")
            f.write("Status: success\n")
            f.write(f"Best objective (TVD): {result['final_fitness']:.6f}\n")
            f.write(f"Generations: {result['generations']}\n")
            f.write(f"Function evaluations: {result['evaluations']}\n\n")
            f.write("Output files:\n")
            for k, v in result["output_files"].items():
                f.write(f"  - {k}: {v}\n")

        print(f"Summary report saved: {summary_file}")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
