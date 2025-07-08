import json
import argparse
from jobflow_remote import submit_flow, set_run_config
from autoplex.auto.GenMLFF.phonon_stability.wrapper_phonon_stability import wrap_stability_flow

serial_gpu_resources = {
    "account": "IscrB_MLSilDia",
    "partition": "boost_usr_prod",
    "qos": "boost_qos_dbg",
    "time": "00:30:00",
    "nodes": 1,
    "ntasks_per_node": 1,
    "cpus_per_task": 8,
    "gres": "gpu:1",
    "mem": "120000",
    "job_name": "stability_phonon_mlip",
    "qerr_path": "stability_phonon_mlip.err",
    "qout_path": "stability_phonon_mlip.out",   
}

serial_cpu_resources = {
    "account": "EUHPC_A04_113",
    "partition": "lrd_all_serial",
    "qos": "boost_qos_dbg",
    "time": "00:30:00",
    "nodes": 1,
    "ntasks_per_node": 1,
    "cpus_per_task": 4,
    "gres": "gpu:0",
    "mem": "30000",
    "job_name": "Sampling",
    "qerr_path": "Sampling.err",
    "qout_path": "Sampling.out",
}


parser = argparse.ArgumentParser(
    description="Submit a workflow to calculate phonon properties using Atomate2."
)
parser.add_argument(
    "--MLIP_path",
    type=str,
    required=True,
    help="Path to the MACE force field to use for the phonon calculations.",
)
parser.add_argument(
    "--MLIP_name",
    type=str,
    required=True,
    help="Name defining the MLIP model to use for the phonon calculations performed by atomate2.",
)
parser.add_argument(
    "--MLIP_kwargs",
    type=json.loads,
    required=True,
    help="JSON string with model calculator kwargs to use for the phonon calculations.",
)
# "calculator_kwargs": {"model": model_path, "device" : "cuda"} #MACE    
# "calculator_kwargs": {"load_path": model_path, "device" : "cuda"} #MatterSim        
# "calculator_kwargs": {"checkpoint_path": model_path, "cpu": False, "seed": 42}, #META_eqv2
parser.add_argument(
    "--data_file",
    type=str,
    required=True,
    help="Path to ASE-readable structures file",
)
parser.add_argument(
    "--data_name",
    type=str,
    default="data",
    help="Name to use for the data in the flow. This will be used to name the jobs.",
)
parser.add_argument(
    "--phonon_maker_kwargs",
    type=json.loads,
    default={},
    help="JSON string with kwargs to use for the phonon flow maker. If not provided, default values will be used.",
)
parser.add_argument(
    "--max_parallel_flows",
    type=int,
    default=None,
    help="Maximum number of parallel phonon flows to run. If None, all phonon flows are run in parallel.",
)
parser.add_argument(
    "--compute_metrics",
    action="store_true",
    default=False,
    help="If set, compute the stability metrics for the phonon calculations. If not set, only the phonon calculations are performed.",
)

#Parse the arguments
args = parser.parse_args()

#Model parameters
model_path = args.MLIP_path
model_name = args.MLIP_name
model_kwargs = args.MLIP_kwargs

#Data parameters
data_file = args.data_file
data_name = args.data_name

#Phonon parameters
phonon_kwargs = {
            "min_length": 3.0,
            "use_symmetrized_structure": "conventional",
            "create_thermal_displacements": False,
            "store_force_constants": False,
            "prefer_90_degrees": False,
            "generate_frequencies_eigenvectors_kwargs": {"tstep": 100},
        }
if args.phonon_maker_kwargs:
    phonon_kwargs.update(args.phonon_maker_kwargs)

#Flow parameters
max_parallel_flows = args.max_parallel_flows
compute_metrics = args.compute_metrics

#Assemble the stability flow
flow = wrap_stability_flow(
    model_path=model_path, 
    model_name=model_name,
    model_kwargs=model_kwargs,
    phonon_kwargs=phonon_kwargs,
    data_path=data_file, 
    data_name=data_name, 
    max_parallel_flows=max_parallel_flows,
    compute_metrics=compute_metrics
    )

#Proper configuration of the flow
flow = set_run_config(
    flow, name_filter=f"phonons_", resources=serial_gpu_resources, worker="schedule_worker", #exec_config="meta_config"
)
flow = set_run_config(
    flow, name_filter="stability_metrics", exec_config="dump_config"
)

#Submit the flow
submit_flow(
    flow, worker="local_worker", 
    resources={}, 
    project="PhononStability"
)