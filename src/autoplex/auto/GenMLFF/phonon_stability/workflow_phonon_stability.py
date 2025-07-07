import argparse
import numpy as np

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from jobflow import Flow, job
from jobflow_remote import submit_flow, set_run_config
from autoplex.auto.GenMLFF.phonon_stability.phonon_stability import local_phonon_flow, compute_stability_metrics

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


def stability_flow(
        data_path, 
        model_path, 
        model_name="MatterSim",
        data_name="data", 
        max_parallel_flows=None,
        compute_metrics=False,
        ):
    #Read the structures from the file
    structures = read(data_path, index=":")

    #Define phonon maker kwargs
    ph_maker_kwargs = {
        "min_length": 3.0,
        "use_symmetrized_structure": "conventional",
        "create_thermal_displacements": False,
        "store_force_constants": False,
        "prefer_90_degrees": False,
        "generate_frequencies_eigenvectors_kwargs": {"tstep": 100},
    }

    #Define static maker kwargs
    st_maker_kwargs = {
        "force_field_name": f"MLFF.{model_name}", # MLFF.MACE, MLFF.MatterSim, MLFF.META_eqv2
        # "calculator_kwargs": {"model": model_path, "device" : "cuda"} #MACE    
        # "calculator_kwargs": {"load_path": model_path, "device" : "cuda"} #MatterSim        
        "calculator_kwargs": {"checkpoint_path": model_path, "cpu": False, "seed": 42}, #META_eqv2
    }

    #Define relax maker kwargs
    relax_maker_kwargs = {
        "force_field_name": f"MLFF.{model_name}",
        "calculator_kwargs": {"checkpoint_path": model_path, "cpu": False, "seed": 42}         
    }

    #Define joblist
    joblist, output = [], []

    #Assign a unique id to each structure
    structures_idxs = [idx for idx, _ in enumerate(structures)]

    #Divide the list of structures and idxs in batches if max_parallel_flows is set
    if max_parallel_flows is not None:
        #Get batch size
        batch_size = max(1, len(structures) // max_parallel_flows)
        num_batches = len(structures)//batch_size if len(structures) % batch_size == 0 else len(structures)//batch_size + 1
    else: 
        batch_size, num_batches = 1, len(structures)

    #Create batches of structures and idxs
    # (num_batches - 1) batches of size batch_size and one last batch with the remaining structures
    structure_batches = [
        structures[i * batch_size: (i + 1) * batch_size] for i in range(num_batches - 1)
    ] + [structures[(num_batches - 1) * batch_size:]]
    idxs_batches = [
        structures_idxs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches - 1)
    ] + [structures_idxs[(num_batches - 1) * batch_size:]]        

    # Define one serial_phononflow job for each structure batch
    for structures, idxs in zip(structure_batches, idxs_batches):
        # Convert ASE Atoms objects to Pymatgen Structures
        pmg_structures = [AseAtomsAdaptor.get_structure(structure) for structure in structures]

        # Create a serial_phonon flow for that batch of structures
        serial_ph = local_phonon_flow(
            ph_makers_kwargs=ph_maker_kwargs, 
            st_makers_kwargs=st_maker_kwargs,
            relax_makers_kwargs=relax_maker_kwargs,
            pmg_structures=pmg_structures,
            idxs_structures=idxs,
            output_dir=f"results_{model_name}_{data_name}",  # Directory to store the results
        )
        if len(idxs) == 1:
            serial_ph.name = f"phonons_{model_name}_{data_name}_{idxs[0]}"
        else:
            serial_ph.name = f"phonons_{model_name}_{data_name}_{idxs[0]}-{idxs[-1]}"

        #Populate the joblist and output
        joblist.append(serial_ph), output.append(serial_ph.output) #output is a list (batches) of list (batched_structures) of dict (result)
    #Assemble the phonons flow
    phonons_flow = Flow(joblist, output, name=f"ph_{model_name}_{data_name}_stability")
    
    if compute_metrics:
        #Compute dynamic stability metric for each structure
        metric_job = compute_stability_metrics(phonons_flow.output, neg_modes_threshold=0.01)
        metric_job.name = "stability_metrics_job"

        #Assemble the stability flow
        final_flow = Flow([phonons_flow, metric_job], metric_job.output, name=f"ph_{model_name}_{data_name}_stability")
    else:
        final_flow = phonons_flow

    return final_flow


parser = argparse.ArgumentParser(
    description="Submit a workflow to calculate phonon properties using Atomate2."
)
parser.add_argument(
    "--structures_file",
    type=str,
    required=True,
    help="Path to ASE-readable structures file",
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
    "--data_name",
    type=str,
    default="data",
    help="Name to use for the data in the flow. This will be used to name the jobs.",
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
structure_path = args.structures_file
model_path = args.MLIP_path
model_name = args.MLIP_name
data_name = args.data_name
max_parallel_flows = args.max_parallel_flows
compute_metrics = args.compute_metrics

#Assemble the stability flow
flow = stability_flow(
    data_path=structure_path, 
    model_path=model_path, 
    model_name=model_name,
    data_name=data_name,
    max_parallel_flows=max_parallel_flows,
    compute_metrics=compute_metrics,
    )

#Proper configuration of the flow
flow = set_run_config(
    flow, name_filter=f"phonons_", resources=serial_gpu_resources, worker="meta_worker", exec_config="dump_config"
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