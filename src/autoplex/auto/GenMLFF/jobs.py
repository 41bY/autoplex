"""Jobs for running the workflow."""

import yaml
from dataclasses import field
from jobflow import job, Flow, Response
from autoplex.auto.GenMLFF.rss import RandomizedStructureMaker
from autoplex.auto.GenMLFF.labelling import MLIPStaticLabelling, QEstaticLabelling, qe_params_from_config
from autoplex.auto.GenMLFF.dataset import DatasetMaker, data_ensembler_from_config
from autoplex.auto.GenMLFF.training import MLIPEnsembleMaker
from autoplex.auto.GenMLFF.sampling import EnsembleEvaluatorMaker
from autoplex.auto.GenMLFF.mattergen import MatterGenMaker, params_from_config


#STRUCTURE GENERATION JOBS
@job
def RSS(
    name: str = "random_search_structure_job",
    tag: str = None,
    output_file_name: str = "random_structs.extxyz",
    generated_struct_numbers: list[int] | None = None,
    buildcell_options: list[dict] | None = None,
    cell_seed_path: str | None = None,
    fragment_file: str | None = None,
    fragment_numbers: list[str] | None = None,
    remove_tmp_files: bool = True,
    num_processes: int = 1,
):
    """
    Initialize the RandomizedStructureMaker with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the RandomizedStructureMaker.
    """
    #Parameters for RandomizedStructureMaker
    rss_params = {
        "name": name,
        "tag": tag,
        "output_file_name": output_file_name,
        "generated_struct_numbers": generated_struct_numbers,
        "buildcell_options": buildcell_options,
        "cell_seed_path": cell_seed_path,
        "fragment_file": fragment_file,
        "fragment_numbers": fragment_numbers,
        "remove_tmp_files": remove_tmp_files,
        "num_processes": num_processes,
    }

    # Execute randomized structure generation
    # and return the paths to the generated structures
    randomized_structures_paths = RandomizedStructureMaker(**rss_params).make()

    return randomized_structures_paths

@job
def MatterGen(
    params: dict | None = None,
):
    """
    Initialize the MatterGenMaker with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the MatterGenMaker.
    """
    #Initialize MatterGenMaker with the provided parameters
    matgen_params = params_from_config(params)

    # Execute MatterGen generation
    generated_structure_paths = MatterGenMaker(**matgen_params).make()

    return generated_structure_paths
###################################################################################


# MLIP EVALUATION JOB
@job
def evaluate_mlip_ensemble(
    name: str = "do_relaxation_and_sampling",
    mlip_type: str | None = None,
    mlip_paths: list[str] | None = None,
    mlip_errors: list[float] | None = None,
    mlip_kwargs: dict | None = None,
    structure_paths: list[str] | None = None,
    pre_trained_model: str | None = None,
    pre_trained_kwargs: dict | None = None,
):
    """
    Initialize the MLIPEnsembleEvaluator with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the MLIPEnsembleEvaluator.
    """
    #Check if the mlip_paths and structure_paths are provided
    if mlip_paths is None or structure_paths is None:
        raise ValueError("mlip_paths and structure_paths must be provided.")
    
    #Collect parameters for MLIPEnsembleEvaluator
    ensemble_evaluator_params = {
        "mlip_type": mlip_type,
        "mlip_paths": mlip_paths,
        "mlip_errors": mlip_errors,
        "mlip_kwargs": mlip_kwargs,
        "structure_paths": structure_paths,
        "pre_trained_model": pre_trained_model,
        "pre_trained_kwargs": pre_trained_kwargs,
    }

    # Execute relaxation of the generated structures
    # Perform the ensemble evaluation and sample the most "problematic" structures
    # Return the paths to the sampled structures
    evaluated_structures_path = EnsembleEvaluatorMaker(**ensemble_evaluator_params).make()

    return evaluated_structures_path
####################################################################################


# STATIC LABELLING JOBS
@job
def MLscf(
    name: str = "do_mlip_static_labelling",
    mlip_type: str | None = None,
    mlip_kwargs: dict | None = None,
    structure_paths: list[str] | None = None,
    structure_types: list[str] | None = None,
    isolated_atom: bool = False,
    isolated_species: list[str] | None = None,
    isolatedatom_box: list[float] | None = None,
    dimer: bool = False,
    dimer_pairs: list[tuple[str]] | None = None,
    dimer_range: list[float] | None = None,
    dimer_num: int = 21,
    dimer_box: list[float] | None = None,
):
    """
    Initialize the MLIPStaticLabelling with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the MLIPStaticLabelling.
    """
    #Parameters for MLIPStaticLabelling
    mlip_params = {
        "name": name,
        "mlip_type": mlip_type,
        "mlip_kwargs": mlip_kwargs,    
        "structure_paths": structure_paths,
        "structure_types": structure_types,    
        "isolated_atom": isolated_atom,
        "isolated_species": isolated_species,
        "isolatedatom_box": isolatedatom_box,
        "dimer": dimer,
        "dimer_pairs": dimer_pairs,
        "dimer_range": dimer_range,
        "dimer_num": dimer_num,
        "dimer_box": dimer_box,
    }

    # Execute MLIP static labelling
    # and return the paths to the labelled structures
    labelled_structures_path = MLIPStaticLabelling(**mlip_params).make()

    return labelled_structures_path

@job
def QEscf(
    params: dict | None = None,
):
    """
    Initialize the QEScfLabelling with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the QEScfLabelling.
    """
    #Initialize QEstaticLabelling with the provided parameters
    qe_params = qe_params_from_config(params)    

    # Execute QE static labelling
    # return a list containing (list[success], list[pwo], list[outdir]) for each worker
    output_per_worker = QEstaticLabelling(**qe_params).make()

    return output_per_worker
###################################################################################


# DATSET PREPARATION JOB
@job
def dataset_ensembler(
    params: dict | None = None,
):
    """
    Initialize the DatasetEnsembler with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the DatasetEnsembler.
    """
    #Initialize DatasetMaker with the provided parameters
    dataset_maker_params = data_ensembler_from_config(params)

    # Execute dataset ensembling
    # and return the path to the ensembled dataset
    ensembled_dataset_paths = DatasetMaker(**dataset_maker_params).make()

    return ensembled_dataset_paths
##################################################################################

# TRAINING JOB
@job
def fit_mlip_ensemble(
    name: str = "do_mlip_ensemble_fitting",
    dataset_path: str | None = None,
    mlip_type: str | None = None,
    mlip_train_kwargs: dict = field(default_factory=dict),
    num_models: int = 1,
    remove_model_datasets: bool = True,
):
    """
    Initialize the MLIPEnsembleMaker with the provided parameters.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the MLIPEnsembleMaker.
    """
    #Check if the dataset_path is provided
    if dataset_path is None:
        raise ValueError("dataset_path must be provided.")
    
    #Collect parameters for MLIPEnsembleMaker
    mlip_ensemble_params = {
        "dataset_path": dataset_path,
        "mlip_type": mlip_type,
        "mlip_train_kwargs": mlip_train_kwargs,
        "num_models": num_models,
        "remove_model_datasets": remove_model_datasets,
    }

    # Execute MLIP ensemble fitting: return {"mlip_paths": [paths], "train_errors": [floats], "test_errors": [floats], "converges": [bools]}
    fitted_model_paths = MLIPEnsembleMaker(**mlip_ensemble_params).make()

    return fitted_model_paths
###################################################################################

# WRITING RESTART FILE JOB
@job
def write_restart(
    name: str = "do_write_restart",
    restart_fname: str | None = None,
    current_iteration_id: int | None = None,
    final_dataset_path: str | None = None,
    final_mlip_paths: list[str] | None = None,
    final_mlip_errs: list[float] | None = None,
):
    """
    Write the restart file for the GenMLFF workflow.

    Parameters
    ----------
    kwargs: dict
        Dictionary containing the parameters for the restart.
    """
    #Assert that every argument is provided
    if restart_fname is None or current_iteration_id is None or final_dataset_path is None or final_mlip_paths is None or final_mlip_errs is None:
        raise ValueError("All parameters must be provided to write a valid restart file.")
    
    #Dump restart parameters into a dictionary
    restart_params = {
        "iteration": current_iteration_id,
        "dataset_output": final_dataset_path,
        "mlip_paths": final_mlip_paths,
        "mlip_errors": final_mlip_errs,
    }

    #Write the restart file in .yaml format
    with open(restart_fname, 'w') as restart_file:
        yaml.dump(restart_params, restart_file, default_flow_style=False)

    return restart_fname
###################################################################################