import os
import yaml
import logging
from dataclasses import field
from jobflow import Flow, job, Response
from autoplex.auto.GenMLFF.jobs import (
    RSS,
    MatterGen,
    evaluate_mlip_ensemble,
    MLscf,
    QEscf,
    dataset_ensembler,
    fit_mlip_ensemble,
    write_restart,
)

#Set logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="mlip_fitting.log"
)


@job
def initial_iteration(
    name: str = "initial_iteration",
    rss_params: dict = field(default_factory=dict),
    mattergen_params: dict = field(default_factory=dict),
    mlip_params: dict = field(default_factory=dict),
    qe_params: dict = field(default_factory=dict),
    dataset_params: dict = field(default_factory=dict),
    train_params: dict = field(default_factory=dict),
):
    """
    Build the initial iteration flow.
    """
    #Define joblist
    joblist = []

    #Initialize the provided structure generation job
    if rss_params is not None: # Use RandomSearchStructure for structure generation
        generation_job = RSS(**rss_params)
    elif mattergen_params is not None: # Use MatterGen for structure generation
        generation_job = MatterGen(mattergen_params)
    joblist.append(generation_job)

    # Initialize Single-point-calulation job
    if qe_params is not None:
        #Update the QE parameters with the ensemble job output
        qe_params['fname_structures'] = generation_job.output
        
        # Run QEscf for single-point calculations
        qe_job = QEscf(qe_params)
        joblist.append(qe_job)
        
        #Get output from the QE job
        scf_output = qe_job.output
    
    elif mlip_params is not None:
        # If MLIP parameters are provided, use MLscf for single-point calculations
        mlip_job = MLscf(**mlip_params, structure_paths=generation_job.output)
        joblist.append(mlip_job)
        #Get output from the MLIP job
        scf_output = mlip_job.output

    else:
        raise ValueError("Either mlip_params or qe_params must be provided for single-point calculations.")

    # Initialize the DatasetEnsembler with the provided parameters and
    dataset_params["labeled_output"] = scf_output
    dataset_job = dataset_ensembler(dataset_params)
    joblist.append(dataset_job)

    # Initialize the MLIPEnsembleMaker with the provided parameters
    mlip_ensemble_job = fit_mlip_ensemble(**train_params, dataset_path=dataset_job.output)
    joblist.append(mlip_ensemble_job)

    #Create a Flow object to manage the jobs
    flow = Flow(
        jobs=joblist, 
        name=name, 
        output={
            "generation_output": generation_job.output,
            "scf_output": scf_output,
            "dataset_output": dataset_job.output,
            "training_output": mlip_ensemble_job.output,
        })

    return Response(
        replace=flow,
        output=flow.output)


@job
def standard_iteration(
    name: str = "standard_iteration",
    rss_params: dict = field(default_factory=dict),
    mattergen_params: dict = field(default_factory=dict),
    ensemble_params: dict = field(default_factory=dict),
    mlip_params: dict = field(default_factory=dict),
    qe_params: dict = field(default_factory=dict),
    dataset_params: dict = field(default_factory=dict),
    train_params: dict = field(default_factory=dict),
    previous_dataset_path: str | None = None,
    previous_mlip_paths: list[str] | None = None,
    previous_mlip_errors: list[float] | None = None,
):
    """
    Build the standard iteration flow.
    """
    #Define joblist
    joblist = []

    #Initialize the provided structure generation job
    if rss_params is not None: # Use RandomSearchStructure for structure generation
        generation_job = RSS(**rss_params)
    elif mattergen_params is not None: # Use MatterGen for structure generation
        generation_job = MatterGen(mattergen_params)
    joblist.append(generation_job)

    #Initialize the RandomizedStructureMaker with the provided parameters
    #Initialized ensemble_params from the previous iteration
    ensemble_params['mlip_paths'] = previous_mlip_paths
    ensemble_params['mlip_errors'] = previous_mlip_errors   
    ensemble_job = evaluate_mlip_ensemble(**ensemble_params, structure_paths=generation_job.output)
    joblist.append(ensemble_job)

    # Initialize Single-point-calulation job
    if qe_params is not None:
        #Update the QE parameters with the ensemble job output
        qe_params['fname_structures'] = ensemble_job.output
        
        # Run QEscf for single-point calculations
        qe_job = QEscf(qe_params)
        joblist.append(qe_job)

        #Get output from the QE job
        scf_output = qe_job.output
    
    elif mlip_params is not None:
        # If MLIP parameters are provided, use MLscf for single-point calculations
        mlip_job = MLscf(**mlip_params, structure_paths=ensemble_job.output)
        joblist.append(mlip_job)
        #Get output from the MLIP job
        scf_output = mlip_job.output

    else:
        raise ValueError("Either mlip_params or qe_params must be provided for single-point calculations.")

    # Initialize the DatasetEnsembler with the provided parameters and from the previous iteration
    dataset_params["labeled_output"] = scf_output
    dataset_params["pre_database_dir"] = previous_dataset_path
    dataset_job = dataset_ensembler(dataset_params)
    joblist.append(dataset_job)

    # Initialize the MLIPEnsembleMaker with the provided parameters
    mlip_ensemble_job = fit_mlip_ensemble(**train_params, dataset_path=dataset_job.output)
    joblist.append(mlip_ensemble_job)

    #Create a Flow object to manage the jobs
    flow = Flow(
        jobs=joblist, 
        name=name, 
        output={
            "rss_output": generation_job.output,
            "scf_output": scf_output,
            "dataset_output": dataset_job.output,
            "training_output": mlip_ensemble_job.output,
        })    

    return Response(
        replace=flow,
        output=flow.output)


@job
def GenMLFlow(
    name: str = "GenMLFlow",
    input_fname: str | None = None,
):
    """
    Build the whole GenMLFF workflow.
    """

    #Read input parameters from a YAML file if provided
    if input_fname is not None:
        with open(input_fname, 'r') as f:
            GenML_params = yaml.safe_load(f)
    else:
        raise ValueError("input_fname must be provided to read the input parameters.")

    #Unpack the parameters
    rss_params = GenML_params.get("rss_params", None)
    mattergen_params = GenML_params.get("mattergen_params", None)
    ensemble_params = GenML_params.get("ensemble_params", None)
    mlip_params = GenML_params.get("mlip_params", None)
    qe_params = GenML_params.get("qe_params", None)
    dataset_params = GenML_params.get("dataset_params", None)
    train_params = GenML_params.get("train_params", None)

    #Assertions for required parameters
    assert rss_params is not None or mattergen_params is not None, "rss_params or mattergen_params must be provided"
    assert ensemble_params is not None, "ensemble_params must be provided"
    assert mlip_params is not None or qe_params is not None, "mlip_params or qe_params must be provided"
    assert dataset_params is not None, "dataset_params must be provided"
    assert train_params is not None, "train_params must be provided"

    #Read restart file
    restart_params, restart_fname = {}, os.path.dirname(input_fname) + "/GenMLFF_restart.yaml"
    if os.path.exists(restart_fname):
        logging.info(f"Found restart file {restart_fname}, reading restart parameters...")

        #Read restart file, contains:
        # "iteration" : 0,  # The previous iteration index
        # "dataset_output" : "previous_dataset_path"
        # "mlip_paths" : ["previous_mlip_path1", "previous_mlip_path2", ...]
        # "mlip_errors" : [0.01, 0.02, ...]
        with open(restart_fname, 'r') as f:
            restart_params = yaml.safe_load(f)

    #Define joblist
    joblist = []

    #If restart parameters are provided, use them to initialize the previous iteration
    if restart_params:
        logging.info(f"Restarting from iteration {restart_params['iteration']}")
        previous_iteration = standard_iteration(
            rss_params=rss_params,
            mattergen_params=mattergen_params,
            ensemble_params=ensemble_params,
            mlip_params=mlip_params,
            qe_params=qe_params,
            dataset_params=dataset_params,
            train_params=train_params,
            previous_dataset_path=restart_params.get('dataset_output'),
            previous_mlip_paths=restart_params.get('mlip_paths'),
            previous_mlip_errors=restart_params.get('mlip_errors'),
        )
        previous_iteration.name = f"standard_iteration_{restart_params['iteration'] + 1}"
        joblist.append(previous_iteration)
    
    else:
        logging.info(f"Starting GenMLFF workflow from scratch, setting up iteration 0")
        #Add the initial iteration job
        previous_iteration = initial_iteration(
            rss_params=rss_params,
            mattergen_params=mattergen_params,
            mlip_params=mlip_params,
            qe_params=qe_params,
            dataset_params=dataset_params,
            train_params=train_params,
        )
        joblist.append(previous_iteration)

    #Get the number of remaining iterations
    num_remaining_iterations = GenML_params.get("num_iterations", 1) - 1

    #Get current iteration number
    current_iteration_id = restart_params.get('iteration') + 1 if restart_params else 1

    #Loop over the number of requested iterations
    for i in range(current_iteration_id, current_iteration_id + num_remaining_iterations):
        #Add the standard iteration job for each iteration
        current_iteration = standard_iteration(
            rss_params=rss_params,
            mattergen_params=mattergen_params,
            ensemble_params=ensemble_params,
            mlip_params=mlip_params,
            qe_params=qe_params,
            dataset_params=dataset_params,
            train_params=train_params,
            previous_dataset_path=previous_iteration.output['dataset_output'],
            previous_mlip_paths=previous_iteration.output['training_output']['mlip_paths'],
            previous_mlip_errors=previous_iteration.output['training_output']['train_errors'],
        )
        current_iteration.name = f"standard_iteration_{i}"
        joblist.append(current_iteration)

        #Update the previous iteration to the current one
        previous_iteration = current_iteration
    
    #Define write_restart job
    write_restart_job = write_restart(
        restart_fname=restart_fname,
        current_iteration_id=int(current_iteration_id + num_remaining_iterations),
        final_dataset_path=previous_iteration.output['dataset_output'],
        final_mlip_paths=previous_iteration.output['training_output']['mlip_paths'],
        final_mlip_errs=previous_iteration.output['training_output']['train_errors'],
    )
    joblist.append(write_restart_job)

    #Create a Flow object to manage the jobs
    flow = Flow(jobs=joblist, name=name)

    return Response(replace=flow, output={
            "final_dataset": previous_iteration.output['dataset_output'],
            "final_models": previous_iteration.output['training_output'],
        })