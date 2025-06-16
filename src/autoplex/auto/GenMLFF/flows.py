import yaml
from dataclasses import field
from jobflow import Flow, job, Response
from autoplex.auto.GenMLFF.jobs import (
    RSS,
    evaluate_mlip_ensemble,
    MLscf,
    QEscf,
    dataset_ensembler,
    fit_mlip_ensemble,
)


@job
def initial_iteration(
    name: str = "initial_iteration",
    rss_params: dict = field(default_factory=dict),
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

    #Initialize the RandomizedStructureMaker with the provided parameters
    rss_job = RSS(**rss_params)
    joblist.append(rss_job)

    # Initialize Single-point-calulation job
    if qe_params is not None:
        # If QE parameters are provided, use QEscf for single-point calculations
        qe_job = QEscf(**qe_params, fname_structures=rss_job.output)
        joblist.append(qe_job)
        #Get output from the QE job
        scf_output = qe_job.output
    
    elif mlip_params is not None:
        # If MLIP parameters are provided, use MLscf for single-point calculations
        mlip_job = MLscf(**mlip_params, structure_paths=rss_job.output)
        joblist.append(mlip_job)
        #Get output from the MLIP job
        scf_output = mlip_job.output

    else:
        raise ValueError("Either mlip_params or qe_params must be provided for single-point calculations.")

    # Initialize the DatasetEnsembler with the provided parameters and
    dataset_job = dataset_ensembler(**dataset_params, labeled_output=scf_output)
    joblist.append(dataset_job)

    # Initialize the MLIPEnsembleMaker with the provided parameters
    mlip_ensemble_job = fit_mlip_ensemble(**train_params, dataset_path=dataset_job.output)
    joblist.append(mlip_ensemble_job)

    #Create a Flow object to manage the jobs
    flow = Flow(
        jobs=joblist, 
        name=name, 
        output={
            "rss_output": rss_job.output,
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
    ensemble_params: dict = field(default_factory=dict),
    mlip_params: dict = field(default_factory=dict),
    qe_params: dict = field(default_factory=dict),
    dataset_params: dict = field(default_factory=dict),
    train_params: dict = field(default_factory=dict),
):
    """
    Build the standard iteration flow.
    """
    #Define joblist
    joblist = []

    #Initialize the RandomSearchStructure with the provided parameters
    rss_job = RSS(**rss_params)
    joblist.append(rss_job)

    #Initialize the RandomizedStructureMaker with the provided parameters
    #TODO: Model paths inside ensemble_params should be initialized from the previous iteration
    ensemble_job = evaluate_mlip_ensemble(**ensemble_params, structure_paths=rss_job.output)
    joblist.append(ensemble_job)

    # Initialize Single-point-calulation job
    if qe_params is not None:
        # If QE parameters are provided, use QEscf for single-point calculations
        qe_job = QEscf(**qe_params, fname_structures=ensemble_job.output)
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

    # Initialize the DatasetEnsembler with the provided parameters and
    dataset_job = dataset_ensembler(**dataset_params, labeled_output=scf_output)
    joblist.append(dataset_job)

    # Initialize the MLIPEnsembleMaker with the provided parameters
    mlip_ensemble_job = fit_mlip_ensemble(**train_params, dataset_path=dataset_job.output)
    joblist.append(mlip_ensemble_job)

    #Create a Flow object to manage the jobs
    flow = Flow(
        jobs=joblist, 
        name=name, 
        output={
            "rss_output": rss_job.output,
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
    ensemble_params = GenML_params.get("ensemble_params", None)
    mlip_params = GenML_params.get("mlip_params", None)
    qe_params = GenML_params.get("qe_params", None)
    dataset_params = GenML_params.get("dataset_params", None)
    train_params = GenML_params.get("train_params", None)

    #Assertions for required parameters
    assert rss_params is not None, "rss_params must be provided"
    assert ensemble_params is not None, "ensemble_params must be provided"
    assert mlip_params is not None or qe_params is not None, "mlip_params or qe_params must be provided"
    assert dataset_params is not None, "dataset_params must be provided"
    assert train_params is not None, "train_params must be provided"


    #Define joblist
    joblist = []

    #Add the initial iteration job
    previous_iteration = initial_iteration(
        rss_params=rss_params,
        mlip_params=mlip_params,
        qe_params=qe_params,
        dataset_params=dataset_params,
        train_params=train_params,
    )
    joblist.append(previous_iteration)

    #Loop over the number of requested iterations
    for i in range(GenML_params.get("num_iterations", 1)):
        #Get ensemble MLIP
        ensemble_params['mlip_paths'] = previous_iteration.output['training_output']['mlip_paths']
        ensemble_params['mlip_errors'] = previous_iteration.output['training_output']['train_errors']

        #Add the standard iteration job for each iteration
        current_iteration = standard_iteration(
            rss_params=rss_params,
            ensemble_params=ensemble_params,
            mlip_params=mlip_params,
            qe_params=qe_params,
            dataset_params=dataset_params,
            train_params=train_params,
        )
        current_iteration.name = f"standard_iteration_{i+1}"
        joblist.append(current_iteration)

        #Update the previous iteration to the current one
        previous_iteration = current_iteration

    #Create a Flow object to manage the jobs
    flow = Flow(jobs=joblist, name=name)

    return Response(replace=flow, output={
            "final_dataset": previous_iteration.output['dataset_output'],
            "final_models": previous_iteration.output['training_output'],
        })