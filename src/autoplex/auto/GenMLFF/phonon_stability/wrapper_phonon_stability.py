from jobflow import job
from autoplex.auto.GenMLFF.phonon_stability.phonon_stability import StabilityPhononFlowMaker

@job
def wrap_stability_flow(
        model_path: str | None = None, 
        model_name: str | None = None,
        model_kwargs: dict | None = None,
        phonon_kwargs: dict | None = None,
        data_path:str | None = None, 
        data_name: str | None = None, 
        max_parallel_flows: int | None = None,
        compute_metrics: bool = False,
        ):
    """
    Wrapper function to create a stability flow for phonon calculations.
    Parameters:
        model_path (str): Path to the MLIP model.
        model_name (str): Name of the MLIP model.
        model_kwargs (dict): Dictionary of model calculator kwargs.
        phonon_kwargs (dict): Dictionary of phonon calculation parameters.
        data_path (str): Path to the data file containing structures.
        data_name (str): Name for the data in the flow.
        max_parallel_flows (int): Maximum number of parallel flows to run.
        compute_metrics (bool): Whether to compute stability metrics.
    Returns:
        Response: Replace this job with an instance of StabilityPhononFlowMaker that contains the stability flow.
    """
    # Create an instance of the StabilityPhononFlowMaker
    stability_flow_maker = StabilityPhononFlowMaker(
        model_path=model_path,
        model_name=model_name,
        model_kwargs=model_kwargs,
        max_parallel_flows=max_parallel_flows,
        compute_metrics=compute_metrics
    )

    # Create the stability flow using the maker
    stability_flow = stability_flow_maker.make(
        phonon_kwargs=phonon_kwargs,
        data_path=data_path,
        data_name=data_name
    )

    return stability_flow