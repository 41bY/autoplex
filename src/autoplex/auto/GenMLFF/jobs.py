"""Jobs for running the workflow."""

from jobflow import job

from autoplex.auto.GenMLFF.rss import RandomizedStructureMaker


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
def MLIPStaticLabelling():
    pass


do_mlip_static_labelling = MLIPStaticLabelling(
    name="do_mlip_labelling",
    isolated_atom=False,
    isolated_species=None,
    dimer=False,
    dimer_species=None,
    dimer_range=None,
    dimer_num=21,
    mlip_type="MACE",
    mlip_path="/leonardo_work/EUHPC_A04_113/Alberto/mace/pre-trained-models/mace-mpa-0-medium.model", #Pre-trained model: testing
    mlip_kwargs={"device" : "cuda"},
).make(structure_paths=do_randomized_structure_generation.output, config_type=config_type) #Assume structure paths is list[path]