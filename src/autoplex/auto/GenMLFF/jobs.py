"""Jobs for running the workflow."""

from jobflow import job
from dataclasses import field
from autoplex.auto.GenMLFF.rss import RandomizedStructureMaker
from autoplex.auto.GenMLFF.labelling import MLIPStaticLabelling


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