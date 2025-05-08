"""Class for Random Search Structure execution."""

import os
import re
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from shutil import which
from subprocess import run

import ase.io
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from jobflow import Maker
from monty.dev import requires
from pymatgen.core import Element


@dataclass
class RandomizedStructureMaker(Maker):
    """
    Maker to create random structures using the 'buildcell' tool.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    struct_number : int
        Expected number of generated randomized unit cells.
    tag: str
        Tag of systems. It can also be used for setting up elements and stoichiometry.
        For example, 'SiO2' will generate structures with a 2:1 ratio of Si to O.
    output_file_name: str
        Name of the file to store all generated structures.
    remove_tmp_files: bool
        Remove all temporary files raised by buildcell to save memory.
    buildcell_option: dict
        Customized parameters for buildcell.
    cell_seed_path: str
        Path to the custom buildcell control file, which ends with '.cell'. If this file exists,
        the buildcell_option argument will no longer take effect.
    num_processes: int
        Number of processes to use for parallel computation.
    fragment: Atoms | list[Atoms] (optional)
        Fragment(s) for random structures, e.g. molecules, to be placed indivudally intact.
        atoms.arrays should have a 'fragment_id' key with unique identifiers for each fragment if in same Atoms.
        atoms.cell must be defined (e.g. Atoms.cell = np.eye(3)*20).
    fragment_numbers: list[str] (optional)
        Numbers of each fragment to be included in the random structures. Defaults to 1 for all specified.
    """
    name: str = "do_randomized_structure_generation"
    tag: str = "SiO2"
    output_file_name: str = "random_structs.extxyz"
    generated_struct_numbers: list[int] | None = None
    buildcell_options: list[dict] | None = None
    cell_seed_path: str | None = None
    fragment_file: str | None = None
    fragment_numbers: list[str] | None = None
    remove_tmp_files: bool = True
    num_processes: int = 1

    @requires(
        which("buildcell"),
        "RSS flows requires the executable 'buildcell' to be in PATH. "
        "Please follow the instructions in the autoplex documentation to install "
        "the AIRSS library and add it to PATH. Link to the documentation:"
        " https://autoatml.github.io/autoplex/user/index.html#enabling-rss-workflows",
    )

    def make(self):
        #Define the job list and the structures paths (output)
        job_list, structures_paths = [], []

        #Loop over the number of distinct structure generations
        for i, struct_number in enumerate(self.generated_struct_numbers):
            buildcell_option = None
            if self.buildcell_options is not None:
                assert len(self.generated_struct_numbers) == len(self.buildcell_options)
                buildcell_option = self.buildcell_options[i]

            # Create a job for each generations
            fout_name = f"{self.output_file_name}_{i}.extxyz"
            structures_path = self.random_structure_generation(
                tag=self.tag,
                struct_number=struct_number,
                buildcell_option=buildcell_option,
                output_file_name=fout_name,
                remove_tmp_files=self.remove_tmp_files,                
                cell_seed_path=self.cell_seed_path,                
                fragment_file=self.fragment_file,
                fragment_numbers=self.fragment_numbers,
                num_processes=self.num_processes,
                )
            # job_struct.name = f"do_randomized_structure_generation_{i}"

            #TODO: Selection/Sampling of the structures will be performed by another job
            #Append job to list of jobs
            # job_list.append(job_struct)
            structures_paths.append(structures_path) #job_struct.output is a list of paths to extxyz files (structures)   

        # return Response(replace=Flow(job_list), output=structures_paths)
        # return Flow(jobs=job_list, output=structures_paths, name="do_randomized_structure_generation")     
        return structures_paths

    def random_structure_generation(
        self, 
        tag: str,
        struct_number: int, 
        output_file_name: str = "random_structs.extxyz",
        remove_tmp_files: bool = True,
        cell_seed_path: str | None = None,
        buildcell_option: dict | None = None,
        fragment_file: str | None = None,
        fragment_numbers: list[str] | None = None,
        num_processes: int = 1,
        )-> str:
        """Generate random structures using the 'buildcell' tool.
        Returns the path of the generated structures."""

        if cell_seed_path:
            if not os.path.isfile(cell_seed_path):
                raise FileNotFoundError(
                    f"No file found at the specified path: {cell_seed_path}"
                )
            bc_file = cell_seed_path

        else:
            buildcell_parameters = [
                "SLACK=0.25",
                "OVERLAP=0.1",
                "COMPACT",
                "MINSEP=1.5",
            ]

            if buildcell_option is not None:
                buildcell_parameters = self._update_buildcell_option(
                    buildcell_option, buildcell_parameters
                )

            elements = self._extract_elements(tag)  # {"Si":1, "O":2}

            if "SPECIES" in buildcell_option and fragment_file is not None:
                raise ValueError(
                    "Cannot use 'SPECIES' and 'fragment' together in buildcell options.\n"
                    "Specify your fragment only and use NFORM to control their number."
                )

            if buildcell_option is None or (
                "SPECIES" not in buildcell_option and fragment_file is None
            ):
                make_species = self._make_species(elements)  # Si%NUM=1,O%NUM=2
                buildcell_parameters = self._update_buildcell_option(
                    {"SPECIES": make_species}, buildcell_parameters
                )

            if (
                buildcell_option is None
                or (
                    "VARVOL" not in buildcell_option
                    and "TARGVOL" not in buildcell_option
                )
                or "MINSEP" not in buildcell_option
            ):
                r0 = {}
                varvol = {}
                num_atom_formula = 0
                total_varvol_formula = 0

                for ele in elements:
                    r0[ele] = covalent_radii[atomic_numbers[ele]]

                    if Element(ele).is_metal:
                        varvol[ele] = 5.5 * np.power(r0[ele], 3)
                    else:
                        varvol[ele] = 14.5 * np.power(r0[ele], 3)

                    total_varvol_formula += varvol[ele] * elements[ele]

                    num_atom_formula += elements[ele]

                if buildcell_option is None or (
                    "VARVOL" not in buildcell_option
                    and "TARGVOL" not in buildcell_option
                ):
                    mean_var = total_varvol_formula / num_atom_formula * len(elements)
                    buildcell_parameters = self._update_buildcell_option(
                        {"TARGVOL": f"{mean_var*0.8}-{mean_var*1.2}"},
                        buildcell_parameters,
                    )

                if (
                    buildcell_option is None
                    or "MINSEP" not in buildcell_option
                ):
                    minsep = self._make_minsep(r0)
                    buildcell_parameters = self._update_buildcell_option(
                        {
                            "MINSEP": minsep,
                        },
                        buildcell_parameters,
                    )

            if fragment_file is not None:
                fragment = ase.io.read(fragment_file, index=":")

                if len(fragment) == 1:
                    fragment = fragment[0]

                if isinstance(fragment, Atoms):
                    if fragment_numbers is None:
                        fragment_numbers = [1 for _ in fragment]
                    if "fragment_id" not in fragment.arrays:
                        fragment.arrays["fragment_id"] = [
                            f"{1}-f" for i in fragment
                        ]

                elif isinstance(fragment, list):
                    if fragment_numbers is None:
                        fragment_numbers = [
                            1 for _ in range(sum([len(i) for i in fragment]))
                        ]
                    write_fragment = fragment[0]
                    for frag in fragment[
                        1:
                    ]:  # merge all separate fragments into one Atoms object
                        write_fragment += frag

                fragment_parameters = [
                    "%BLOCK POSITIONS_ABS",
                ]
                symbols = fragment.get_chemical_symbols()
                for i, val in enumerate(fragment.get_positions(wrap=True)):
                    if i == 0:
                        newline = (
                            f"{symbols[i]} {val[0]:.8f} {val[1]:.8f} {val[2]:.8f}"
                            f" # {fragment.arrays['fragment_id'][i]}"
                            f" % NUM={fragment_numbers[i]}"
                        )
                    else:
                        newline = (
                            f"{symbols[i]} {val[0]:.8f} {val[1]:.8f} {val[2]:.8f}"
                            f" # {fragment.arrays['fragment_id'][i]}"
                        )
                    fragment_parameters.append(newline)
                fragment_parameters.append("%ENDBLOCK POSITIONS_ABS")

                buildcell_parameters = (
                    fragment_parameters + buildcell_parameters
                )  # prepend with structural info

            self._cell_seed(buildcell_parameters, tag)
            bc_file = f"{tag}.cell"

        with Pool(processes=num_processes) as pool:
            args = [
                (i, bc_file, tag, remove_tmp_files)
                for i in range(struct_number)
            ]
            atoms_group = pool.starmap(self._parallel_process, args)

        atoms_group = [
            atom for atom in atoms_group if not np.isnan(atom.get_positions()).any()
        ]

        ase.io.write(
            output_file_name, atoms_group, parallel=False, format="extxyz"
        )

        #Return path of the ase-file with generated structures
        return os.path.join(Path.cwd(), output_file_name)

    def _update_buildcell_option(self, updates, origin) -> list:
        """
        Update buildcell parameters based on a dictionary of updates.

        Parameters
        ----------
        updates: dict
            A dictionary consisting of new values to update buildcell parameters.
        origin: list
            The default list of buildcell parameters.
        """
        updated_keys = set()

        for i, option in enumerate(origin):
            option_key = option.split("=")[0]
            if option_key in updates:
                origin[i] = f"{option_key}={updates[option_key]}"
                updated_keys.add(option_key)

        for key, value in updates.items():
            if key not in updated_keys:
                origin.append(f"{key}={value}")

        return origin

    def _cell_seed(
        self,
        buildcell_parameters: list,
        tag: str,
    ):
        """
        Prepare the seed file for buildcell.

        Parameters
        ----------
        buildcell_parameters: (list of str) e.g. ['VARVOL=20']
            List of parameters for creating the seed file for buildcell.
        tag: str
            Tag of systems.
        """
        bc_file = f"{tag}.cell"
        contents = []
        flag = False  # for printing blocks correctly with '#'
        for i in buildcell_parameters:
            if i.startswith("%") or flag:
                flag = not (flag and i.startswith("%"))
                contents.append(i + "\n")
            else:
                contents.append("#" + i + "\n")

        with open(bc_file, "w") as f:
            f.writelines(contents)

    def _extract_elements(self, input_str: str) -> dict[str, int]:
        """
        Extract elements and their counts from a chemical formula string.

        Parameters
        ----------
        input_str: str
            A string representing a chemical formula (e.g., "SiO2").

        Returns
        -------
        Dict[str, int]
            A dictionary. For example, the input "SiO2" would return {"Si": 1, "O": 2}.
        """
        elements: dict[str, int] = {}
        pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
        matches = pattern.findall(input_str)

        for match in matches:
            element, count = match
            count = int(count) if count else 1
            if element in elements:
                elements[element] += count
            else:
                elements[element] = count

        return elements

    def _make_species(self, elements: dict[str, int]) -> str:
        """
        Create a formatted string from a dictionary of element symbols and their counts.

        Parameters
        ----------
        elements: dict
            A dictionary of element symbols and their counts, e.g., {"Si": 1, "O": 2}.

        Returns
        -------
        str
            A formatter string. For example, the input {"Si": 1, "O": 2} would return "Si%NUM=1,O%NUM=2".
        """
        output = ""
        for element, count in elements.items():
            output += f"{element}%NUM={count},"
        return output[:-1]

    def _make_minsep(self, r: dict[str, float]) -> str:
        """
        Generate a minsep string based on the radii of the elements.

        Parameters
        ----------
        r: dict
            A dictionary of element symbols and their atomic radii.

        Returns
        -------
        str
            A formatted string. For example, the input {"Si": 1.1, "O": 0.66} would
            return "1.5 Si-Si=1.76 Si-O=1.408 O-O=1.056".

        TODO: set up robust heuristics for multi-component systems
        """
        keys = list(r.keys())
        if len(keys) == 1:
            return str(1.6 * r[keys[0]])

        minsep = "1.5 "
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                el1, el2 = keys[i], keys[j]
                r1, r2 = r[el1], r[el2]
                result = (r1 + r2) / 2 * 1.6

                minsep += f"{el1}-{el2}={result} "

        return minsep[:-1]

    def _parallel_process(
        self, i: int, bc_file: str, tag: str, remove_tmp_files: bool
    ) -> Atoms:
        """
        Run the 'buildcell' command in parallel.

        Parameters
        ----------
        i: int
            Unique index to differentiate temporary files.
        bc_file: str
            Path to the input 'buildcell' file.
        tag: str
            Tag used to differentiate temporary files.
        remove_tmp_files: bool
            If True, remove temporary files after processing.

        """
        tmp_file_name = "tmp." + str(i) + "." + tag + ".cell"
        tmp_error_file_name = "tmp_error." + str(i) + "." + tag + ".cell"

        with (
            open(bc_file) as bc_file_handle,
            open(tmp_file_name, "w") as tmp_file_handle,
            open(tmp_error_file_name, "w") as tmp_error_file_handle,
        ):
            run(
                "buildcell",
                stdin=bc_file_handle,
                stdout=tmp_file_handle,
                stderr=tmp_error_file_handle,
                shell=True,
                check=True,
            )

        atom = ase.io.read(tmp_file_name, parallel=False)
        atom.info["unique_starting_index"] = i

        if "castep_labels" in atom.arrays:
            del atom.arrays["castep_labels"]

        if "initial_magmoms" in atom.arrays:
            del atom.arrays["initial_magmoms"]

        if remove_tmp_files:
            os.remove(tmp_file_name)
            os.remove(tmp_error_file_name)

        return atom