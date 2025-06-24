import os
import argparse
from itertools import combinations_with_replacement
from ase import Atoms
from ase.io import write

def prepare_dimers(
    atom_pair: tuple[str, str],
    dimer_distances: list[float] = [1.0, 2.0, 3.0, 4.0],
    void: float = 10.0,
) -> list[Atoms]:
    """Prepare a list of shifted dimers from a pair of atomic species.

    Parameters
    ----------
    atom_pair : tuple[str, str]
        Pair of atomic species to create dimers from, e.g. ("H", "O").
    dimer_distances : list[float], optional
        List of distances between the two atoms in the dimer, by default [1.0, 2.0, 3.0, 4.0]
    void : float, optional
        Size of the void around the dimer to avoid interactions, by default 10.0

    Returns
    -------
    list[Atoms]
        List of ASE Atoms objects representing the dimers.
    """
    
    # Prepare shifted dimers
    dimers = []
    for distance in dimer_distances:
        # Create dimer with specified distance
        dimer = Atoms(
            [atom_pair[0], atom_pair[1]],
            positions=[(0, 0, 0), (distance, 0, 0)],
            cell=[distance + void, void, void],  # Large enough cell to avoid interactions
            pbc=True,  # Periodic boundary conditions
        )
        # Add dimer to list
        dimers.append(dimer)
    
    return dimers

def prepare_isolated_atoms(
    atom: str,
    void: float = 10.0,
) -> Atoms:
    """Prepare an isolated atom with a large void around it.

    Parameters
    ----------
    atom : str
        Atomic species to create the isolated atom from, e.g. "H".
    void : float, optional
        Size of the void around the atom to avoid interactions, by default 10.0

    Returns
    -------
    Atoms
        ASE Atoms object representing the isolated atom.
    """
    
    # Create isolated atom with large void
    isolated_atom = Atoms(
        [atom],
        positions=[(0, 0, 0)],
        cell=[void, void, void],  # Large enough cell to avoid interactions
        pbc=True,  # Periodic boundary conditions
    )
    
    return isolated_atom

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare dimers and isolated atoms to be computed for the initial dataset.")
    parser.add_argument(
        "atom_species",
        type=str,
        nargs="+",
        help="Atomic species to create dimers from, e.g. 'H O'.",
    )
    parser.add_argument(
        "--dimer_distances",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 4.0],
        help="List of distances between the two atoms in the dimer.",
    )
    parser.add_argument(
        "--void",
        type=float,
        default=10.0,
        help="Size of the void around the dimer to avoid interactions.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="dimers_and_isolated_atoms",
        help="Output directory to save the prepared structures.",
    )    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    #Create output directory if it does not exist
    os.makedirs(args.outdir, exist_ok=True)

    # Prepare dimers from all combinations of atomic species
    num_dimer_structures = 0
    for atom_pair in combinations_with_replacement(args.atom_species, 2):
        #Get dimer structures for the current pair of atoms
        dimers = prepare_dimers(atom_pair, args.dimer_distances, args.void)
        num_dimer_structures += len(dimers)

        #Save dimer structures to the extxyz file
        dimer_filename = os.path.join(args.outdir, f"dimer_{atom_pair[0]}_{atom_pair[1]}.extxyz")
        write(dimer_filename, dimers, format='extxyz')
    
    # Prepare isolated atoms
    num_isolated_structures = len(args.atom_species)
    for atom in args.atom_species:
        #Get isolated atom structure
        isolated_atom = prepare_isolated_atoms(atom, args.void)

        #Save isolated atom structure to the extxyz file
        isolated_filename = os.path.join(args.outdir, f"isolated_{atom}.extxyz")
        write(isolated_filename, isolated_atom, format='extxyz')

    # Print the number of structures prepared
    print(f"{num_dimer_structures} structures with dimers combinations saved in {args.outdir}/dimer_*.extxyz.")
    print(f"{num_isolated_structures} structures with isolated atoms saved in {args.outdir}/isolated_*.extxyz.")