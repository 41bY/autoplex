import argparse
from pandas import DataFrame
from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor


def dataset2csv(structures: list[Atoms], properties: list[str]) -> DataFrame:
    """Convert a dataset of ASE atoms structures to a CSV file with material_id and CIF format.
    Parameters
    ----------
    structures : list[Atoms]
        List of ASE Atoms objects representing the structures.
    properties : list[str]
        List of property names to extract from the atoms.info dictionary.
    Returns
    -------
    DataFrame
        A pandas DataFrame containing the (index), material_id, CIF string, and specified properties."""

    #Convert structures to DataFrame with material_id and CIF
    csv_rows = []
    for i, atoms in enumerate(structures):
        #Define csv row
        csv_row = {"material_id" : None, "cif": None, **{prop: None for prop in properties}}

        #Get material_id
        mid = f"user-{i}"
        csv_row["material_id"] = mid
        
        #Get cif from pymatgen structure
        cif_str = AseAtomsAdaptor.get_structure(atoms).to(fmt="cif")
        csv_row["cif"] = cif_str

        #Extract properties if they exist
        for prop in properties:
            #Check if ASE default property
            if prop in ["energy", "forces", "stress", "magmom"]:
                #Get ASE default property
                value = atoms.calc.results.get(prop, None)

                #Check if value is array-like
                if isinstance(value, (list, tuple)):
                    #Convert to string representation
                    value = ', '.join(map(str, value))
                elif isinstance(value, (int, float)):
                    #Convert to string representation
                    value = str(value)
                
                #Set value in csv_row
                csv_row[prop] = value

            elif prop in atoms.info:
                #Get value-like property from atoms.info
                value = atoms.info[prop]
                csv_row[prop] = value
        
        #Assemble row for csv DataFrame
        csv_rows.append(csv_row)

    #Create DataFrame
    df = DataFrame(csv_rows)    

    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="""Convert ASE-readable files containing structures and properties to CSV file for MatterGen.
                                     The CSV file will contain material_id, CIF format, and specified properties.
                                     Array-like properties will be converted to string representation but, at the moment, cannot be used by MatterGen.""")
    parser.add_argument("ASE_readable_path", type=str, help="Input file containing ASE structures.")
    parser.add_argument("output_file", type=str, help="Output path of CSV file.")
    parser.add_argument("--properties", nargs="+", default=["energy"], 
                        help="List of properties to extract from the structures. If some structure does not have a property, it will be set to None in the CSV file. Default is ['energy'].")
    return parser.parse_args()

if __name__ == "__main__":
    #Parse command line arguments
    args = parse_args()

    #Read structures from ASE-readable file
    structures = read(args.ASE_readable_path, index=":")

    #Convert dataset to CSV DataFrame
    df = dataset2csv(structures, args.properties)

    #Save DataFrame to CSV file
    df.to_csv(args.output_file, index=True)