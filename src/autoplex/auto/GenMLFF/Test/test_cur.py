import yaml
import argparse
from autoplex.auto.GenMLFF.sampling import SamplingMaker

#Define the path to the configuration file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration file",
)

#Read configuration file
args = parser.parse_args()
with open(args.config, "r") as f:
    sampling_params = yaml.safe_load(f)

#Instantiate the SamplingMaker class
sampling_maker = SamplingMaker(**sampling_params)
#Run the sampling process
sampled_structures_path = sampling_maker.make()
print(f"Sampled structures saved to: {sampled_structures_path}")