from Random import seed
import os, sys, subprocess
import Constants

INPUT_FOLDER = "input"
SCRIPT_NAME = "Main.py"
SEEDS = list(range(30))
INPUT_CONFIG_PREFIX = "hyp9"

def main():
    for s in SEEDS:
        print(f"Testing input configurations with seed: {s}")
        seed(s) # set seed from Random.py

        for filename in os.listdir(INPUT_FOLDER):
            if filename.startswith(INPUT_CONFIG_PREFIX) and filename.endswith(".json"):
                print(f"Processing: {filename}", end=' ')
                
                output_name = filename.split('.')[0] + "s-" + str(s) + "." + filename.split('.')[1]
                result = subprocess.run(
                    [sys.executable, SCRIPT_NAME, filename, output_name],
                    capture_output=True,
                    text=True
                )

                if result.stderr:
                    print(f"Error on processing {filename}: ", result.stderr)
                    break
                else:
                    print(f"Simulation data saved in output/{output_name}.json")

if __name__ == "__main__":
    main()