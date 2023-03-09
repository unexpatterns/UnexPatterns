"""Mine unexpected patterns."""
import os
import sys

from src.algorithm import run_unex_patterns
from src.compressors import generation_complexity
from src.data import load_data, read_parameters


# ******************************************************** #
# Mine unexpected patterns
# ******************************************************** #

if __name__ == '__main__':

    # Get parameters
    FILENAME = sys.argv[1]
    parameters = read_parameters(FILENAME)
    OUTPATH = os.path.join(os.getcwd(), parameters.get('patterns_path'))

    for dataset in parameters.get('datasets'):
        print(f"Dataset: {dataset}\n{'='*(len(dataset)+10)}")

        # Load netset data
        adjacency, biadjacency, names, names_col, labels = load_data(dataset)

        # Compute generation complexities
        complexity_gen_graphs = generation_complexity(adjacency, biadjacency,
                                                      n_attrs=15, n_iter=300)

        for s in parameters.get('s'):
            print(f"Parameter s={s}\n{'-'*(len(str(s))+12)}")

            # Run algorithm
            outfilename = f"{dataset}_{str(parameters.get('beta'))}_{str(s)}_{parameters.get('delta')}"
            nb_patterns = run_unex_patterns(adjacency=adjacency,
                                            biadjacency=biadjacency,
                                            names_col=names_col,
                                            complexity_gen_graphs=complexity_gen_graphs,
                                            order_attributes=True,
                                            s=s,
                                            beta=parameters.get('beta'),
                                            delta=parameters.get('delta'),
                                            without_constraints=False,
                                            outfile=outfilename,
                                            outpath=OUTPATH)
