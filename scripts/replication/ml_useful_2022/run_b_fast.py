import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import run_b
run_b.main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ml_useful_data/2018-01.csv",
           horizons=(1,), n_origins=16)
