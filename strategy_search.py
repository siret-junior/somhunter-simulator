
import numpy as np
import os
import multiprocessing as mp
import time
from datetime import datetime
import pickle
from dataclasses import dataclass

import argparse

from bayesian_ranker import BayesianRanker

from bow_initializer import BoWInitializer

from ransam_display import RanSamDisplay
from topn_display import TopNDisplay
from som_display import SOMDisplay 

from ransam_prior_user import RanSamPriorUser

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--processes", default=-1, type=int, help="Number of precesses spawned.")


parser.add_argument("--annotations", default="data/annotations.csv", type=str,
                    help="Annotations to be simulated.")

parser.add_argument("--dataset_path", default="v3c1", type=str,
                    help="Root to dataset path.")
parser.add_argument("--features_name", default="V3C1_20191228.w2vv.images.normed.128pca.viretfromat", type=str,
                    help="Name of file with image features.")

parser.add_argument("--keywords_list_name", default="word2idx.txt", type=str,
                    help="Name of file with keyword features.")
parser.add_argument("--kw_features_name", default="txt_weight-11147x2048floats.bin", type=str,
                    help="Name of file with keyword features.")
parser.add_argument("--kw_bias_name", default="txt_bias-2048floats.bin", type=str,
                    help="Name of file with keyword bias.")
parser.add_argument("--pca_matrix_name", default="V3C1_20191228.w2vv.pca.matrix.bin", type=str,
                    help="Name of file with pca matrix.")
parser.add_argument("--pca_mean_name", default="V3C1_20191228.w2vv.pca.mean.bin", type=str,
                    help="Name of file with pca mean.")

parser.add_argument("--pickle_root", default="pickle", type=str,
                    help="Root of pickle models.")
parser.add_argument("--pickle_model", default="pcu.prior.pickle", type=str,
                    help="Name of pickled user model.")

parser.add_argument("--verbose", default=False, action="store_true", help="Verbose")


@dataclass
class SimParameters:
    likes: int
    display_types: list
    database_part: float
    text_query: str
    target_id: int

class Simulator(mp.Process):

    def __init__(self, sim_args, par_q: mp.Queue, res_q: mp.Queue, **wargs):
        super().__init__(**wargs)
        np.random.seed(args.seed)
        self._par_q = par_q
        self._res_q = res_q

        features = np.fromfile(os.path.join(sim_args.dataset_path, sim_args.features_name), dtype='float32')
        features = features[3:]
        features = features.reshape(int(features.shape[0] / 128), 128)
        self._features = features
        self._kw_init = BoWInitializer(features, 
            os.path.join(sim_args.dataset_path, sim_args.keywords_list_name), 
            os.path.join(sim_args.dataset_path, sim_args.kw_features_name),
            os.path.join(sim_args.dataset_path, sim_args.kw_bias_name),
            os.path.join(sim_args.dataset_path, sim_args.pca_matrix_name),
            os.path.join(sim_args.dataset_path, sim_args.pca_mean_name)
            )
        with open(os.path.join(sim_args.pickle_root, sim_args.pickle_model), 'rb') as handle:
            self._user = pickle.load(handle)
        self._user._features = features
        self._ranker = BayesianRanker(features, features.shape[0])

        self._displays = {"som": SOMDisplay(self._features, seed=sim_args.seed), "top": TopNDisplay()}

    def run(self):
        while True:
            par = self._par_q.get()
            if par is None:
                break
            
            # Parse simulation parameters
            likes = par.likes
            display_types = par.display_types
            database_part = par.database_part
            text_query = par.text_query
            target_id = par.target_id

            # Make some assumtions on parameters
            assert likes > 0
            assert likes < 64
            assert database_part is None or (database_part <= 1.0 and database_part > 0.0) 
            assert isinstance(target_id, int)

            # Initialize search structures
            self._user._count = likes
            self._user._target = target_id

            self._ranker.reset()
            self._ranker._scores = self._kw_init.score(text_query)
            self._ranker.normalize()

            # Run simulations
            found = -1
            for iteration, disp_type in enumerate(display_types):
                display = self._displays[disp_type].generate(self._ranker.scores)

                if target_id in display:
                    found = iteration
                    break

                likes = self._user.decision(display)
                self._ranker.apply_feedback(likes, display)
                
            # Return result
            par.found = found
            self._res_q.put(par)
            


def main(args):
    np.random.seed(args.seed)
    processes = args.processes
    if processes <= 0:
        processes = mp.cpu_count()
    
    par_q = mp.Queue()
    res_q = mp.Queue()
    jobs = []
    for i in range(processes):
        sim = Simulator(args, par_q, res_q, name=f"Simulator {i}")
        jobs.append(sim)
        sim.start()
    
    # Add parameters
    like_counts = range(1, 5)
    display_types = [["som" for _ in range(10)], 
                    ["top" for _ in range(10)],
                    ["som" for _ in range(5)] + ["top" for _ in range(5)],
                    [("som" if i % 2 == 0 else "top") for i in range(10)],
                    [("som" if i % 2 == 1 else "top") for i in range(10)]]
    targets = []
    text_queries = []
    with open(args.annotations, "r") as f:
        for line in f.readlines():
            target_id, text_query = line.strip().split(",")
            targets.append(int(target_id))
            text_queries.append(text_query)

    reps = 0
    for lik in like_counts:
        for tar, text_query in zip(targets, text_queries):
            for disp_type in display_types:
                par_q.put(SimParameters(lik, disp_type, None, text_query, tar))
                reps += 1

    # Add poison pill
    for i in range(processes):
        par_q.put(None)

    # Collect results
    start = datetime.now()
    print("Simulations started\n")
    res = []
    with open(f"data/strategy_search_output.{int(time.time())}.csv", "w") as of:
        for i in range(reps):
            last_res = res_q.get()
            res.append(last_res)
            delta = datetime.now() - start
            per_instance = delta / len(res)
            left = (reps - len(res)) * per_instance
            print(f"Done: {len(res)}/{reps}\tTime elapsed: {delta}\tTime left: {left}\t\t\t", end="\n", flush=True)
            of.write(f"{last_res.likes},{last_res.display_types},{last_res.database_part},{last_res.text_query},{last_res.target_id},{last_res.found}\n")
            of.flush()

    print("\n********************")
    print(res, flush=True)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
