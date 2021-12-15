
import numpy as np
import os

import argparse

from rankers import BayesianRanker

from initializers.bow_initializer import BoWInitializer
from users import LogitUser
from users import NullUser

from displays.ransam_display import RanSamDisplay
from displays import TopNDisplay
from displays import SOMDisplay 

from users import IdealUser
from users import RanSamUser
from users import RanSamSmoothUser

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--quiet", default=False, action="store_true", help="Quiet")

parser.add_argument("-t", "--target_file", required=False, default="data/study_targets.csv", type=str, 
                    help="File with defined targets. On each line should be 'target_id;text_query;iterations;displayType;likes'")
parser.add_argument("-o", "--output_file", required=False, default="data/study_targets.csv.out", type=str,
                    help="Name of output csv file.")

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

parser.add_argument("--user", default="pcu", type=str,
                    help="Name of user.")
parser.add_argument("--pickle_root", default="pickle", type=str,
                    help="Root of pickle models.")

def load_features(args):
    features = np.fromfile(os.path.join(args.dataset_path, args.features_name), dtype='float32')
    features = features[3:]
    features = features.reshape(int(features.shape[0] / 128), 128)
    return features

def quiet_log(*args, **kwargs):
    pass

def main(args):
    np.random.seed(args.seed)

    log = print
    if args.quiet:
        log = quiet_log

    features = load_features(args)
    kw_init = BoWInitializer(features, 
        os.path.join(args.dataset_path, args.keywords_list_name), 
        os.path.join(args.dataset_path, args.kw_features_name),
        os.path.join(args.dataset_path, args.kw_bias_name),
        os.path.join(args.dataset_path, args.pca_matrix_name),
        os.path.join(args.dataset_path, args.pca_mean_name)
        )
    with open(args.output_file, "w") as of:
        with open(args.target_file, "r") as f:
            content = f.readlines()
            counter = 0
            content_len = str(len(content))
            for line in content:
                line = line.strip()
                counter += 1
                log(str(counter) + "/" + content_len, line, end="")
                tokens = line.split(";")

                # Parse arguments
                target_id = int(tokens[0])
                text_query = tokens[1].strip()
                iterations = int(tokens[2])
                display_type = tokens[3].lower()
                num_of_likes = list(map(lambda x: int(x), filter(lambda x: len(x) > 0, tokens[4].split(","))))

                # Prepare search structures
                display_gen = TopNDisplay()
                if display_type == "som":
                    display_gen = SOMDisplay(features)

                ranker = BayesianRanker(features, features.shape[0])
                if text_query:
                    ranker._scores = kw_init.score(text_query)

                user_type = args.user.lower()
                if user_type == "pcu":
                    user = RanSamUser(features, target_id, 13)
                elif user_type == "ideal":
                    user = IdealUser(features, target_id)
                elif user_type == "null":
                    user = NullUser()
                elif user_type == "logit":
                    user = LogitUser(features, target_id, pickle_file = os.path.join(args.pickle_root, "smf.all.full.pickle"))
                

                # Generate first display
                found = -1
                
                for iteration in range(iterations + 1):
                    display = display_gen.generate(ranker.scores)
                    # Check if found
                    if target_id in display:
                        found = iteration
                        break
                    
                    # Skip feedback on last iteration
                    if iteration < iterations:
                        user._count = num_of_likes[iteration]
                        likes = user.decision(display)
                        ranker.apply_feedback(likes, display)
                        log(".", end="", flush=True)
                    

                of.write(line + ";" + str(found) + "\n")
                of.flush()
                log("DONE", flush=True)
        
    log("Simulations done")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
