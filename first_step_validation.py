import numpy as np
import pandas as pd

import argparse

from topn_display import TopNDisplay
from som_display import SOMDisplay
from bayesian_ranker import BayesianRanker
from ideal_user import IdealUser
from null_user import NullUser
from ransam_display import RanSamDisplay
from ransam_user import RanSamUser
from ransam_smooth_user import RanSamSmoothUser
from bow_initializer import BoWInitializer

import statsmodels
import statsmodels.formula.api as smf


parser = argparse.ArgumentParser()

parser.add_argument("--model", default=0, type=int, help="Index of the model (0-6).")

def main(args):
    features = np.fromfile("./v3c1/V3C1_20191228.w2vv.images.normed.128pca.viretfromat", dtype='float32')
    features = features[3:]
    features = features.reshape(int(features.shape[0] / 128), 128)
    print("Features loaded", features.shape)

    data = pd.read_csv("./data/result_collection.csv")
    data = data[~data['target_id'].isin([1088886,171357,0])]
    data = data.sort_values(["user", "timestamp"])
    print("Data loaded")

    kw_initializer = BoWInitializer(features, 
                "./v3c1/word2idx.txt", 
                "./v3c1/txt_weight-11147x2048floats.bin", 
                "./v3c1/txt_bias-2048floats.bin",
                "./v3c1/V3C1_20191228.w2vv.pca.matrix.bin",
                "./v3c1/V3C1_20191228.w2vv.pca.mean.bin")

    ranker = BayesianRanker(features, features.shape[0])

    def likes_real(row):
        display = row.filter(regex="D.*_id$").to_numpy(dtype=int)
        selection_mask = row.filter(regex="D.*_is_selected$").to_numpy(dtype=bool)
        return display[selection_mask]

    def likes_pcu(row):
        count = np.sum(row.filter(regex="D.*_is_selected$").to_numpy())
        user = RanSamUser(features, int(row["target_id"]), 13, count)
        display = row.filter(regex="D.*_id$")
        return user.decision(display.to_numpy(dtype=int))

    def likes_null(row):
        count = np.sum(row.filter(regex="D.*_is_selected$").to_numpy())
        user = NullUser(count)
        display = row.filter(regex="D.*_id$")
        return user.decision(display.to_numpy(dtype=int))

    def likes_ideal(row):
        count = np.sum(row.filter(regex="D.*_is_selected$").to_numpy())
        user = IdealUser(features, int(row["target_id"]), count)
        display = row.filter(regex="D.*_id$")
        return user.decision(display.to_numpy(dtype=int))

    def create_predict_model(model):
        def predict_model(row):
            ids = row.filter(regex="D.*_id").to_list()
            target = int(row['target_id'])
            count = np.sum(row.filter(regex="D.*_is_selected$").to_numpy())

            dists = row.filter(regex="D[0-9]*_distance_to_target").to_numpy()
            dranks = np.sum(np.reshape(dists, (-1, 1)) > np.reshape(dists, (1, -1)), axis=-1)
            first = np.array([1] + 63 * [0])
            border = np.ones([8,8], dtype=bool)
            border[1:7,1:7] = False
            border = border.reshape(-1)
            row1 = [max((8-pos)/8,0) for pos in range(64)]

            vals={'D_distance_to_target': dists.tolist(),
                'D_rank': dranks.tolist(),
                'first': first.tolist(),
                'border': border.tolist(),
                'row1': row1}
            curr=pd.DataFrame(vals)

            probabs = model.predict(curr)
            probabs /= np.sum(probabs)
            return display[np.random.choice(probabs.shape[0], count, p=probabs, replace=False)]
        
        return predict_model

    models = [("real", likes_real), ("pcu", likes_pcu), ("null", likes_null), ("ideal", likes_ideal)]
    models += [(model_type, create_predict_model(statsmodels.iolib.smpickle.load_pickle(f"pickle/smf.all.{model_type}.pickle")) )
            for model_type in ['full', 'rank0', 'distance'] ]

    test_model = models[args.model]
    model_name, get_likes = test_model
    print("Testing model", model_name, flush=True)

    prev_user = None
    prev_target = None
    prev_text = None

    disp_gen_top = TopNDisplay()
    disp_gen_som = SOMDisplay(features)

    results = []

    __counter = 0

    for _, row in data.iterrows():
        
        # init condition
        if row["type"] == "text":
            prev_user = row["user"]
            prev_target = row["target_id"]
            prev_text = row["text_query"]
        elif (row["type"] == "feedback" and # simulation condition
            prev_text == row["text_query"] and 
            prev_user == row["user"] and
            prev_target == row["target_id"]
            ):
            display = row.filter(regex="D.*_id$")
            scores = kw_initializer.score(prev_text)
            ranker._scores = scores
            ranker.normalize()
            scores = ranker._scores.copy()
            
            # Generate displays
            if row["display_type"] == "top":
                disp_gen = disp_gen_top
            else: #  SOM
                disp_gen = disp_gen_som
                
            # apply feedback for all models
            foundings = 0
            likes = get_likes(row)

            ranker._scores = scores.copy()
            ranker.apply_feedback(likes, display.to_list())
            disp = disp_gen.generate(scores)
            if int(row["target_id"]) in disp:
                foundings = 1
            results.append(foundings)
                
            
            prev_text = None
        __counter += 1
        if __counter % 10 == 0:
            print(__counter, flush=True)
    else:
        print(__counter, flush=True)

    results_df = pd.DataFrame({model_name: results})
    results_df.to_csv(f"./data/first_step_validation_{model_name}.csv")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
