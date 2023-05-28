"""
Copyright 2023  Mathieu Chevalley, Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Tuple

import os
import numpy as np
import pandas as pd
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import remove_lowly_expressed_genes
import lightgbm as lgb
from tqdm import tqdm


def get_topK_pairs(expression_matrix: pd.DataFrame, T: float = 0.5):
    corrs = []
    for gene1 in tqdm(expression_matrix.columns):  # from
        for gene2 in expression_matrix.columns:  # target

            if gene1 == gene2:
                continue

            if gene1 in expression_matrix.index:
                exp_obs_gene1 = expression_matrix.loc["non-targeting", gene1]
                exp_inv_gene1 = expression_matrix.loc[gene1, gene1]
                exp_obs_gene2 = expression_matrix.loc["non-targeting", gene2]
                exp_inv_gene2 = expression_matrix.loc[gene1, gene2]

                exp_gene1 = pd.concat([exp_obs_gene1.sample(exp_inv_gene1.shape[0], random_state=0), exp_inv_gene1])
                exp_gene2 = pd.concat([exp_obs_gene2.sample(exp_inv_gene2.shape[0], random_state=0), exp_inv_gene2])
            else:
                exp_gene1 = expression_matrix.loc["non-targeting", gene1]
                exp_gene2 = expression_matrix.loc["non-targeting", gene2]
            corrs.append([gene1, gene2, np.abs(exp_gene1.corr(exp_gene2))])

    corrs = pd.DataFrame(corrs, columns=["From", "To", "weights"])\
        .sort_values(by="weights", ascending=False)
    topK_pairs = [(i, j) for i, j in corrs.loc[corrs["weights"] > T, ["From", "To"]].values]

    return corrs, topK_pairs

def create_dataset(expression_matrix, pairs):
    dataset = []
    gene_names = expression_matrix.columns
    expression_summary = expression_matrix.reset_index().groupby("index").mean()

    for gene1 in gene_names:
        for gene2 in gene_names:

            if gene1 == gene2:
                continue

            index = gene1 + "_" + gene2
            data = [index] + [expression_summary.loc["non-targeting", gene1],
                              expression_summary.loc["non-targeting", gene2],
                              expression_summary.loc[gene1, gene1] if gene1 in expression_summary.index else 0,
                              expression_summary.loc[gene1, gene2] if gene1 in expression_summary.index else np.nan]
            if (gene1, gene2) in pairs:
                data += [1]
            else:
                data += [0]

            dataset.append(data)

    dataset = pd.DataFrame(dataset).set_index(0)
    colnames = dataset.columns.to_list()
    colnames[-1] = "label"
    dataset.columns = colnames
    return dataset

def train_lgb(dataset: pd.DataFrame, lgb_params: dict):
    X = dataset.drop(columns="label")
    y = dataset["label"]
    lgb_data = lgb.Dataset(X, y)

    gbm = lgb.train(params=lgb_params, train_set=lgb_data, keep_training_booster=True)
    return gbm


class Custom(AbstractInferenceModel):
    def __init__(self):
        super().__init__()
        self.gene_expression_threshold = 0.25

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ):
        """
            expression_matrix: numpy array of size n_samples x n_genes, which contains the expression values
                                of each gene in different cells
            interventions: list of size n_samples. Indicates which gene has been perturbed in each sample.
                            If value is "non-targeting", no gene was targeted (observational sample).
                            If value is "excluded", a gene was perturbed which is not in gene_names (a confounder was perturbed).
                            You may want to exclude those samples or still try to leverage them.
            gene_names: names of the genes of size n_genes. To be used as node names for the output graph.


        Returns:
            List of string tuples: output graph as list of edges.
        """
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 5,
            'max_depth': 2,
            'min_data_in_leaf': 5,
            'learning_rate': 0.05,
            'min_gain_to_split': 0.01,
            'num_iterations': 1000,
            'num_threads': 8,
            'verbose': 0,
        }
        T = 0.1
        N = 1000

        expression_matrix = pd.DataFrame(expression_matrix, index=interventions, columns=gene_names)

        all_corrs, topK_pairs = get_topK_pairs(expression_matrix, T)

        the_mean = np.mean(expression_matrix, axis=0)
        the_std = np.std(expression_matrix, axis=0)
        expression_matrix = (expression_matrix - the_mean) / the_std

        dataset = create_dataset(expression_matrix, topK_pairs)
        train_data = dataset.sample(frac=1, random_state=0)

        gbm = train_lgb(train_data, params)
        predictions = gbm.predict(dataset.drop(columns="label"))
        network = dataset.iloc[np.argsort(predictions)[::-1][:N]].index

        # You may want to postprocess the output network to select the edges with stronger expected causal effects.
        return [tuple(pair.split("_")) for pair in network]
