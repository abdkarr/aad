import numpy as np

def calc_agreement_mat(responses: np.ndarray):
    n_annotators = responses.shape[0]

    agreement_mat = np.zeros((n_annotators, n_annotators))
    n_co_observed = np.zeros((n_annotators, n_annotators))

    observed_tasks = []
    for i in range(n_annotators):
        observed_tasks.append(np.where(responses[i, :] > 0)[0])

    for i in range(n_annotators):
        for j in range(i+1, n_annotators):
            common_tasks = np.intersect1d(
                observed_tasks[i], observed_tasks[j], assume_unique=True
            )
            
            n_co_observed[i, j] = len(common_tasks)
            n_co_observed[j, i] = n_co_observed[i, j]

            responses_i = responses[i, common_tasks]
            responses_j = responses[j, common_tasks]
            agreement_mat[i, j] = np.mean(responses_i == responses_j)
            agreement_mat[j, i] = agreement_mat[i, j]

    return agreement_mat, n_co_observed