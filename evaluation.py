###############################################################################
###############################################################################
#
#  EVALUATION OF RECOMMENDER SYSTEMS
#
metrics = ["hr", "prec", "recall", "map", "mrr", "f0.5", "f1", "f2"]
tops_n = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100]
eval_metrics = [
    "hr@1", "hr@2", "hr@3", "hr@5", "hr@10", "hr@15", "hr@20", "hr@30", "hr@40", "hr@50", "hr@100",
    "prec@1", "prec@2", "prec@3", "prec@5", "prec@10", "prec@15", "prec@20", "prec@30", "prec@40", "prec@50",
    "prec@100",
    "recall@1", "recall@2", "recall@3", "recall@5", "recall@10", "recall@15", "recall@20", "recall@30", "recall@40",
    "recall@50", "recall@100",
    "map@1", "map@2", "map@3", "map@5", "map@10", "map@15", "map@20", "map@30", "map@40", "map@50", "map@100",
    "mrr@1", "mrr@2", "mrr@3", "mrr@5", "mrr@10", "mrr@15", "mrr@20", "mrr@30", "mrr@40", "mrr@50", "mrr@100",
    "f0.5@1", "f0.5@2", "f0.5@3", "f0.5@5", "f0.5@10", "f0.5@15", "f0.5@20", "f0.5@30", "f0.5@40", "f0.5@50",
    "f0.5@100",
    "f1@1", "f1@2", "f1@3", "f1@5", "f1@10", "f1@15", "f1@20", "f1@30", "f1@40", "f1@50", "f1@100",
    "f2@1", "f2@2", "f2@3", "f2@5", "f2@10", "f2@15", "f2@20", "f2@30", "f2@40", "f2@50", "f2@100"
]

class Evaluation(object):

    @staticmethod
    def evaluation_metric_list():
        return eval_metrics

    def __init__(self, links_to_rec, rec_links):

        # if len(rec_links) > 0:
        #    print "rec_links",rec_links[rec_links.keys()[0]]
        #    print "links_to_rec",links_to_rec[links_to_rec.keys()[0]]
        if len(links_to_rec) < 0: raise ValueError('Evaluation : links_to_rec bad value!')
        if len(rec_links) < 0: raise ValueError('Evaluation : rec_links bad value!')
        self.links_to_rec = links_to_rec
        self.rec_links = rec_links
        self.rec_links_binary = {}
        self.result_values = {}
        self.result_weights = {}
        self.evaluation_metrics = eval_metrics

    def get_evaluation_metrics(self):
        return self.evaluation_metrics

    def get_result_values(self):
        return self.result_values

    def get_result_weights(self):
        return self.result_weights

    def compute_evaluation_results(self):
        # transform all recommendation list to binary list
        for u in self.links_to_rec.keys():
            self.rec_links_binary[u] = []
            # print u,"-------"
            # print list(self.links_to_rec[u])[:5]
            # print list(self.rec_links[u])[:5]
            for i in self.rec_links[u]:
                if i in self.links_to_rec[u]:
                    self.rec_links_binary[u].append(1)
                else:
                    self.rec_links_binary[u].append(0)

        # return all evaluation
        eval_results, eval_result_weights = {}, {}
        eval_results["hr"], eval_result_weights["hr"] = self._get_hit_ratio()
        eval_results["prec"], eval_result_weights["prec"] = self._get_precision()
        eval_results["recall"], eval_result_weights["recall"] = self._get_recall()
        eval_results["map"], eval_result_weights["map"] = self._get_map()
        eval_results["mrr"], eval_result_weights["mrr"] = self._get_mrr()
        eval_results["f0.5"], eval_result_weights["f0.5"] = self._get_fmeasure(0.5)
        eval_results["f1"], eval_result_weights["f1"] = self._get_fmeasure(1)
        eval_results["f2"], eval_result_weights["f2"] = self._get_fmeasure(2)

        for metric in metrics:
            for n in tops_n:
                self.result_values[metric + "@" + str(n)] = eval_results[metric][n]
                self.result_weights[metric + "@" + str(n)] = eval_result_weights[metric][n]

    #
    #
    ###########################################################################
    # private methods
    #
    ###########################################################################
    # return [hr@5, hr@10, hr@15, hr@20, hr@30, hr@40, hr@50]
    def _get_hit_ratio(self):
        nb_u = 1.0 * len(self.rec_links_binary)
        hri, wi = {}, {}
        for i in tops_n:
            hri[i], wi[i] = 0.0, nb_u

        if nb_u > 0:
            for u in self.rec_links_binary.keys():
                for i in tops_n:
                    if sum(self.rec_links_binary[u][:i]) >= 1:
                        hri[i] += 1
            for i in tops_n:
                hri[i] = (hri[i] * 1.0) / nb_u

        return hri, wi

    #
    ###########################################################################
    # return [precision@5, precision@10, precision@15, precision@20, precision@30, precision@40, precision@50]
    def _get_precision(self):
        preci, deno_preci, nume_preci, wi = {}, {}, {}, {}
        for i in tops_n:
            preci[i], deno_preci[i], nume_preci[i], wi[i] = 0.0, 0.0, 0.0, 0.0

        for u in self.rec_links_binary.keys():
            for i in tops_n:
                # denominator [number of recommendations]
                deno_preci[i] += len(self.rec_links[u][:i])

                # numerator [number of good recommendations]
                nume_preci[i] += sum(self.rec_links_binary[u][:i])

        for i in tops_n:
            # compute the precision metric
            preci[i] = (1.0 * nume_preci[i]) / (1.0 * deno_preci[i]) if deno_preci[i] > 0 else 0.0
            wi[i] = deno_preci[i]

        return preci, wi

    #
    ###########################################################################
    # return [recall@5, recall@10, recall@15, recall@20, recall@30, recall@40, recall@50]
    def _get_recall(self):
        recalli, deno_recalli, nume_recalli, wi = {}, {}, {}, {}
        for i in tops_n:
            recalli[i], deno_recalli[i], nume_recalli[i], wi[i] = 0.0, 0.0, 0.0, 0.0

        for u in self.rec_links_binary.keys():
            for i in tops_n:
                # denominator [number of links observed]
                u_nb_links_to_rec = len(self.links_to_rec[u])
                deno_recalli[i] = deno_recalli[i] + u_nb_links_to_rec

                # numerator [number of good recommendations]
                nume_recalli[i] += sum(self.rec_links_binary[u][:i])

        for i in tops_n:
            # compute the recall metric
            recalli[i] = (1.0 * nume_recalli[i]) / (1.0 * deno_recalli[i]) if deno_recalli[i] > 0 else 0.0
            wi[i] = deno_recalli[i]

        return recalli, wi

    #
    ###########################################################################
    # return [map@5, map@10, map@15, map@20, map@30, map@40, map@50]
    def _get_map(self):
        nb_u = 1.0 * len(self.rec_links_binary)
        mapi, nume_mapi, wi, = {}, {}, {}
        for i in tops_n:
            mapi[i], nume_mapi[i], wi[i] = 0.0, 0.0, nb_u

        if nb_u > 0:
            for u in self.rec_links_binary.keys():
                for i in tops_n:
                    nume_mapi[i] += self._get_average_precision(self.rec_links_binary[u][:i])
            for i in tops_n:
                # compute the map metric
                mapi[i] = (1.0 * nume_mapi[i]) / (1.0 * nb_u)
        return mapi, wi

    def _get_average_precision(self, user_rec_links_binary):
        average_precision = 0  # average precision
        indexes_good_rec = [index for index, val in enumerate(user_rec_links_binary) if val == 1]
        for index_val in indexes_good_rec:
            index_position = indexes_good_rec.index(index_val)
            average_precision += (1.0 * (index_position + 1)) / (1.0 * (index_val + 1))
        average_precision = (1.0 * average_precision) / (1.0 * len(indexes_good_rec)) if len(
            indexes_good_rec) > 0 else 0.0
        return average_precision

    #
    ###########################################################################
    # Mean Reciprocal Rank
    # return [mrr@5, mrr@10, mrr@15, mrr@20, mrr@30, mrr@40, mrr@50]
    def _get_mrr(self):
        nb_u = 1.0 * len(self.rec_links_binary)
        mrri, nume_mrri, wi, = {}, {}, {}
        for i in tops_n:
            mrri[i], nume_mrri[i], wi[i] = 0.0, 0.0, nb_u

        if nb_u > 0:
            for u in self.rec_links_binary.keys():
                for i in tops_n:
                    nume_mrri[i] += self._get_reciprocal_rank(self.rec_links_binary[u][:i])
            for i in tops_n:
                # compute the mrr metric
                mrri[i] = nume_mrri[i] / (1.0 * nb_u)
        return mrri, wi

    def _get_reciprocal_rank(self, user_rec_links_binary):
        reciprocal_rank = 0.0  # reciprocal rank
        if 1 in user_rec_links_binary:
            index_first_good_rec = user_rec_links_binary.index(1)
            reciprocal_rank = 1.0 / (1.0 * (index_first_good_rec + 1))
        return reciprocal_rank

    #
    ###########################################################################
    # F-Measure
    # return [fm@5, fm@10, fm@15, fm@20, fm@30, fm@40, fm@50]
    def _get_fmeasure(self, b):

        true_positivei, false_negativei, false_positivei, fmi, wi = {}, {}, {}, {}, {}
        for i in tops_n:
            true_positivei[i], false_negativei[i], false_positivei[i], fmi[i], wi[i] = 0.0, 0.0, 0.0, 0.0, 0.0

        for u in self.rec_links_binary.keys():
            for i in tops_n:
                # True positive [number of good recommendations]
                true_positivei[i] += sum(self.rec_links_binary[u][:i])

                # False negative [number of links observed but which are not predicted]
                false_negativei[i] += (len(self.links_to_rec[u]) - true_positivei[i])

                # False positive [predicted True but which are False]
                false_positivei[i] += (i - true_positivei[i])

        for i in tops_n:
            # numerator of F-measure
            fmi[i] = (1 + b * b) * true_positivei[i]

            # denominator of F-measure
            wi[i] = ((1 + b * b) * true_positivei[i]) + (b * b * false_negativei[i]) + false_positivei[i]

            fmi[i] = (1.0 * fmi[i]) / (1.0 * wi[i]) if wi[i] > 0 else 0.0

        return fmi, wi
#
###############################################################################
###############################################################################