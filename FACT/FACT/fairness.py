import numpy as np
import scipy as scp
from itertools import chain, combinations
from sklearn.metrics import confusion_matrix

def compute_stats(X, y, model):
    y_pred = model.predict(X)
    conf = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = conf.ravel()
    PPV = TP / (TP + FP)
    TPR = TP / (TP + FN)
    FDR = FP / (TP + FP)
    FPR = FP / (FP + TN)
    FOR = FN / (TN + FN)
    FNR = FN / (TP + FN)
    NPV = TN / (TN + FN)
    TNR = TN / (TN + FP)
    ACC = (TP + TN) / y_pred.shape[0]

    out = dict()
    out['conf'] = conf
    out['TN'] = TN
    out['FP'] = FP
    out['FN'] = FN
    out['TP'] = TP
    out['PPV'] = PPV
    out['TPR'] = TPR
    out['FDR'] = FDR
    out['FPR'] = FPR
    out['FOR'] = FOR
    out['FNR'] = FNR
    out['NPV'] = NPV
    out['TNR'] = TNR
    out['ACC'] = ACC
    
    return out

class FairnessMeasures():
    def __init__(self, X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, model, sens_idx, pos_label=1, neg_label=0):
        # X_train, y_train, X_test, y_test : dataset containing all attributes
        # X_train_removed, X_test_removed : dataset without the sensitive attributes
        # sens_idx : index of the sensitive attribute to investigate
        # pos_label/neg_label : categorical value the sensitive attribute takes

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_removed = X_train_removed
        self.X_test_removed = X_test_removed
        self.model = model
        self.sens_idx = sens_idx
        self.pos_label = pos_label
        self.neg_label = neg_label
        
        self.pos_group = np.where(X_test[:, sens_idx] == pos_label)[0]
        self.neg_group = np.where(X_test[:, sens_idx] == neg_label)[0]
        
        self.pos_group_num = self.pos_group.shape[0]
        self.neg_group_num = self.neg_group.shape[0]

        self.y_pred = model.predict(X_test_removed)
        self.pos_pred = np.where(self.y_pred == 1)[0]
        self.neg_pred = np.where(self.y_pred == 0)[0]
        
        self.pos_gt = np.where(y_test == 1)[0]
        self.neg_gt = np.where(y_test == 0)[0]

        mu1_idx = list(set(self.pos_gt).intersection(set(self.pos_group)))
        self.mu_pos = len(mu1_idx)
        mu2_idx = list(set(self.pos_gt).intersection(set(self.neg_group)))
        self.mu_neg = len(mu2_idx)
        self.pos_base_rate = self.mu_pos / self.pos_group_num
        self.neg_base_rate = self.mu_neg / self.neg_group_num

        out = compute_stats(X_test_removed, y_test, model)
        
        self.conf = out['conf']
        self.TN, self.FP, self.FN, self.TP = out['TN'], out['FP'], out['FN'], out['TP']
        self.ACC = out['ACC']
        
        self.PPV = self.TP / (self.TP + self.FP)
        self.TPR = self.TP / (self.TP + self.FN)
        self.FDR = self.FP / (self.TP + self.FP)
        self.FPR = self.FP / (self.FP + self.TN)
        self.FOR = self.FN / (self.TN + self.FN)
        self.FNR = self.FN / (self.TP + self.FN)
        self.NPV = self.TN / (self.TN + self.FN)
        self.TNR = self.TN / (self.TN + self.FP)

        # Split up the data into two groups for easier comparison
        self.X_test_pos_sens = X_test_removed[self.pos_group, :]
        self.X_test_neg_sens = X_test_removed[self.neg_group, :]
        self.y_test_pos_sens = y_test[self.pos_group]
        self.y_test_neg_sens = y_test[self.neg_group]

        self.pos_group_stats = compute_stats(self.X_test_pos_sens, self.y_test_pos_sens, self.model)
        self.neg_group_stats = compute_stats(self.X_test_neg_sens, self.y_test_neg_sens, self.model)

        self.proba =  hasattr(model, 'predict_proba')
        if self.proba:
            self.prob_vals = self.model.predict_proba(X_test_removed)[:,1]
            num_bin = 100
            self.bins = np.linspace(0, 1, num_bin+1)
            disc = np.digitize(self.prob_vals, self.bins)

            self.pval_disc = np.copy(self.prob_vals)
            for i in np.unique(disc):
                idx = np.where(disc == i)[0]
                self.pval_disc[idx] = self.bins[i-1]

            self.bin_vals = []

            for i in np.unique(disc):
                idx = np.where(disc == i)[0]
                d1 = len(set(idx).intersection(set(self.pos_group)))
                n1 = len(set(idx).intersection(set(self.pos_group)).intersection(set(self.pos_gt)))
                d2 = len(set(idx).intersection(set(self.neg_group)))
                n2 = len(set(idx).intersection(set(self.neg_group)).intersection(set(self.pos_gt)))

                try:
                    self.bin_vals.append((n1 / d1, n2 / d2))
                except ZeroDivisionError:
                    if d1 == 0:
                        if d2 == 0:
                            self.bin_vals.append((0,0))
                        else:
                            self.bin_vals.append((0, n2 / d2))
                    else:
                        if d2 == 0:
                            self.bin_vals.append((n1 / d1, 0))

    def group_parity(self):
        pos_val = len(set(self.pos_group).intersection(set(self.pos_pred))) / len(set(self.pos_group))
        neg_val = len(set(self.neg_group).intersection(set(self.pos_pred))) / len(set(self.neg_group))
        return pos_val, neg_val

    def predictive_rate_parity(self):
        pos_val = self.pos_group_stats['PPV']
        neg_val = self.neg_group_stats['PPV']
        return pos_val, neg_val

    def false_pos_balance(self):
        # False positive error rate balance (predictive inequality): equal FPR for both groups
        pos_val = self.pos_group_stats['FPR']
        neg_val = self.neg_group_stats['FPR']
        return pos_val, neg_val

    def false_neg_balance(self):
        # False negative error rate balance (equal opportunity): equal FNR for both groups
        pos_val = self.pos_group_stats['FNR']
        neg_val = self.neg_group_stats['FNR']
        return pos_val, neg_val

    def overall_accuracy_equality(self):
        # overall accuracy equality: equal prediction accuracy
        pos_val = self.pos_group_stats['ACC']
        neg_val = self.neg_group_stats['ACC']
        return pos_val, neg_val

    def equalized_odds(self):
        # equal TPR and FPR for both groups
        # output: ((TPR for pos group, TPR for neg group), (FRP for pos group, FPR for neg group))
        pos_val1 = self.pos_group_stats['TPR']
        neg_val1 = self.neg_group_stats['TPR']

        pos_val2 = self.pos_group_stats['FPR']
        neg_val2 = self.neg_group_stats['FPR']

        return ((pos_val1, neg_val1), (pos_val2, neg_val2))

    def cond_use_accuracy_equality(self):
        # conditional use accuracy equality:: equal PPV and NPV for both groups
        # output: ((PPV for pos group, PPV for neg group), (NPV for pos group, NPV for neg group))
        pos_val1 = self.pos_group_stats['PPV']
        neg_val1 = self.neg_group_stats['PPV']

        pos_val2 = self.pos_group_stats['NPV']
        neg_val2 = self.neg_group_stats['NPV']
        return ((pos_val1, neg_val1), (pos_val2, neg_val2))

    def test_fairness(self):
        if not self.proba:
            raise ValueError('probabilistic output not supported for this classifier.')
        out = []
        for (v1, v2) in self.bin_vals:
            out.append(v1 - v2)
        return out

    def well_calibration(self):
        if not self.proba:
            raise ValueError('probabilistic output not supported for this classifier.')
        out = []
        for e, (v1, v2) in zip(self.bins[:-1], self.bin_vals):
            out.append((e, v1, v2))
        return out

    def pos_class_balance(self):
        if not self.proba:
            raise ValueError('probabilistic output not supported for this classifier.')
        id1 = list(set(self.pos_gt).intersection(set(self.pos_group)))
        id2 = list(set(self.pos_gt).intersection(set(self.neg_group)))
        return np.mean(self.pval_disc[id1]), np.mean(self.pval_disc[id2])

    def neg_class_balance(self):
        if not self.proba:
            raise ValueError('probabilistic output not supported for this classifier.')
        id1 = list(set(self.neg_gt).intersection(set(self.pos_group)))
        id2 = list(set(self.neg_gt).intersection(set(self.neg_group)))
        return np.mean(self.pval_disc[id1]), np.mean(self.pval_disc[id2])

    # Diff versions
    def group_parity_diff(self):
        a, b = self.group_parity()
        return np.abs(a - b)

    def disparate_impact(self):
        a, b = self.group_parity()
        return b / a

    def predictive_rate_parity_diff(self):
        a, b = self.predictive_rate_parity()
        return np.abs(a - b)

    def false_pos_balance_diff(self):
        # PE
        a, b = self.false_pos_balance()
        return np.abs(a - b)

    def false_neg_balance_diff(self):
        # EOp
        a, b = self.false_neg_balance()
        return np.abs(a - b)

    def overall_accuracy_equality_diff(self):
        a, b = self.overall_accuracy_equality()
        return np.abs(a - b)

    def equalized_odds_diff(self):
        ((a,b), (c,d)) = self.equalized_odds()
        return np.abs(a-b), np.abs(c-d)

    def cond_use_accuracy_equality_diff(self):
        ((a,b), (c,d)) = self.cond_use_accuracy_equality()
        return np.abs(a-b), np.abs(c-d)

    def pos_class_balance_diff(self):
        a, b = self.pos_class_balance()
        return np.abs(a-b)

    def neg_class_balance_diff(self):
        a, b = self.neg_class_balance()
        return np.abs(a-b)

    def well_calibration_diff(self):
        calib = self.well_calibration()
        out = 0
        for (e, v1, v2) in calib:
            out += (e - v1)**2 + (e-v2)**2
        return np.sqrt(out / len(calib))

    def well_calibration_group_diff(self):
        calib = self.well_calibration()
        out1 = 0
        out2 = 0
        for (e, v1, v2) in calib:
            out1 += (e - v1) ** 2
            out2 += (e - v2) ** 2
        return np.sqrt(out1 / len(calib)), np.sqrt(out2 / len(calib))


def compute_base_rates(X, y, sens_idx, pos_label=1, neg_label=0):
    """
    Compute base rates for both groups
    :param X: features
    :param y: labels
    :param sens_idx: sensitive feature index
    :param pos_label: value for positive group
    :param neg_label: value for negative group
    :return: base rates for pos and neg group
    """
    pos_group = np.where(X[:, sens_idx] == pos_label)[0]
    neg_group = np.where(X[:, sens_idx] == neg_label)[0]
    N1 = pos_group.shape[0]
    N2 = neg_group.shape[0]
    pos_gt = np.where(y == 1)[0]
    neg_gt = np.where(y == 0)[0]
    mu1_idx = list(set(pos_gt).intersection(set(pos_group)))
    mu1 = len(mu1_idx)
    mu2_idx = list(set(pos_gt).intersection(set(neg_group)))
    mu2 = len(mu2_idx)
    r1 = mu1 / N1
    r2 = mu2 / N2
    return r1, r2


