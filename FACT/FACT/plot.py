import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from .lin_opt import *
from .helper import *
from .fairness import *
from .data_util import *

def plot_synth_data(X, y, sens_attr, axs=None):
    if axs == None:
        f, axs = plt.subplots()
    pos_idx = np.where(sens_attr == 1)[0]
    neg_idx = np.where(sens_attr == 0)[0]
    pos_labels = np.where(y == 1)[0]
    neg_labels = np.where(y == 0)[0]
    pos_class_pos_group = list(set(pos_idx).intersection(set(pos_labels)))
    pos_class_neg_group = list(set(pos_labels).intersection(set(neg_idx)))
    neg_class_pos_group = list(set(neg_labels).intersection(set(pos_idx)))
    neg_class_neg_group = list(set(neg_labels).intersection(set(neg_idx)))
    axs.scatter(X[pos_class_pos_group,0], X[pos_class_pos_group,1], edgecolors='g', marker='o', facecolors='none', alpha=0.7)
    axs.scatter(X[pos_class_neg_group,0], X[pos_class_neg_group,1], color='g', marker='x', alpha=0.7)
    axs.scatter(X[neg_class_pos_group,0], X[neg_class_pos_group,1], edgecolors='r', marker='o', facecolors='none', alpha=0.7)
    axs.scatter(X[neg_class_neg_group,0], X[neg_class_neg_group,1], color='r', marker='x', alpha=0.7)
    axs.set_xlabel('$x_1$')
    axs.set_ylabel('$x_2$')
    return axs

def plot_accuracy_obj(fm, names, eps_up=0.03, axs=None, color=None, label=None, alpha=1):
    if axs == None:
        f, axs = plt.subplots()

    # Second plot: optimize performance
    def _sanity_check(start):
        tmp = np.linspace(start, 0, 1000)
        for r in tmp:
            res = test_fair_instance(fm, names, opt_target='performance', eps=r)
            if res.success:
                return False
        return True

    #eps_vals = np.linspace(eps_up, 0, 1000)
    eps_vals = np.logspace(0, -6, 500)
    objs = []
    eps_used = []
    failed_spot = None
    for eps in eps_vals:
        res = test_fair_instance(fm, names, opt_target='performance', eps=eps)
        if res.success:
            objs.append(res.fun)
            eps_used.append(eps)
        else:
            failed_spot = eps
            nn = ','.join(names)
            print('{}\t{}'.format(nn, failed_spot))
            break
            #print(failed_spot)
            #if _sanity_check(failed_spot):
            #    break
            #else:
            #    raise ValueError('Solution range not complete')

    objs = np.array(objs)
    objs[objs < 0 ] = 0
    omin = 1 - np.max(objs)
    omax = 1 - np.min(objs)
    axs.set_ylabel('Relative Accuracy (1 - $\delta$)')
    axs.set_xlabel('Fairness Relaxation ($\epsilon$)')
    if failed_spot != None and label != None:
        label += ' cannot find solution for $\epsilon < $%0.4f'%(failed_spot)
    #axs.plot(eps_used, 1-objs, label=label)
    if color != None:
        axs.semilogx(eps_used, 1-objs, color=color, label=label, alpha=alpha)
    else:
        axs.semilogx(eps_used, 1-objs, label=label, alpha=alpha)
    if label != None:
        axs.legend()
    return axs

def accuracy_meo(fm, names):

    # Second plot: optimize performance
    def _sanity_check(start):
        tmp = np.linspace(start, 0, 1000)
        for r in tmp:
            res = test_fair_instance(fm, names, opt_target='performance', eps=r)
            if res.success:
                return False
        return True

    #eps_vals = np.linspace(eps_up, 0, 1000)
    eps_vals = np.logspace(0, -6, 500)
    objs = []
    eps_used = []
    meo = []
    failed_spot = None
    for eps in eps_vals:
        res = test_fair_instance(fm, names, opt_target='performance', eps=eps)
        if res.success:
            objs.append(res.fun)
            eps_used.append(eps)
            meo.append(compute_meo(res.x))
        else:
            failed_spot = eps
            nn = ','.join(names)
            print('{}\t{}'.format(nn, failed_spot))
            break
            #print(failed_spot)
            #if _sanity_check(failed_spot):
            #    break
            #else:
            #    raise ValueError('Solution range not complete')

    objs = np.array(objs)
    objs[objs < 0 ] = 0



    return 1-objs, meo

def compute_meo(confusion):
    tp1 = confusion[0]
    fn1 = confusion[1]
    fp1 = confusion[2]
    tn1 = confusion[3]
    tp0 = confusion[4]
    fn0 = confusion[5]
    fp0 = confusion[6]
    tn0 = confusion[7]

    tpr0 = tp0 / (tp0 + fn0)
    tpr1 = tp1 / (tp1 + fn1)
    fpr0 = fp0 / (fp0 + tn0)
    fpr1 = fp1 / (fp1 + tn1)

    tpr_diff = tpr0 - tpr1
    fpr_diff = fpr0 - fpr1

    meo = np.abs(tpr_diff + fpr_diff) / 2
    return meo

def plot_eps_delta_curves(fm, some_names, lmbds_used, groups=None, colors=None, save=False, data_name=None):
    f, axs = plt.subplots(figsize=(4,2))

    # plot solid lines
    for i, name in enumerate(some_names):
        idx = groups[i]
        axs = plot_accuracy_obj(fm, name, eps_up=1, axs=axs, color=colors[idx], alpha=1)

    # plot crosses
    for i, l in enumerate(lmbds_used):
        vals = l[1]
        ll = l[0]
        x = np.array([v[1] for v in vals])
        x[x < 1e-6] = 1e-6
        y = np.array([v[0] for v in vals])
        idx = groups[i]
        axs.scatter(x, y, color=colors[idx], marker='x', alpha=0.4)

    axs.set_title(data_name)
    axs.set_ylim(0.3, None)
    #axs.set_xlim(1e-13, None)
    plt.gca().invert_xaxis()

    # save
    if save:
        if data_name == None:
            raise ValueError('Input dataname.')
        else:
            fname = 'eps_delta_plot_%s.pdf'%data_name
        plt.savefig(fname, bbox_inches='tight', dpi=200)
    return axs

def plot_accuracy_contours(mats_dict, name, M_const, b_const, bound=3, save=False, data_name=None, axs=None):
    abbrvs = get_abbrvs()
    M, b = get_with_names(mats_dict, name)
    Ms = [get_with_names(mats_dict, [n])[0] for n in name]

    try:
        nd = len(bound)
    except TypeError:
        nd = 0

    if nd == 0:
        xx, yy = np.meshgrid(np.arange(-bound, bound, 0.1), np.arange(-bound, bound, 0.1))
    elif nd == 2:
        xx, yy = np.meshgrid(np.arange(-bound[0], bound[0], 0.1), np.arange(-bound[1], bound[1], 0.1))
    xx = np.power(10, xx)
    yy = np.power(10, yy)
    xr = xx.ravel()
    yr = yy.ravel()

    z = np.zeros(xr.shape[0])
    for i, (x,y) in enumerate(zip(xr, yr)):
        lmbd = [x, y]
        res = solve_LAFOP_multireg(M, M_const, b_const, lmbd, name, Ms)
        z[i] = 1 - res.fun
    z = z.reshape(xx.shape)
    z[z > 1.0] = 1.0
    
    if axs == None:
        f, axs = plt.subplots(figsize=(2,2))
    ff = axs.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.8, vmin=0.32,vmax=0.96)
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.set_aspect('equal')
    axs.set_xlabel('$\lambda_{%s}$'%abbrvs[name[0]], fontsize=13)
    axs.set_ylabel('$\lambda_{%s}$'%abbrvs[name[1]], fontsize=13)
    axs.set_title(data_name)
    if data_name == 'S(B)' or data_name == 'S(U)' or data_name=='Adult':
        ticks = [0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.96]
    tmp = f.colorbar(ff, ticks=ticks)
    if save:
        plt.savefig('contours_%s.pdf'%data_name, bbox_inches='tight', dpi=300)
    return axs, tmp

def plot_slices(mats_dict, list_name, M_const, b_const, save=False, data_name=None):
    abbrvs = get_abbrvs()
    names = []
    # create permutations
    for n in list_name:
        tmp = list_name.copy()
        tmp.remove(n)
        tmp.append(n)
        names.append(tmp)
    num_plots = len(names)
    abbrv_n = [abbrvs[n] for n in list_name]

    f, axs = plt.subplots(1,num_plots,figsize=(3*num_plots,2), sharey=True)
    ref_vals = [0.01, 0.1, 1., 10., 100.]
    for i, name in enumerate(names):
        M, b = get_with_names(mats_dict, name)
        Ms = [get_with_names(mats_dict, [n])[0] for n in name]
        results = []
        for ref in ref_vals:
            out = []
            l_vals = np.logspace(-3,3,50)
            for j in l_vals:
                lmbd = np.ones(len(list_name)) * ref
                lmbd[-1] = j
                res = solve_LAFOP_multireg(M, M_const, b_const, lmbd, name, Ms)
                out.append(1 - res.fun)
            results.append(out)

        legend_label = '='.join(['$\lambda_{%s}$'%(abbrvs[n]) for n in name[:-1]])
        for r, res in zip(ref_vals, results):
            axs[i].semilogx(l_vals, res, label='%s=%0.0E'%(legend_label, r))
        axs[i].set_xlabel('$\lambda_{%s}$'%(abbrvs[name[-1]]))
        if i == 0:
            axs[i].set_ylabel('Relative Accuracy\n($1 - \delta$)')
        #axs[i].set_title('Adding %s to (%s, %s)'%(abbrvs[name[-1]], abbrvs[name[0]], abbrvs[name[1]]))  
        axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), labelspacing=0.01, ncol=1, fontsize=10)

    #f.suptitle('2D Slices of %dD Surface with (%s)'%(len(list_name)+1, ','.join(abbrv_n)), fontsize=14, y=1.01)
    #plt.tight_layout()
    if save:
        plt.savefig('slices_%s.pdf'%data_name, bbox_inches='tight', dpi=300)
    return axs

def plot_frontier(fm, fairness_name, axs=None, save=True, label=False, plot=False):
    eps_vals = np.logspace(0, -6,100)
    acc_vals = []
    eps_used = []
    for eps in eps_vals:
        res = test_fair_instance(fm, [fairness_name], eps=eps, epsdelta=False)
        if res.success:
            acc_vals.append(1 - res.fun)
            eps_used.append(eps)
        else:
            break

    if plot:
        if axs == None:
            f, axs = plt.subplots()

        if label:
            axs.semilogx(eps_used, acc_vals, color='k', label='frontier')
        else:
            axs.semilogx(eps_used, acc_vals, color='k')

        #if model_results is not None:
        #    for k in model_results.keys():
        #        fairness, accuracy = model_results[k]
        #        axs.scatter(fairness, accuracy, label=k)
        # other formatting stuff
        axs.set_xlim(np.min(eps_used), np.max(eps_used))
        axs.set_ylim(np.min(acc_vals), 1.01)
        axs.invert_xaxis()
        axs.set_title('Model-Agnostic Pareto Frontier on Accuracy and %s'%fairness_name)
        axs.set_ylabel('Accuracy')
        axs.set_xlabel('Fairness gap [%s]\n(smaller the better)'%fairness_name)
        axs.set_aspect('equal')
        if label:
            axs.legend()
        plt.tight_layout()
        if save:
            plt.savefig('frontier_%s.pdf'%fairness_name, bbox_inches='tight', dpi=300)
    return axs, eps_used, acc_vals

##################
# Outdated methods
##################
def plot_fairness_gap(fm, mats_dict, names, axs=None, label=None):
    if axs == None:
        f, axs = plt.subplots()
    M, b = get_with_names(mats_dict, names)

    # First plot: optimize fairness
    errs = np.linspace(0,0.2,100)
    objs = []
    xvals = []
    for e in errs:
        res = test_fair_instance(fm, names, opt_target='fairness', err_ub=e)
        objs.append(res.fun)
        xvals.append(res.x)
    np.abs(np.dot(M, res.x) - np.squeeze(b))
    objs = np.array(objs)

    axs.plot(1-errs, objs, label=label)
    axs.set_title('Accuracy vs. Fairness Trade-off for \n %s'%names)
    axs.set_xlabel('Accuracy Lower Bound')
    axs.set_ylabel('$||Ax - b||_2$ (smaller then better)')
    return axs

def plot_tradeoff_curves(fm, mats_dict, names, save=True, eps_up=0.03, label=None):
    f, axs = plt.subplots(1, 2, figsize=(5*2, 3))
    axs[0] = plot_fairness_gap(fm, mats_dict, names, label=label, axs=axs[0])
    axs[1] = plot_accuracy_obj(fm, names, eps_up=eps_up, label=label, axs=axs[1])
    plt.tight_layout()
    if save:
        plt.savefig('acc_fairness.png', dpi=200)
    return axs
