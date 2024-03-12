from numpy import (
    vstack,
    random,
    round,
    var,
    zeros,
    mean,
    argsort,
    nanmean,
    array,
    empty,
    arange,
    nan_to_num,
)
import matplotlib.pyplot as plt
from wrf_fvcom.variables import PerturbedVariable, VariableDistribution
from surrogate.utils import surrogate_model_predict

###########################################
#  ____     ___    ____     ___    _      #
# / ___|   / _ \  | __ )   / _ \  | |     #
# \___ \  | | | | |  _ \  | | | | | |     #
#  ___) | | |_| | | |_) | | |_| | | |___  #
# |____/   \___/  |____/   \___/  |_____| #
#                                         #
###########################################


class smethod(object):
    def __init__(self, variable_matrix, sens_names):
        self.dim = variable_matrix.shape[1]
        self.sens = dict((k, [None] * self.dim) for k in self.sens_names)
        self.sens_ready = dict((k, [False] * self.dim) for k in self.sens_names)
        self.ptypes = []
        for scheme in variable_matrix['scheme']:
            variable = PerturbedVariable.class_from_scheme_name(scheme)
            if variable.variable_distribution == VariableDistribution.DISCRETEUNIFORM:
                self.ptypes.append('int')
            else:
                self.ptypes.append('float')


class sobol(smethod):
    ## Main and total are based on Saltelli 2010; it computes joint in the total sense!
    ##
    ## Initialization
    def __init__(self, variable_matrix):
        print('Initializing SOBOL')
        self.sens_names = ['main', 'total', 'jointt']
        smethod.__init__(self, variable_matrix, self.sens_names)

    def sample(self, ninit, parameter_types=None):
        print('Sampling SOBOL')

        sam1 = random.rand(ninit, self.dim)
        sam2 = random.rand(ninit, self.dim)

        for pp, par_type in enumerate(self.ptypes):
            if par_type == 'int':
                sam1[:, pp] = round(sam1[:, pp])
                sam2[:, pp] = round(sam2[:, pp])

        xsam = vstack((sam1, sam2))

        for id in range(self.dim):
            samid = sam1.copy()
            samid[:, id] = sam2[:, id]
            xsam = vstack((xsam, samid))

        self.nsam = xsam.shape[0]
        self.sens_ready['main'] = True
        self.sens_ready['total'] = True
        self.sens_ready['jointt'] = True

        return xsam

    def compute(self, ysam, computepar=None):
        ninit = self.nsam // (self.dim + 2)
        y1 = ysam[ninit : 2 * ninit]
        yvar = var(ysam[: 2 * ninit])
        si = zeros((self.dim,))
        ti = zeros((self.dim,))
        jtij = zeros((self.dim, self.dim))

        for id in range(self.dim):
            y2 = ysam[2 * ninit + id * ninit : 2 * ninit + (id + 1) * ninit] - ysam[:ninit]
            si[id] = mean(y1 * y2) / yvar
            ti[id] = 0.5 * mean(y2 * y2) / yvar
            for jd in range(id):
                y3 = (
                    ysam[2 * ninit + id * ninit : 2 * ninit + (id + 1) * ninit]
                    - ysam[2 * ninit + jd * ninit : 2 * ninit + (jd + 1) * ninit]
                )
                jtij[id, jd] = ti[id] + ti[jd] - 0.5 * mean(y3 * y3) / yvar

        self.sens['main'] = si
        self.sens['total'] = ti
        self.sens['jointt'] = jtij.T

        return self.sens


def compute_sensitivities(surrogate_model, variable_matrix, sample_size=10000, kl_dict=None):

    # get the sensitivity sample matrix
    SensMethod = sobol(variable_matrix)
    xsam = SensMethod.sample(sample_size)
    # evaluate the surrogate model at the samples
    ysam = surrogate_model_predict(surrogate_model, xsam, kl_dict=kl_dict)

    npts = ysam.shape[1]
    variable_names = []
    variable_prior = ''
    for sdx, scheme in enumerate(variable_matrix['scheme']):
        variable_name = PerturbedVariable.class_from_scheme_name(scheme).name
        if variable_name != variable_prior:
            variable_names.append(variable_name)
        variable_prior = variable_name
    ndim = len(variable_names)
    sens_dict = {
        'main': zeros((npts, ndim)),
        'total': zeros((npts, ndim)),
        'jointt': zeros((npts, ndim, ndim)),
        'variable_names': variable_names,
    }
    for i in range(npts):
        sens = SensMethod.compute(ysam[:, i] - ysam[:,i].mean())
        vdx = -1
        variable_prior = ''
        for sdx, scheme in enumerate(variable_matrix['scheme']):
            variable_name = PerturbedVariable.class_from_scheme_name(scheme).name
            if variable_name != variable_prior:
                vdx += 1
            variable_prior = variable_name
            sens_dict['main'][i, vdx] += sens['main'][sdx]
            sens_dict['total'][i, vdx] += sens['total'][sdx]
            # go in one loop further for joint sensitivities
            vdxx = -1
            variable_priorr = ''
            for sdxx, schemee in enumerate(variable_matrix['scheme']):
                variable_namee = PerturbedVariable.class_from_scheme_name(schemee).name
                if variable_namee != variable_priorr:
                    vdxx += 1
                variable_priorr = variable_namee
                if vdx == vdxx: # add internal parameterizations interactions to main effect
                   sens_dict['main'][i, vdx] += sens['jointt'][sdx, sdxx]
                else:
                   sens_dict['jointt'][i, vdx, vdxx] += sens['jointt'][sdx, sdxx]


    return sens_dict, ysam


def plot_sens(
    sensdata,
    pars,
    cases,
    vis='bar',
    reverse=False,
    par_labels=[],
    case_labels=[],
    colors=[],
    ncol=4,
    grid_show=True,
    xlbl='',
    ylbl='sensitivity',
    legend_show=2,
    legend_size=10,
    xdatatick=[],
    figname=None,
    showplot=False,
    senssort=True,
    topsens=[],
    lbl_size=22,
    yoffset=0.1,
    ylim_max=None,
    title='',
    xticklabel_size=None,
    xticklabel_rotation=0,
    maxlegendcol=4,
):
    """Plots sensitivity for multiple observables"""

    ncases = sensdata.shape[0]
    npar = sensdata.shape[1]

    wd = 0.6

    assert set(pars) <= set(range(npar))
    assert set(cases) <= set(range(ncases))

    # Set up the figure
    # TODO need to scale figure size according to the expected amount of legends

    if xticklabel_size is None:
        xticklabel_size = int(400 / ncases)

    fig = plt.figure(figsize=(20, 12))
    fig.add_axes([0.1, 0.2 + yoffset, 0.8, 0.6 - yoffset])

    # Default parameter names
    if par_labels == []:
        for i in range(npar):
            par_labels.append(('par_' + str(i + 1)))
    # Default case names
    if case_labels == []:
        for i in range(ncases):
            case_labels.append(('case_' + str(i + 1)))

    if reverse:
        tmp = par_labels
        par_labels = case_labels
        case_labels = tmp
        tmp = pars
        pars = cases
        cases = tmp
        sensdata = sensdata.transpose()

    npar_ = len(pars)
    ncases_ = len(cases)

    if senssort:
        sensind = argsort(nanmean(sensdata, axis=0))[::-1]
    else:
        sensind = arange(npar_)

    if topsens == []:
        topsens = npar_

    # Create colors list
    if colors == []:
        colors_ = set_colors(topsens)
        colors_.extend(set_colors(npar_ - topsens))
        colors = [0.0 for i in range(npar_)]
        for i in range(npar_):
            colors[sensind[i]] = colors_[i]

    case_labels_ = []
    for i in range(ncases_):
        case_labels_.append(case_labels[cases[i]])

    if xdatatick == []:
        xflag = False
        xdatatick = array(range(1, ncases_ + 1))
        sc = 1.0
    else:
        xflag = True
        sc = float(xdatatick[-1] - xdatatick[0]) / ncases_

    if vis == 'graph':
        for i in range(npar_):
            plt.plot(
                xdatatick_,
                sensdata[cases, i],
                '-o',
                color=colors[pars[i]],
                label=par_labels[pars[i]],
            )
    elif vis == 'bar':
        curr = zeros((ncases_))
        # print pars,colors
        for i in range(npar_):
            plt.bar(
                xdatatick,
                sensdata[cases, i],
                width=wd * sc,
                color=colors[pars[i]],
                bottom=curr,
                label=par_labels[pars[i]],
            )
            curr = nan_to_num(sensdata[cases, i]) + curr

        if not xflag:
            plt.xticks(
                array(range(1, ncases_ + 1)), case_labels_, rotation=xticklabel_rotation
            )

        plt.xlim(xdatatick[0] - wd * sc / 2.0 - 0.1, xdatatick[-1] + wd * sc / 2.0 + 0.1)

        # else:
        #    xticks(xdatatick)

    plt.ylabel(ylbl, fontsize=lbl_size)
    plt.xlabel(xlbl, fontsize=lbl_size)
    plt.title(title, fontsize=lbl_size)

    maxsens = max(curr.max(), 1.0)
    if ylim_max is None:
        plt.ylim([0, maxsens])
    else:
        plt.ylim([0, ylim_max])

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handles[i] for i in sensind[:topsens]]
    labels = [labels[i] for i in sensind[:topsens]]
    if legend_show == 1:
        plt.legend(handles, labels, fontsize=legend_size)
    elif legend_show == 2:
        plt.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(0.0, -0.15),
            fancybox=True,
            shadow=True,
            ncol=min(ncol, maxlegendcol),
            labelspacing=-0.1,
            fontsize=legend_size,
        )
    elif legend_show == 3:
        plt.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(0.0, 1.2),
            fancybox=True,
            shadow=True,
            ncol=min(ncol, maxlegendcol),
            labelspacing=-0.1,
            fontsize=legend_size,
        )

    if not xflag:
        zed = [
            tick.label.set_fontsize(xticklabel_size)
            for tick in plt.gca().xaxis.get_major_ticks()
        ]

    plt.grid(grid_show)

    if figname is not None:
        plt.savefig(figname)
    if showplot:
        plt.show()


def set_colors(npar):
    """ Sets a list of different colors of requested length, as rgb triples"""
    colors = []
    pp = 1 + int(npar / 6)
    for i in range(npar):
        c = 1 - float(int((i / 6)) / pp)
        b = empty((3))
        for jj in range(3):
            b[jj] = c * int(i % 3 == jj)
        a = int(int(i % 6) / 3)
        colors.append(
            (
                (1 - a) * b[2] + a * (c - b[2]),
                (1 - a) * b[1] + a * (c - b[1]),
                (1 - a) * b[0] + a * (c - b[0]),
            )
        )

    return colors
