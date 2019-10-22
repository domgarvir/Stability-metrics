# coding=utf-8
import sys # system parameters library, to work with arguments
import os  # operating system library, to change directories
import pandas as pd # Data análisis library, to work with csv
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler #machine learning library, data standarization
import numpy as np # basic scientific computing library, for cvariance matrix
from sklearn.decomposition import PCA as sklearnPCA #machine learning library, PCA analysis
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set() #for heatmap
sns.set(font_scale=1.0)
from math import sqrt as sqrt
import scipy.stats as stat
from scipy.special import gamma as gammaf
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, set_link_color_palette
from scipy.spatial.distance import pdist
from sklearn import linear_model
from pylab import *
import matplotlib.colors
import networkx as nx
import collections
import itertools
from collections import OrderedDict
import random as rnd

pd.set_option('precision', 2)
pd.set_option('display.max_columns', None)


####### calculations #####
#dataframes
#recalculation so all metrics' values increase with increasing stability
def prepare_dataframe(df_all,negative_metrics,nosign_metrics):


    df_all['Next2nd'] = (df_all['N'] - df_all['Robst'] * df_all['N']) #number of extinctions
    df_all['maxNext2nd'] = (df_all['N'] - df_all['minRobst'] * df_all['N']) #maximum nb of extinctions
    df_all['Ext_half'] = df_all['Ext_half'] / (df_all['N'])

    df_all['maxDBRobst'] = merge_bigger_fabs(df_all['maxDBRost'].tolist(), df_all['minDBRobst'].tolist())
    df_all['minDBRobst'] = merge_minor_fabs(df_all['maxDBRost'].tolist(), df_all['minDBRobst'].tolist())

    df_all['L1DBRMt_s']=df_all["L1DBRMt_s"]
    df_all['avL1DBRMt_i'] = df_all['L1DBRMt_s'] / (df_all['N'] * df_all['N'])  # idem
    df_all['absumSNorm'] = df_all['absumS'] / (df_all['N'] * df_all['N'])

    for column in nosign_metrics:
        try:
            df_all[column] = np.abs(df_all[column])
        except:
            pass


    for column in negative_metrics:
        try:
            df_all[column] = -1 * df_all[column]
        except:
            pass

    return df_all
def merge_bigger_fabs(a,b):
    #print("len is "  , len(a))
    max_list=[]
    for index in range(len(a)):
        if (np.fabs(a[index])>np.fabs(b[index])):
            max_value=a[index]
        else :
            max_value=b[index]
        max_list.append(np.fabs(max_value))
    #print(max_list)
    #print(len(a),len(max_list))
    return max_list
def merge_minor_fabs(a,b):
    #print("len is "  , len(a))
    min_list=[]
    for index in range(len(a)):
        if (np.fabs(a[index])<np.fabs(b[index])):
            min_value=a[index]
        else :
            min_value=b[index]
        min_list.append(np.fabs(min_value))
    #print(max_list)
    #print(len(a),len(max_list))
    return min_list
#sample 100 networks of each size to study thier stability
def get_sample_df(full_df,col_to_sample,sample_size,minsize=5,maxsize=100):

    sample_df=pd.DataFrame()

    for index in range(minsize,maxsize):
        criteria=(col_to_sample==index)
        #add a random sample with "sample_size" networks for each size, if there are not enough it uses all of them once.
        if (criteria.sum()>0):
            sample_df=sample_df.append(full_df[criteria].sample(min(sample_size,criteria.sum())))

    #print(sample_df)
    return sample_df

#return the corr_values and p_values as two dataframes of all pairwise correlations.
def get_full_correlation(df,corrtype,**kwargs):
    if (len(kwargs.keys()) == 0): #full dataset against itself
        l1 = list(df)
        l2=l1
    elif (len(kwargs.keys())==1) :#names against themselves
        l1 = kwargs['l1']
        l2=l1
    else : #one list versus another
        l1 = kwargs['l1']
        l2 = kwargs['l2']



    corr_values=pd.DataFrame(columns=l1,index=l2)
    p_values=pd.DataFrame(columns=l1, index=l2)
    scov=pd.DataFrame()
    scov=spearman_cov(df) #spearman covariance matrix, no I have to keep the elements and build the datafrme
    dfn = mynormalize(df) #mean normalization before calculating covariances
    pcov=dfn.cov() #calculate pearson covariances

    for x in l1:
        for y in l2:
            if (corrtype == "pearson") :
                corr_values[x][y],p_values[x][y]=stat.pearsonr(df[x], df[y])
            elif (corrtype == "spearman"):
                corr_values[x][y], p_values[x][y] = stat.spearmanr(df[x], df[y])

            else :
                print("What correlation? Specify a valid one")
                return
    corr_values = corr_values[corr_values.columns].astype(float)
    p_values = p_values[p_values.columns].astype(float)
    scov = scov[scov.columns].astype(float)

    #print(scov)

    return corr_values, p_values, scov, pcov
#return two dataframes of corr_values and pvalues trough size,with a moving window
def get_full_correlation_bysize(df,col_N,window,corrtype,**kwargs):

    print_scatt_full=0

    l1 = []
    l2 = []
    all_names = []

    if (len(kwargs.keys()) == 0):
        l1 = list(df)
        l2 = l1
        for i in range(len(l1)):
            for j in range(i+1,len(l1)):
                all_names.append(l1[i]+"."+l1[j])

    elif (len(kwargs.keys()) == 1):
        l1 = kwargs['l1']
        l2 = l1
        for i in range(len(l1)):
            for j in range(i+1,len(l1)):
                all_names.append(l1[i]+"."+l1[j])
    else:
        l1 = kwargs['l1']
        l2 = kwargs['l2']
        for i in range(len(l1)):
            for j in range(len(l2)):
                all_names.append(l2[j]+"."+l1[i])
    #print(l1,l2)
    all_names_cov = []
    for i in range(len(l1)):
        for j in range(i,len(l1)):
            all_names_cov.append(l1[i]+"."+l1[j])



    #print(all_names)
    corr_sdf=pd.DataFrame(columns=all_names)
    pval_sdf=pd.DataFrame(columns=all_names)
    scov_sdf = pd.DataFrame(columns=all_names_cov) #with self covariation
    pcov_sdf = pd.DataFrame(columns=all_names_cov)  # with self covariation

    #
    #obtain all those values tfor each size window
    for index in range(window.minsize, window.maxsize):
        #criteria = ((col_N >= index ) & (col_N< (index + window.width)))
        criteria = col_N.between(index, index+window.width, inclusive=True)
        # its possible to only take a subset of each window, so the sampling is uniform
        if (df[criteria].shape[0]>0) :
            sample_size = min(window.sample_size, df[criteria].shape[0])
            print("size %s, %s communites sampled " % (index , sample_size),df[criteria].shape[0], end='\r')
            sub_df = df[criteria].sample(sample_size)
            #sub_df = df[criteria]
            corr_mat, pval_mat, scov_mat, pcov_mat = get_full_correlation(sub_df,corrtype,l1=l1,l2=l2)
            #print(corr_mat)
            if (len(kwargs.keys()) < 2): #square : nvalues=2*len(l1)-l1
                corr_values = get_upper_values_from_matrix(corr_mat.values)
                pval_values = get_upper_values_from_matrix(pval_mat.values)
                scov_values = get_upper_values_from_matrix_wdiagonal(scov_mat.values)
                pcov_values = get_upper_values_from_matrix_wdiagonal(pcov_mat.values)
            else: #not square : nvalues=len(l1)*len(l2)
                corr_values = corr_values=corr_mat.T.values.flatten()
                pval_values = pval_values=corr_mat.T.values.flatten()
                scov_values = scov_values=scov_mat.T.values.flatten()
                pcov_values = scov_values = pcov_mat.T.values.flatten()

            #print(corr_values)
            corr_sdf.loc[index] = corr_values
            pval_sdf.loc[index] = pval_values
            scov_sdf.loc[index] = scov_values
            pcov_sdf.loc[index] = pcov_values



    #print(corr_sdf.head())
    #print(corr_sdf.mean())
    #we can now order them by the value of the correlation
    corr_sum = np.sum(np.absolute(corr_sdf), axis=0)
    new_colnames = corr_sum.sort_values(ascending=False).index.tolist()
    corr_sdf=corr_sdf[new_colnames]#no reindexamos porque ese orden es el del tamaño
    pval_sdf=pval_sdf[new_colnames]#idem
    #scov_sdf=scov_sdf[new_colnames]
    #and change the index so it reflects the mean value of size in the window rather than the minimum
    #corr_sdf.index += (window.width/2)
    #pval_sdf.index += (window.width/2)

    # #get also average correlations troguht the moving window
    # #now moving window is 1, so we dont need averaging for the covariance
    # corr_av=pd.DataFrame(columns=l2,index=l1)
    # corr_std=pd.DataFrame(columns=l2,index=l1)
    #
    # #print(corr_av)
    # for i in range(len(l1)):
    #     for j in range(len(l2)):
    #         if (len(kwargs.keys()) < 2):
    #             if (i>j) :
    #                 corr_av[l2[j]][l1[i]] = corr_sdf.mean()[l2[j] + "." + l1[i]]
    #                 corr_av[l1[i]][l2[j]] = corr_sdf.mean()[l2[j] + "." + l1[i]]
    #                 corr_std[l2[j]][l1[i]] = corr_sdf.std()[l2[j] + "." + l1[i]]
    #                 corr_std[l1[i]][l2[j]] = corr_sdf.std()[l2[j] + "." + l1[i]]
    #             if(i==j):
    #                 corr_av[l2[j]][l1[i]] = 1.
    #                 corr_std[l1[i]][l2[j]] = 1.
    #         else:
    #             corr_av[l2[j]][l1[i]] = corr_sdf.mean()[l2[j] + "." + l1[i]]
    #             corr_std[l2[j]][l1[i]] = corr_sdf.std()[l2[j] + "." + l1[i]]
    # for i in range(len(l1)):
    #     for j in range(len(l2)):
    #         if (len(kwargs.keys()) < 2):
    #             if (i>j) :
    #                 corr_av[l2[j]][l1[i]] = corr_sdf.mean()[l2[j] + "." + l1[i]]
    #                 corr_av[l1[i]][l2[j]] = corr_sdf.mean()[l2[j] + "." + l1[i]]
    #                 corr_std[l2[j]][l1[i]] = corr_sdf.std()[l2[j] + "." + l1[i]]
    #                 corr_std[l1[i]][l2[j]] = corr_sdf.std()[l2[j] + "." + l1[i]]
    #             if(i==j):
    #                 corr_av[l2[j]][l1[i]] = 1.
    #                 corr_std[l1[i]][l2[j]] = 1.
    #         else:
    #             corr_av[l2[j]][l1[i]] = corr_sdf.mean()[l2[j] + "." + l1[i]]
    #             corr_std[l2[j]][l1[i]] = corr_sdf.std()[l2[j] + "." + l1[i]]

    #print(corr_av)
    #return size-dataframes

    # new_l1, new_l2 = get_ordered_names(corr_av, l1=l1, l2=l2)
    # corr_av = corr_av[new_l1]
    # corr_av = corr_av.reindex(new_l2)
    # corr_std = corr_std[new_l1]
    # corr_std = corr_std.reindex(new_l2)

    #return corr_sdf, pval_sdf, corr_av, corr_stdç
    return corr_sdf, pval_sdf, scov_sdf, pcov_sdf
#return the cov_values and p_values as two dataframes.
def spearman_cov(df):

    df_rank = pd.DataFrame()
    for var in list(df):
        df_rank[var] = stat.rankdata(df[var])

    cov_values=df_rank.cov()
    cov_values = cov_values[cov_values.columns].astype(float)

    return cov_values
#without diagonal elements
def get_av_corr_from_sdf(df, **kwargs):
    #print(df)
    l1 = []
    l2 = []
    if (len(kwargs.keys()) == 0):
        l1 = list(df)
        l2 = l1


    elif (len(kwargs.keys()) == 1):
        l1 = kwargs['l1']
        l2 = l1

    else:
        l1 = kwargs['l1']
        l2 = kwargs['l2']

    #l1=[rename_dict[x] for x in l1]
    #l2=[rename_dict[x] for x in l2]
    #print(l1, l2)
    #print(list(df))
    # get also average correlations troguht the moving window
    df_av = pd.DataFrame(columns=l2, index=l1)
    df_std = pd.DataFrame(columns=l2, index=l1)
    #print(df_av)
    for i in range(len(l1)):
        for j in range(len(l2)):
            if (len(kwargs.keys()) < 2):
                if (i > j):
                    df_av[l2[j]][l1[i]] = df.mean()[l2[j] + "." + l1[i]]
                    df_av[l1[i]][l2[j]] = df.mean()[l2[j] + "." + l1[i]]
                    df_std[l2[j]][l1[i]] = df.std()[l2[j] + "." + l1[i]]
                    df_std[l1[i]][l2[j]] = df.std()[l2[j] + "." + l1[i]]
                if (i == j):
                    df_av[l2[j]][l1[i]] = 1.
                    df_std[l1[i]][l2[j]] = 1.
            else:
                df_av[l2[j]][l1[i]] = df.mean()[l2[j] + "." + l1[i]]
                df_std[l2[j]][l1[i]] = df.std()[l2[j] + "." + l1[i]]

    #print(df_av)
    #print(1-df_av)
    # return size-dataframes

    new_l1, new_l2 = get_ordered_names(df_av, l1=l1, l2=l2)
    df_av = df_av[new_l1]
    df_av = df_av.reindex(new_l2)
    df_std = df_std[new_l1]
    df_std = df_std.reindex(new_l2)

    return df_av, df_std
#with diagonal elements
def get_av_cov_from_sdf(df, **kwargs):
    #print(df)
    l1 = []
    l2 = []
    if (len(kwargs.keys()) == 0):
        l1 = list(df)
        l2 = l1


    elif (len(kwargs.keys()) == 1):
        l1 = kwargs['l1']
        l2 = l1

    else:
        l1 = kwargs['l1']
        l2 = kwargs['l2']

    #l1=[rename_dict[x] for x in l1]
    #l2=[rename_dict[x] for x in l2]
    #print(l1, l2)
    #print(list(df))
    # get also average correlations troguht the moving window
    df_av = pd.DataFrame(columns=l2, index=l1)
    #df_std = pd.DataFrame(columns=l2, index=l1)
    #print(df_av)
    for i in range(len(l1)):
        for j in range(len(l2)):
            if (len(kwargs.keys()) < 2):
                if (i > j):
                    df_av[l2[j]][l1[i]] = df.mean()[l2[j] + "." + l1[i]]
                    df_av[l1[i]][l2[j]] = df.mean()[l2[j] + "." + l1[i]]
                    #df_std[l2[j]][l1[i]] = df.std()[l2[j] + "." + l1[i]]
                    #df_std[l1[i]][l2[j]] = df.std()[l2[j] + "." + l1[i]]
                if (i == j):
                    df_av[l2[j]][l1[i]] = df.mean()[l2[j] + "." + l1[i]]
                    #df_std[l1[i]][l2[j]] = df.std()[l2[j] + "." + l1[i]]
            else:
                df_av[l2[j]][l1[i]] = df.mean()[l2[j] + "." + l1[i]]
                #df_std[l2[j]][l1[i]] = df.std()[l2[j] + "." + l1[i]]

    #print(df_av)
    #print(1-df_av)
    # return size-dataframes

    # new_l1, new_l2 = get_ordered_names(df_av, l1=l1, l2=l2)
    # df_av = df_av[new_l1]
    # df_av = df_av.reindex(new_l2)
    # df_std = df_std[new_l1]
    # df_std = df_std.reindex(new_l2)

    return df_av

#helper for the function above
def get_upper_values_from_matrix(matrix):

    values=[]
    for i in range(matrix.shape[0]):
        for j in range(i+1,matrix.shape[1]):
            values.append(matrix[i][j])

    return values
def get_upper_values_from_matrix_wdiagonal(matrix):

    values=[]
    for i in range(matrix.shape[0]):
        for j in range(i,matrix.shape[1]):
            values.append(matrix[i][j])

    return values

#return dataframe with a different order of columns
def get_ordered_names(df,**kwargs):
    l1 = kwargs['l1']
    l2 = kwargs['l2']
    corr_sum = np.sum(np.absolute(df), axis=1)
    new_l2 = corr_sum.sort_values(ascending=False).index.tolist()
    if (l1==l2):
        new_l1=new_l2
    else :
        corr_sum = np.sum(np.absolute(df), axis=0)
        new_l1 = corr_sum.sort_values(ascending=False).index.tolist()
    return new_l1, new_l2
#normalize (mean normalization) or standardize database
def standardize(df):
    return (df - df.mean()) / df.std()
def mynormalize(df):
    return (df - df.mean()) / (df.max() - df.min())
    #return (df - df.min()) / (df.max() - df.min())

#networks
#get network of stability metrics
def get_network_of_metrics_gml(sdf, measures, namepart ,Link_Treshold=0,measures_to_erase=[]):
    positions = nx.read_pajek("positions_new.net")
    Metrics = get_network_of_metrics(sdf,Link_Treshold)
    Metrics.add_nodes_from(measures)
    Metrics.remove_nodes_from(measures_to_erase)
    for node in Metrics:
        try:
            Metrics.node[node]['x'] = positions.nodes[node]['x']
            Metrics.node[node]['y'] = positions.nodes[node]['y']
        except:
            Metrics.node[node]['x'] = rnd.randint(1,20)
            Metrics.node[node]['y'] = rnd.randint(1,20)


    filename="../OUTPUT/metric_network_%s.gml" % (namepart)
    nx.write_gml(Metrics, filename)

    return Metrics

#helper function for the function above
def get_network_of_metrics(sdf,Link_Treshold=0):

    #for each column, split it it into the two metrics, and create the links
    G = nx.Graph()

    ##
    #to print also a table with the mean and std


    positions=nx.read_pajek("positions_new.net")
    for namepair in list(sdf):
        name=namepair.split(".")
        if np.abs(sdf.mean()[namepair]) > Link_Treshold :
            G.add_edge(name[0],name[1], weight=np.abs(sdf.mean()[namepair]), sign=(sdf.mean()[namepair]/np.abs(sdf.mean()[namepair])), std=np.abs(sdf.std()[namepair]))
            try:
                G.node[name[0]]['x'] = positions.nodes[name[0]]['x']
                G.node[name[0]]['y'] = positions.nodes[name[0]]['y']
            except:
                G.node[name[0]]['x'] = rnd.randint(1,10)
                G.node[name[0]]['y'] = rnd.randint(1,10)
            try :
                G.node[name[1]]['y'] = positions.nodes[name[1]]['y']
                G.node[name[1]]['x'] = positions.nodes[name[1]]['x']
            except:
                G.node[name[1]]['x'] = rnd.randint(1, 10)
                G.node[name[1]]['y'] = rnd.randint(1, 10)



    return G
#get lumped network to study inter-group connection intensity
def get_lumped_network(group_dict,big_net,namepart):


    net=nx.Graph()

    #dictionary with the group of each metric
    node_group={}

    net.add_nodes_from(group_dict.keys())
    #añadimos la lista de medidas que hay en cada nodo
    for node in net:
        net.nodes[node]['nodes']=group_dict[node]['nodes']
        net.nodes[node]['corr']=0
        net.nodes[node]['count']= 0
        for metric in group_dict[node]['nodes']:
            node_group[metric]=node

    #print(node_group)
    #print(net.nodes())
    #añadimos las correlaciones absolutas que hay entre grupos
    for edge in big_net.edges():
        #if two nodes in same group: adds to self correlation
        #print(edge[0],node_group[edge[0]],edge[1],node_group[edge[1]])
        if (node_group[edge[0]]==node_group[edge[1]]):
            #print("same group!")
            net.nodes[node_group[edge[0]]]['corr'] += big_net.edges[edge]['weight']
            net.nodes[node_group[edge[0]]]['count'] += 1
        #if teo nodes in differnt groups, adds to links correlation
        else :
            #print("different group!")
            try:
                net.edges[(node_group[edge[0]],node_group[edge[1]])]['weight'] += big_net.edges[edge]['weight']
                net.edges[(node_group[edge[0]],node_group[edge[1]])]['count'] += 1
            except:
                net.add_edge(node_group[edge[0]], node_group[edge[1]], weight=big_net.edges[edge]['weight'] )
                net.add_edge(node_group[edge[0]], node_group[edge[1]], count=1)

    for node in net:
        net.nodes[node]['corr'] = net.nodes[node]['corr']/net.nodes[node]['count']
        #print(net.nodes[node])

    for edge in net.edges:
        net.edges[edge]['weight'] = net.edges[edge]['weight']/net.edges[edge]['count']
        #print(edge, net.edges[edge])

    filename = "../OUTPUT/mergedmetrics_%s.gml" % (namepart)
    nx.write_gml(net, filename)

    return net
def get_most_corr_metrics_of_group(group_dict,net,coph_df,namepart):
#get the metric that is the most correlated to all tohers inside same group
    lol_influences=[]
    contributors={}
    isolated ={}
    groups_n=list(group_dict.keys())

    for index in groups_n : #in each group
        metrics=group_dict[index]['nodes']
        #print("metrics:%s, group=%s"% (metrics, index))
        influent_metrics=[]
        isolated_metrics=[]
        max_influence=0
        min_influence=10000
        influent_metric=0
        isolated_metric=0
        for m1 in metrics: #for each metric
            influence = 0.
            cophenet = 0.
            for m2 in metrics: #for each of its links inside the group
                try:
                    influence += net[m1][m2]['weight']
                    cophenet += coph_df[m1][m2]
                    #print(m1,m2,net[m1][m2]['weight'])
                except:
                    pass
            if (influence > max_influence):
                max_influence=influence
                influent_metric=m1
            if (influence < min_influence):
                min_influence=influence
                isolated_metric=m1
         #   print("%s influence is %f" % (m1,influence))
            lol_influences.append([m1,influence,index,cophenet])
        #print("max influence is %f" % max_influence)
        #contributors[influent_metric]=groups_n[index]
        influent_metrics.append(influent_metric)
        isolated_metrics.append(isolated_metric)

        contributors[groups_n[index]]=influent_metrics
        isolated[groups_n[index]]=isolated_metrics

    influencesDF=pd.DataFrame(lol_influences,columns=['metric','influence','group','Ddistance'])

    return contributors, isolated, influencesDF

####### FIGURES ##########
##FIGURE 1
def plot_figure_1(sdf):
    filename="../OUTPUT/Figure1.pdf"

    name3 = 'Rinf.<s_ij>'
    name2 = 'Is.tmax'  # decrease
    name4 = 'RM.SM'  # change
    name1 = 'TM.<TM>'  # remain over
    names = [[name1, name2], [name3, name4]]
    place = [['lower right', 'lower right'], ['lower right', 'lower right']]
    nrows = 2
    ncol = 2

    # row and column sharing
    sns.set_style("white")
    f, axarr = plt.subplots(nrows, ncol, sharex='col', sharey='row')
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    f.add_subplot(111, frameon=False)
    for i in range(nrows):
        for j in range(ncol):

            axarr[i, j].set_ylim(-1., 1.)

            try:
                rename = names[i][j].split(".")

                label = "%s vs %s" % (rename[0], rename[1])
            except:
                renamea = names[i][j][0].split(".")

                renameb = names[i][j][1].split(".")

                label = "%s vs %s\n\n\n\n\n\n %s vs %s" % (renamea[0], renamea[1], renameb[0], renameb[1])

            sdf.plot(ax=axarr[i, j], y=names[i][j], legend=False, fontsize="12")

    sns.set()
    plt.xlabel("Community Size (species nb.)", fontsize=14)
    plt.ylabel(r"Spearman's correlation coefficient ($\rho$)", fontsize=14)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    text(0.05, 0.9, 'A)', transform=axarr[0, 0].transAxes, fontsize=14)
    text(0.05, 0.9, 'B)', transform=axarr[0, 1].transAxes, fontsize=14)
    text(0.05, 0.9, 'C)', transform=axarr[1, 0].transAxes, fontsize=14)
    text(0.05, 0.9, 'D)', transform=axarr[1, 1].transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(filename)

    return
def plot_figure_1_supp(sdf,measures):

    #Clasify all correlations in 4 cathegories: CTE,INCR, DECR, CROSSING
    #sdf=sdf_1.rolling(10).mean()
    all_corr=list(sdf) #correlation names
    types_df=pd.DataFrame(index=all_corr)
    types_df['class']=-1 #all correlations unassigned

    change_tresh=0.1


    cte=( np.fabs(sdf[0:10].mean() - sdf[80:91].mean()) <change_tresh)
    #pairwise correlations similar in small and large communities
    change1=(~cte) #pairwise correlations that change in small vs large communities

    ctemedbig=(np.fabs(sdf[45:55].mean() - sdf[80:91].mean()) < change_tresh) #pairwise correlations that are similar in medium and large communities

    decrease=(np.fabs(sdf[80:91].mean())< np.fabs(sdf[0:11].mean())) #correlations that decrease in strength
    increase=(np.fabs(sdf[80:91].mean())>np.fabs(sdf[0:11].mean())) #correlations that increase in strength


    # print(increase.sum())
    change_sign=((np.sign(sdf[0:11].mean()) != np.sign(sdf[80:91].mean())) & (np.fabs(sdf[80:91].mean())>0.1) & (np.fabs(sdf[0:11].mean())>0.1)) #correlations that clearly change sign
    positive=(sdf[40:51].mean() - sdf[40:51].std()) > 0 #positive correlation in medium-sized communities
    negative=(sdf[40:51].mean() + sdf[40:51].std()) < 0 #negative correlation in medium-sized communities

    types_df[cte]=1 #correlations that remain constat at all sizes
    types_df[(change1 & change_sign & (~ctemedbig))]=5 #correlations that change at all sizes and change sugn
    big_change=(change1 & change_sign & (~ctemedbig))
    types_df[(change1 & (~ctemedbig)& (~change_sign) & increase)]=6 #constantly increasing correlations
    big_incr=(change1 & (~ctemedbig)& (~change_sign) & increase)
    types_df[(change1 & (~ctemedbig)& (~change_sign) & decrease)]=7 #constantly decreasing correlations
    big_decr=(change1 & (~ctemedbig)& (~change_sign) & decrease)
    types_df[(change1 & change_sign & ctemedbig)]=2 #change size but remain constant in large networks
    med_change=(change1 & change_sign & ctemedbig)
    types_df[(change1 & ctemedbig & (~change_sign) & increase)]=3 #increase and remain constant
    med_incr=(change1 & ctemedbig & (~change_sign) & increase)
    types_df[(change1 & ctemedbig & (~change_sign) & decrease)]=4 #decrease and remain constant
    med_decr=(change1 & ctemedbig & (~change_sign) & decrease)

    type_matr=pd.DataFrame(columns=measures,index=measures)

    type_matr.rename(index=rename_dict,columns=rename_dict,inplace=True)

    type_matr=type_matr.astype(float) #dataframe for type of correlation

    for namepair in all_corr:
        name1,name2=namepair.split(".")
        type_matr.loc[name1,name2]=types_df.loc[namepair].values[0]
        type_matr.loc[name2,name1]=types_df.loc[namepair].values[0]

    #coloring the matrix of pairwise correlation types.
    cmap = matplotlib.colors.ListedColormap(['#FFFFFF','#B8B8B8','#ddccff','#FFCCCC','#CCE5FF','#cc99ff','#FF6666','#66B2FF'])
    boundaries = [1, 2, 3, 4, 5, 6, 7, 8]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    di={ 1: "G", 2: "D" , 3: "E", 4: "F", 5: "A" ,6 : "B", 7: "C" }
    mask = np.zeros_like(type_matr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    filename = "../OUTPUT/FigS1_ii.pdf"
    fig = plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 19})
    sub1 = fig.add_subplot(111)
    heatmap = sns.heatmap(type_matr.rename(columns=final_rename_dict,index=final_rename_dict), vmax=7., vmin=0, annot=type_matr.replace(di), square=True, cbar=False, cmap=cmap, mask=mask, fmt='')
    loc, labels = plt.xticks()
    heatmap.set_xticklabels(labels, fontsize=22, rotation=55)
    heatmap.set_yticklabels(labels, fontsize=24, rotation=0)  # reversed order for y
    # heatmap.set_yticklabels(labels[::-1], fontsize=15, rotation=0)  # reversed order
    bottom, top = sub1.get_ylim()
    sub1.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(filename, bbox_inches='tight')
    #################################################

    filename = "../OUTPUT/FigS1_i.pdf"
    title=['A','B','C','D','E','F','G']
    criteria=[big_change,big_incr,big_decr,med_change,med_incr,med_decr,cte]
    sns.set_style("whitegrid")
    #sns.set_style("white")

    fig = plt.figure(figsize=(50, 40))
    plt.rcParams.update({'font.size': 10})
    f, axarr = plt.subplots(3, 3, sharex='col', sharey='row')
    axarr=axarr.ravel()
    for i in range(len(title)):
        axarr[i].set_ylim(-1,1)
        axarr[i].set_xlim(0,100)
        axarr[i].set_title(title[i],fontsize="12")

        sdf.loc[0:95,criteria[i]].rolling(6).mean().plot(ax=axarr[i],legend=False,fontsize="12",alpha=0.5,linewidth=1.)
        percentage=(criteria[i].sum()/351.)*100
        label = "%s%%" % (((criteria[i].sum()/351.)*100).round(2))
        axarr[i].annotate(label, xy=(0.75, 0.1), xycoords='axes fraction', size=10, )
        axarr[i].set_facecolor((1.0, 1.0, 1.0))


    axarr[3].set_ylabel(r"Spearman's $\rho$", fontsize=12)

    axarr[7].set_xlabel("Community Size", fontsize=12)


    plt.savefig(filename, bbox_inches='tight')

    return

#plot the dendrogram of cluster of metrics for a given correlation_matrix
# FIGURE2, S3 and S4
def plot_dendrogram(signed_corr_mat,sizename,group_filename,rename_dict):

    corr_mat=np.abs(signed_corr_mat)
    corr_mat = corr_mat[corr_mat.columns].astype(float)

    method="average"
    Z = linkage(corr_mat, method=method, metric='correlation')  # get dendrogram of linkage according to different "distances"
    c, coph_dists = cophenet(Z, pdist(corr_mat))  # get goodness fo fit
    print("Dendrogram Cophenet distance of %s :%f" % (method,c))
    #coph_dists has the distance between each pair of emtrics based on the dendrogram
    #to compare with the disntaces based con correlation we need to retain the upper trigular matrix of corr_mat
    a=np.triu(corr_mat,1).flatten()
    a = a[a != 0]
    corr_dists = a
    coph_D=np.zeros((len(list(corr_mat)),len(list(corr_mat))))
    indices = np.triu_indices(len(list(corr_mat)),1)
    coph_D[indices]= coph_dists
    indices = np.tril_indices(len(list(corr_mat)), -1)
    coph_D[indices]= coph_dists
    coph_df=pd.DataFrame(columns=list(corr_mat), index=list(corr_mat), data=coph_D)

    plt.figure(figsize=(25, 10))
    labelsize=20
    ticksize=15
    title="Hierarchical Clustering Dendrogram %s" % (sizename)
    plt.title(title, fontsize=labelsize)
    plt.xlabel('stock', fontsize=labelsize)
    plt.ylabel('distance', fontsize=labelsize)

    #for coloring groups
    #pal = sns.husl_palette(8, s=.80)
    #pal = [ matplotlib.colors.to_hex(pal[i]) for i in range(len(pal))]
    pal=['#ccd739','#66af31', '#66af31', '#4690cf']
    if sizename=="rSMA" :
        pal = ['#ccd739','#a6a6a6', '#66af31', '#4690cf']

    set_link_color_palette(pal)


    ##color_threshold=0.3,
    den=dendrogram(
        Z,
        leaf_rotation=0,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        orientation="top",
        color_threshold=0.3,
        above_threshold_color='grey',
        labels = corr_mat.columns
        #labels = [rename_dict[i] for i in corr_mat.columns]

    )
#    filename = "../OUTPUT/leafs_%s.pdf" % (sizename)
#    plt.tight_layout()
#    plt.savefig(filename)
    #get color of different clusters
    cluster_dict=get_cluster_classes(den)
    colordf = pd.Series(index=list(corr_mat))
    colordf = colordf.rename(index=rename_dict)

    # color_dict=invert_dict(cluster_dict)
    # #print(color_dict)    #
    # for col in colordf.index:
    #      colordf[col]=color_dict[col][0]

    #or get color from groups
    #print("reading %s\n", group_filename)
    groups = read_groups_from_graphml(group_filename)
    pal = ['#ccd739','#a6a6a6', '#4690cf', '#66af31']

    set_link_color_palette(pal)
    cluster_dict={}
    for node in groups.keys():
        cluster_dict[node]=groups[node]['nodes']
    color_dict=invert_dict(cluster_dict) #this gives a number
    for col in colordf.index:
        try:
            colordf[col]=pal[color_dict[col][0]]
        except:
            colordf[col]="#000000"

    yticks(fontsize=ticksize)
    xticks(rotation=-90, fontsize=ticksize)
    filename = "../OUTPUT/FigS3_B_%s.pdf" % (sizename)
    #print("filename")
    plt.tight_layout()

    #cmap=vlag, PuOr,coolwarm
    cmap="vlag"

    g=sns.clustermap(corr_mat.rename(index=final_rename_dict,columns=final_rename_dict),row_colors=colordf.rename(final_rename_dict),col_colors=colordf.rename(final_rename_dict),method=method,metric="correlation",vmax=1., vmin=-1.,cmap=cmap,linewidths=.75, figsize=(13, 13))

    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=-270)
    plt.savefig(filename)

    get_signed_matrix=True
    if (get_signed_matrix):

        plt.figure(figsize=(25, 10))
        labelsize = 20
        ticksize = 15
        title = "Heatmap %s" % (sizename)
        plt.title(title, fontsize=labelsize)
        #rename(index=rename_dict,columns=rename_dict)
        order=den['ivl']
        #print(order)
        #print(signed_corr_mat.head())
        signed_corr_mat=signed_corr_mat.reindex(index=order,columns=order) #only if not translated before!
        #print(signed_corr_mat.head())
        heat=sns.heatmap(signed_corr_mat.rename(index=final_rename_dict,columns=final_rename_dict), vmax=1., vmin=-1, center=0, annot=False , square=True,  cbar=False, cmap=cmap)
        #plt.show()
        #heat.yaxis.tick_right()
        yticks(rotation=0,fontsize=ticksize)
        xticks(rotation=90, fontsize=ticksize)

        filename = "../OUTPUT/FigS4_%s.pdf" % (sizename)
        plt.tight_layout()
        plt.savefig(filename)

    return order, coph_df
#helper functions for the plot_dendrogram
def get_cluster_classes(den, label='ivl'):
    cluster_idxs = collections.defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes
def read_groups_from_graphml(filename):

    groups={}

    N=nx.read_graphml(filename)

    for node in N.nodes:
        group=N.nodes[node]['Modularity Class']
        #label=rename_dict[N.nodes[node]['label']]
        label=N.nodes[node]['label']
        try: #if group exists we fill with a new node
            groups[group]['nodes'].append(label)
        except:#if the group does not exists, we create it
            groups[group]={}
            groups[group]['nodes']=[label]
            groups[group]['corr']=0
            groups[group]['ind_contrib']=[]


    return groups
#FIGURE S4iv
def plot_violin(sdf_all,significance) :

    #sign study:
    sma_neg_c=((sdf_all[0:11].mean()  + sdf_all[0:11].std()  <0))
    sma_pos_c=((sdf_all[0:11].mean()  - sdf_all[0:11].std()  >0))
    med_neg_c=((sdf_all[40:51].mean() + sdf_all[40:51].std() <0))
    med_pos_c=((sdf_all[40:51].mean() - sdf_all[40:51].std() >0))
    big_neg_c=((sdf_all[80:91].mean() + sdf_all[80:91].std() <0))
    big_pos_c=((sdf_all[80:91].mean() - sdf_all[80:91].std() >0))

    sma_neg_c = ((sdf_all[0:11].mean()  < 0))
    sma_pos_c = ((sdf_all[0:11].mean()  > 0))
    med_neg_c = ((sdf_all[40:51].mean() < 0))
    med_pos_c = ((sdf_all[40:51].mean()  > 0))
    big_neg_c = ((sdf_all[80:91].mean()  < 0))
    big_pos_c = ((sdf_all[80:91].mean()  > 0))

    Percentage=[(sma_neg_c.sum()/351)*100, (sma_pos_c.sum()/351)*100,(med_neg_c.sum()/351)*100, (med_pos_c.sum()/351)*100,(big_neg_c.sum()/351)*100, (big_pos_c.sum()/351)*100]

    label=[]

    for i in range(len(Percentage)): #= "%s vs %s\n\n\n\n\n\n %s vs %s" % (renamea[0], renamea[1],renameb[0],renameb[1])
        label.append("%s%%" % np.round(Percentage[i],2))

    ini_relevant = ((np.abs(sdf_all[0:11].mean()) + sdf_all[0:11].std()) > significance)
    fin_relevant = ((np.abs(sdf_all[80:91].mean()) + sdf_all[0:11].std()) > significance)
    not_relevant = ((np.abs(sdf_all.mean()) + sdf_all.std()) < significance)

    sdf=sdf_all.loc[:,(ini_relevant | fin_relevant) ]

    fig = plt.figure(figsize=(12, 9))
    sdf["size_cat"]= divmod(sdf.index/10, 1)[0]
    dict_names={0.0:'0-10' , 1.0:'10-20', 2.0:'20-30' , 3.0:'30-40' , 4.0:'40-50' , 5.0:'50-60' ,
                6.0:'60-70', 7.0: '70-80', 8-0:'80-90' , 9.0:'90-100'}
    sdf["size_cat"]=sdf["size_cat"].map(dict_names)
    sdf_cat=sdf.groupby("size_cat").mean().stack().reset_index()
    sns.swarmplot(x='size_cat', y=0, data=sdf_cat, palette="viridis")
    plt.xlabel("Community Size",fontsize=18)
    plt.ylabel(r"Spearman's $\rho$", fontsize=18)

    plt.annotate(label[0], xy=(0.2,-0.5), xycoords='data', size=18)
    plt.annotate(label[1], xy=(0.2, 0.9), xycoords='data', size=18)
    plt.annotate(label[2], xy=(4.2, -0.5), xycoords='data', size=18)
    plt.annotate(label[3], xy=(4.2, 0.9), xycoords='data', size=18)
    plt.annotate(label[4], xy=(7.2, -0.5), xycoords='data', size=18)
    plt.annotate(label[5], xy=(7.2, 0.9), xycoords='data', size=18)


    filename="../OUTPUT/FigS4_iv.pdf"
    plt.tight_layout()
    plt.savefig(filename)

    return
#FIGURE S5
def print_corr_matrix(metrics, size_name,order):
    #change names in order to newe names
    #new_order=[rename_dict[x] for x in order]
    new_order=order
    #print(new_order)
    corr_df = nx.to_pandas_adjacency(metrics, weight="weight").round(decimals=2)
    #print(corr_df)
    sd_df = nx.to_pandas_adjacency(metrics, weight="std").round(decimals=2)
    annot_df = corr_df.applymap(str) + " (" + sd_df.applymap(str) + ")"
    annot_df = annot_df.reindex(index=new_order, columns=new_order)
    corr_df = corr_df.reindex(index=new_order, columns=new_order)
    #signed_corr_mat = signed_corr_mat.
    # print(signed_corr_mat.head())
    #heat = sns.heatmap(signed_corr_mat.rename(index=rename_dict, columns=rename_dict), vmax=1., vmin=-1, center=0,
     #                  annot=False, square=True, cbar=False, cmap=cmap)
    #print(corr_df)
    #print("----")
    #print(sd_df)

    filename="../OUTPUT/FigS5_%s.pdf" %(size_name)
    fig = plt.figure(figsize=(55, 20))
    plt.rcParams.update({'font.size': 18})
    sub1 = fig.add_subplot(111)
    heatmap=sns.heatmap(corr_df.rename(columns=final_rename_dict,index=final_rename_dict), vmax=1., center=0 , square=False, annot=annot_df,cbar=False,cmap="coolwarm", fmt = '')
    loc, labels = plt.xticks()
    heatmap.set_xticklabels(labels, fontsize=20, rotation=45)
    heatmap.set_yticklabels(labels, fontsize=20, rotation=0) # reversed order for y
    #heatmap.set_yticklabels(labels[::-1], fontsize=15, rotation=0)  # reversed order

    plt.savefig(filename, bbox_inches='tight')

#FIGURE S6 Explained variance
def get_exp_var_ind(covmat,Rank_df):

    #get equivalente of metric-index in df
    MtoN=pd.Series(Rank_df.index, index=Rank_df['metric'])
    #print(MtoN)
    covmat_T=covmat.values.diagonal().sum()
    #print(covmat_T)
    for var in list(covmat):
        Rank_df.loc[MtoN[var],"exp_var"]=np.power(covmat[var],2).sum()/(covmat_T*covmat[var][var])
        Rank_df.loc[MtoN[var],"simple_exp_var"]=covmat[var][var]/covmat_T
        Rank_df.loc[MtoN[var],"diagonal"]=covmat[var][var]
    return Rank_df

#Analysis of explained variance
def plot_exp_var_bars_ind( Rank, namepart, addname):

    #normalize diagonal to obtain variance individual
    Rank['diagonal_norm'] = Rank['diagonal'] / Rank['diagonal'].sum()

    # create new dataframes in correct order
    plot_df = Rank.sort_values(['exp_var'], ascending=[False])[['metric', 'exp_var', 'diagonal_norm', 'group']]
    plot_df.reset_index(inplace=True)

    #create color dictionary
    dict_color = {0: '#ccd739', 1: '#808080', 3: '#66af31', 2: '#4690cf'}
    mycolor = [dict_color[x] for x in plot_df['group']]

    filename = "../OUTPUT/FigS6_%s%s.pdf" % (namepart,addname)
    plt.rcParams.update({'font.size': 12})
    # fig = plt.figure(figsize=(12, 4.5))
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7.2, 6.5))
    fig.suptitle(namepart)
    # first spearmans'
    plot_df['metric']=[final_rename_dict[x] for x in plot_df['metric']]
    #print(plot_df)
    sub1 = axs
    sub1 = plt.subplot(1, 1, 1)
    sub1 = sns.barplot(x="metric", y="exp_var", palette=mycolor, data=plot_df)
    # plt.xticks(rotation='vertical')
    sub1 = sns.barplot(x="metric", y="diagonal_norm", color='red', data=plot_df)
    sub1.set_ylabel('Explained variance')
    sub1.set_ylim([0, 0.3])
    plt.xticks(rotation='vertical',fontsize=15)
    #plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)

    return
def measure_exp_variance_of_3_groups(Rank,size,size_name,addname):

    #we need to randomly select 3 metrics, one from each group:
    n_repetitions=500
    n_metrics=3

    Exp_Var=[]

    rnd_metrics_index=[0]*n_metrics
    for repetition in range(n_repetitions):

        #choose 1 random metric from each group
        rnd_metrics_index[0] = rnd.randint(0,2)
        rnd_metrics_index[1] = rnd.randint(3, 13)
        rnd_metrics_index[2] = rnd.randint(14, 22)
        #print("rnd_indixs:%s" % rnd_metrics_index)
        #get the added explained variance of those 3
        exp_var=0
        for m_index in rnd_metrics_index:
            exp_var += Rank.loc[m_index,"exp_var"]

        #print(exp_var)
        Exp_Var.append(exp_var)

    #print(np.array(Exp_Var).mean())
    #print(np.array(Exp_Var).std())

    #and get maximum and minimum
    #max:
    filename = "../OUTPUT/Variance.txt"
    if (size_name=="rSMA") :
        f = open(filename, "w+")
    else:
        f = open(filename, "a")

    idx = Rank.groupby(['group'])['exp_var'].transform(max) == Rank['exp_var']
    max_var_metrics = list(Rank[idx]['metric'])  # metric of eahc group with most variance explained
    max_var_exp = Rank[Rank['metric'].isin(max_var_metrics)]['exp_var'].sum()
    f.write("size=%s\n" % size_name)
    f.write("most var explained=%s\n" % max_var_exp)
    #min
    idx = Rank.groupby(['group'])['exp_var'].transform(min) == Rank['exp_var']
    min_var_metrics = list(Rank[idx]['metric'])  # metric of eahc group with least variance explained
    min_var_exp = Rank[Rank['metric'].isin(min_var_metrics)]['exp_var'].sum()
    f.write("min var explained=%s\n" % min_var_exp)
    f.write("average var explained=%s +- %s\n" % (np.array(Exp_Var).mean(), np.array(Exp_Var).std()))
    #most correlated metrics
    idx = Rank.groupby(['group'])['influence'].transform(max) == Rank['influence']
    max_corr_metrics = list(Rank[idx]['metric'])  # metric of eahc group with most variance explained
    mc_var_exp = Rank[Rank['metric'].isin(max_corr_metrics)]['exp_var'].sum()
    f.write("most correlated metrics var explained=%s\n" % (mc_var_exp))

    f.close()
    return

# FIG S7 Covariance Ellipsoid
def Plot_difference_HV(cov_all,Ranking_all,group_dict,addname):
    #metrics to consider, and repetitions to select random metrics
    n_metrics=3
    n_repetitions=2000
    #3 sizesd to cover
    column_names = ['SMA', 'LAR', 'MED']
    #to store different HV obtained when using only "nmetric" metric of each group
    HV_groups=[0]*len(list(group_dict.keys()))
    HV_std_groups = [0]*len(list(group_dict.keys()))
    EV_groups=[0]*len(list(group_dict.keys()))
    #in each size
    for size in range(len(column_names)): #len(column_names)
        #print("size %s " % (column_names[size]))
        Ranking=Ranking_all[size]
        cov=cov_all[size]
        MtoG=pd.Series(Ranking['group'].values, index=Ranking['metric'])

        #get n_metrics from each of the groups and meadure HV
        for group in list(group_dict.keys()):
            metrics_in_group=group_dict[group]['nodes']
            metric_out=['<RE>']
            for metric in metric_out:
                try:
                    metrics_in_group.remove(metric)
                except:
                    pass
            HV=[]
            EIG=[]
            if ( len(metrics_in_group) > n_metrics):
                #make a random selection
                for repetition in range(n_repetitions):
                    metrics=metrics_in_group.copy()
                    selected_metrics=[]
                    for selection in range(n_metrics): #select 3 random metrics
                        index= rnd.randint(0, len(metrics) - 1)
                        selected_metrics.append(metrics.pop(index))

                    HV_temp, eig_v_temp=calculate_HV_covellipsoid(cov.loc[selected_metrics,selected_metrics])
                    #print("metrics from group % are %s, HV:%s" % (group, selected_metrics,HV_temp))
                    HV.append(HV_temp)
                    #EIG.append(eig_v_temp)

            else:
                #just take the three
                selected_metrics=metrics_in_group
                #HV.append(calculate_HV_covellipsoid(cov[selected_metrics]))
                HV_temp, eig_v_temp =calculate_HV_covellipsoid(cov.loc[selected_metrics,selected_metrics])
                #print("metrics from group % are %s, HV:%s" % (group, selected_metrics, HV_temp))
                HV=[HV_temp]*n_repetitions
                #EIG.append(eig_v_temp)
                #EIG.append(eig_v_temp)

            #av_HV = np.array(HV).mean()
            #std_HV = np.array(HV).std()
            HV_groups[group]=HV.copy()
            #HV_std_groups[group]=std_HV
            #eig_df=pd.DataFrame(EIG)
            # print("eig_df: %s" % eig_df)
            # EV_groups[group]=eig_df.mean().values
            # print("eig_mean %s" % eig_df.mean())

        #now selecting one metric from each group
        HV_group_constr=[]
        EIG_group_constr=[]
        for repetition in range(n_repetitions):
            selected_metrics = []
            for group in [0,2,3]:
                metrics_in_group=group_dict[group]['nodes'].copy()
                for metric in metric_out:
                    try:
                        metrics_in_group.remove(metric)
                    except:
                        pass
                index= rnd.randint(0, len(metrics_in_group) - 1)
                selected_metrics.append(metrics_in_group.pop(index))

            HV_temp, eig_v_temp=calculate_HV_covellipsoid(cov.loc[selected_metrics,selected_metrics])
            #print("metrics are %s, HV:%s" % (selected_metrics, HV_temp))
            HV_group_constr.append((HV_temp))
            #EIG_group_constr.append(eig_v_temp)

        eig_group_constr_df=pd.DataFrame(EIG_group_constr)
        EIG_group=eig_group_constr_df.mean().values
        #av_HV_group_constr =np.array(HV_group_constr).mean()
        #std_HV_group_constr=np.array(HV_group_constr).std()

        #buil dataset to represent and work with results
        HV_df=pd.DataFrame(columns=list(group_dict.keys())+['3D'])
        for group in list(group_dict.keys()):
            HV_df[group] = HV_groups[group]
        HV_df['3D']= HV_group_constr

        #print(HV_df.head())
        HV_df.drop([1], axis=1,inplace=True) #not show group 1
        #print(HV_df.head())

        #print(HV_df.mean()/0.8061)
        #print(HV_df.std()/0.8061)

        #plot bars
        fig, ax = plt.subplots()
        #names=list(HV_df[[]])
        names=['Early R.', 'Dist.', 'Sens. ','3 Groups']
        x_pos=np.arange((len(names)))
        CTEs = HV_df.mean().values/0.8061
        error = HV_df.std().values/0.8061
        #HV_df.boxplot(ax=ax)
        ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        #ax.axhline(y=0.8061, linewidth=0.8, color='grey', ls='--')
        ax.set_ylabel('Cov. Ellipsoid HV')
        ax.set_ylim([0,1])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names)
        ax.set_title(column_names[size])
        ax.yaxis.grid(True)

        plt.tight_layout()
        filename="../OUTPUT/FigS7_%s%s.pdf" % (column_names[size], addname)
        plt.savefig(filename)
        plt.close()

    return
#helper for Ellipsoid HV
def calculate_HV_covellipsoid(covmat):

    #as we have the covariance matrix, we only need to diagonalizze and apply the fomula for obtaining the volume from eigenvalues
    eig_vals, eig_vec = np.linalg.eig(covmat)
    eig_vals = eig_vals / eig_vals.sum()  # normalize
    dim = len(list(covmat))
    Cn = (2 / dim) * (pi ** (dim / 2) / gammaf(dim / 2))
    #print("Cn=%s eigv=%s" % (Cn,eig_vals))
    v=np.array(eig_vals)
    v=sqrt(v)
    #print(v)
    # multiply by sqrt of eigenvalues
    volume = Cn * (np.prod(v))
    #print("volumen=%s" %volume)
    return volume, np.sort(eig_vals)/eig_vals.max()

#Helper funtions for correlations
def corrfunc(x, y, axes, **kws):
    r, p = stat.pearsonr(x, y)
    ax = axes
    ax.annotate("r² = {:.2f} p= {:.2f}".format(r*r,p), xy=(.1, .9), xycoords=ax.transAxes)
def corrfunc_spear(x, y, axes, **kws):
    r, p = stat.spearmanr(x, y)
    ax = axes
    ax.annotate("r = {:.2f} p= {:.2f}".format(r, p), xy=(.1, .9), xycoords=ax.transAxes)

####### OTHER HELPERS #############
#return an array with th upper triangular aprt of a matrix
#Dictionary of column names:
rename_dict={'L1DBRMt_s' : "S" , 'absumSNorm' : "<s_ij>" ,
             'RM_p': "RM", 'L1DBRM_p': "SM" , 'avL1DBRM_i': "<SM>" , 'maxSensM_i': "maxSM",
             'TM_p': "TM", 'minTM_i': "minTM" , 'av_TM_i': "<TM>",
             'DBRobst' : "<RE>",'maxDBRobst': "maxRE",
             'Next2nd': "<CE>",'maxNext2nd': "maxCE",
             'Ext_half': "TE",
             'L1DBRobst': "<SE>",'maxL1DBRobst': "maxSE" ,
             'avRM_i': "<RM>",'maxRM_i': "maxRM",
             'R0':"R0", 'Is':"Is", 'Amax':"Amax", 'tmax':"tmax",'Rinf':"Rinf",
             'MR0':"MR0", 'MAmax':"MAmax",'Mtmax':"Mtmax",'MRinf':"MRinf",
             'RMt_s':"Rs",'RMt_p':"Rp", 'L1DBRMt_p':"Sp" ,
             'leading': 'C','isolated': "C inv." ,'maxvarexp': "V", 'minvarexp':               'V inv.',
             'infl_value' :"C Value",'infl_rank': "C rank",'expvar_rank':"V rank",'expvar_value':"V value", "influence": "Total correlation", "exp_var":"Explained variability"
             }
final_rename_dict={
             'RM':"$RM^G$", '<RM>':"$<RM^L>$", 'maxRM':"$RM^L_{max}$",
             'SM':"$SM^G$", '<SM>':"$<SM^L>$", 'maxSM':"$SM^L_{max}$",
             'TM':"$TM^G$", 'minTM':"$TM^L_{min}$", '<TM>':"$<TM^L>$",
             '<RE>':"$<RE>$", 'maxRE':"$RE_{max}$",
             '<SE>':"$<SE>$", 'maxSE':"$SE_{max}$",
             '<CE>':"$<CE>$", 'maxCE':"$CE_{max}$",
             'TE':"$<TE>$",
             '<s_ij>':"$<s_{ij}>$",
             'Amax':"$A_{max}$", 'tmax':"$t_{max}$",
             'Rinf':"$R_{inf}$" , 'R0':"$R_{0}$" ,
             'MAmax':"$MA_{max}$",'Mtmax':"$Mt_{max}$",
             'MRinf':"$MR_{inf}$", 'MR0': "$MR_{0}$",

             'Is':"$Is$", 'S':"$S$"


}

#invert dictionary
def invert_dict(my_dict):
    new_dic = collections.defaultdict(lambda : 'X') #
    for k,v in my_dict.items():
        for x in v:
            new_dic.setdefault(x,[]).append(k)
    return new_dic
def invert_simple_dict(my_dict):
    my_inverted_dict = collections.defaultdict(list)
    {my_inverted_dict[v].append(k) for k, v in my_dict.items()}
    return my_inverted_dict

#List of 27 stability measures:
measures=['R0', 'Is', 'Amax', 'tmax', 'Rinf',
          'MR0', 'MAmax','Mtmax','MRinf',
          "S","<s_ij>" ,
          'TM',  '<TM>', "minTM",
           "<CE>",  "TE", "maxCE",
           "RM" ,"<RM>", "maxRM",
           "SM", "<SM>" , "maxSM",
           "<RE>", "maxRE",
           "<SE>", "maxSE" ]
ordered_measures=['<TM>', "SM", 'MAmax','Amax',
          'R0','Is',  'tmax', 'Rinf',
          'MR0', 'Mtmax','MRinf',
          "S","<s_ij>" ,
          'TM',   "minTM",
           "<CE>",  "TE", "maxCE",
           "RM" ,"<RM>", "maxRM",
           "<SM>" , "maxSM",
           "<RE>", "maxRE",
           "<SE>", "maxSE" ]
#negative defined metrics
negative_metrics=[
    'R0','Amax','tmax','MR0','MAmax','Mtmax',
    'L1DBRMt_s','absumS','absumSNorm',
    'RM_p', 'L1DBRM_p','RMex_p',
    'maxRM_i',
    'avRM_i', 'avL1DBRM_i',
    'maxSensM_i',
    'DBRobst', 'maxDBRobst',
    'L1DBRobst', 'maxL1DBRobst',
    'Ext_area', 'Next2nd','maxNext2nd',
    ]
nosign_metrics=[
        'RM_p','avRM_i','maxRM_i',
        'DBRobst','maxDBRobst',
        ]
maxcorr_metrics=[['MAmax', '<TM>', 'SM'],['MAmax', 'S', '<SE>'],['MAmax', 'S', '<SE>']]
###   Change in default dictionary to make it accesible as C objects ###
class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = plt.colors.rgb2hex(plt.colors.colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>'
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html

