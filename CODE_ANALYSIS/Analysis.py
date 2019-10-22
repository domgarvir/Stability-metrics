from Functions import *

########### GO TO WORKING DIRECTORY (./DATA) ####################
workingdir="../DATA"
os.chdir(workingdir)

resample_database=0
if (len(sys.argv)>1 and (sys.argv[1] == 'resample')): 
    resample_database=1

if (resample_database): #sample 100 networks of each size in the stability database
    print("resample")
    filename_DB = "Dataset_S1.csv"
    df=pd.read_csv(filename_DB,index_col=0,sep=',') #read complete database
    df=df.dropna()   #clean the dataset from posible errors
    sample_df=get_sample_df(df,df["N"],100) #random sample communities
    sample_df.to_csv("sample_db.csv") #store the sample


else : #use the sample in the main text
    filename_DB = "Dataset_S2.csv"

    sample_df=pd.read_csv(filename_DB,index_col=0)

#adjust sign of stability metrics, rename metrics name and normalize
sample_df=prepare_dataframe(sample_df,negative_metrics,nosign_metrics)
sample_df.rename(columns=rename_dict,inplace=True)
clean_sample_df=sample_df[measures]
clean_sample_df['N']=sample_df['N']

normalize=0
if ( normalize == 1 ):
    sample_df_normalized=mynormalize(sample_df)
    sample_df_normalized['N']=sample_df['N']
else:
    sample_df_normalized=sample_df

#### part 1 - Study of correlation trough a gradient of richness
TRESHOLD=0.
windowd={}
window=objdict(windowd)
window.minsize=5 #minimum size of community
window.maxsize=100 #maximum size of the community
window.width=0 #posibility of doing the size increment bigger, so network up to size+width are included
window.sample_size=100 #number of networks to sample at each size-window

#obtain pairwise correlation and covariance: for each size from 5 to 100 species (minsize,maxisze, window.width=0) sample 100 (windown.sample_size) networks and obtain correlation value, and pvalue of correlation, and covariance (spearman's) and covariance (pearsons)
corr_size , pval_size ,scov_size, pcov_size= get_full_correlation_bysize(sample_df_normalized[measures],sample_df['N'],window,"spearman")

#store the correlation trough size/read it.
filename_cor="DB_corrB.csv"
corr_size.to_csv(filename_cor)
#corr_size=pd.read_csv(filename_cor,index_col=0)
filename_cov="DB_covB.csv"
scov_size.to_csv(filename_cov)
#scov_size=pd.read_csv(filename_cov,index_col=0)
filename_pcov="DB_covP.csv"
pcov_size.to_csv(filename_pcov)
#pcov_size=pd.read_csv(filename_pcov,index_col=0)

##plot Fig. 1 and figures in Fig S1
plot_figure_1(corr_size.rolling(6).mean())
plot_figure_1_supp(corr_size,measures)

###########################################################
#### part 2 - Study stability in 3 different sizes: create network of stability, dendrograms, and find representative metrics
#size ranges: 5-15, 85-95, 45-55
size=[[5,15],[85,95],[45,55]]
size_name=["rSMA","rBIG","rMED"]

#storage of covariance matrices and rankings for 3 sizes
av_pcov_all=[]
av_scov_all=[]
s_cov_mat_all=[]
Ranking=[]
#storage of hypervolumes
HV_Rnd=[]
HV_Grp=[]


for index in range(len(size_name)): #len(size_name)
    print(size_name[index])
    #get average values and standard deviation of spearmans correlation over size range
    av_corr, sd_corr = get_av_corr_from_sdf(corr_size.loc[size[index][0]:size[index][1]], l1=measures)
    av_corr = av_corr[av_corr.columns].astype(float)
    # get average values of covariance over size range
    #rank based: spearman-like covariance
    av_scov = get_av_cov_from_sdf(scov_size.loc[size[index][0]:size[index][1]], l1=measures)
    av_scov =av_scov[av_scov.columns].astype(float)
    #value-absed: pearson like covariance
    av_pcov = get_av_cov_from_sdf(pcov_size.loc[size[index][0]:size[index][1]], l1=measures)
    av_pcov = av_pcov[av_pcov.columns].astype(float)

    av_pcov_all.append(av_pcov)
    av_scov_all.append(av_scov)

      #export stability network file for use with gephi modularity
    metrics = get_network_of_metrics_gml(corr_size.loc[size[index][0]:size[index][1]], measures, size_name[index])

    #read file from gephi with groups: This file must be generated in gephi by loading the .gml file created in the function "metrics" above, running the modularity algorithm, and exporting the result in graphml format. The name should be the one used in "group_filename" below.
    group_filename = "groups_%s.graphml" % (size_name[index])
    #obtain groups from modularity algorithm
    groups = read_groups_from_graphml(group_filename)
    #print(groups)
    #study intercorrelation among stability components: Output the merged network.
    merged_network = get_lumped_network(groups, metrics, size_name[index])

    #FIGURE 2,S3,S4,S5
    #output dendrogram and heatmap FigureS3B, FigureS4
    order, coph_dist = plot_dendrogram(av_corr,size_name[index], group_filename, rename_dict)
    #output network of metrics in matrix format
    print_corr_matrix(metrics,size_name[index],order)#FigureS5

    #study data variance
    leaders,isolated,RankDF=get_most_corr_metrics_of_group(groups,metrics,coph_dist, size_name[index]) #get most correlated metrics from each opf the groups.
    plt.close()

    #Get metrics ranked by explained variance
    RankDF_wind=get_exp_var_ind(av_pcov.loc[measures,measures],RankDF.copy())

    #now we have the explained variance, we can obtain the bar plots, or the  table:
    # FigureS6
    plot_exp_var_bars_ind(RankDF_wind,size_name[index],"")
    ###############################################################################
    # not consider metrics without group or uninformative
    measures_no_RE = ordered_measures.copy()
    measures_no_RE.remove('<RE>') #erase uninformative metric
    RankDF.drop(23, inplace=True)
    cov_mat = av_pcov.loc[measures_no_RE, measures_no_RE]

    # get explained variance without idiosyncratic metrics ############################################################33
    #take out R0, tmax, Amax from analisis
    measures_clean=ordered_measures.copy()
    measures_to_remove=['<RE>','Amax','tmax','R0']
    for measure in measures_to_remove:
        measures_clean.remove(measure)

    #also take them out from the ranking db
    for metric_index in range(24,27):
        RankDF.drop(metric_index,inplace=True)

    Ranking.append(RankDF)

    RankDF_noind = get_exp_var_ind(av_pcov.loc[measures_clean,measures_clean], RankDF.copy())

    #("Explained Variance: ")
    #now to select 3 metrics in different forms and see explained variance:
    measure_exp_variance_of_3_groups(RankDF_noind,index,size_name[index],"noind")



##violin plots of pairwise correlations for different community sizes
plot_violin(corr_size,0.0)#FigureS4iv
########################################
# COVARIANCE ELLIPSOID STUDY            ##########################################
########################################
#FigureS7
Plot_difference_HV(av_scov_all,Ranking,groups,"")







quit()



