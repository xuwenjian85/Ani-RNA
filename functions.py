## call from jupyter notebook:
## %run /public236T/test1/axolotl_rev/script/functions.py

# Author: XWJ
# Date: 2025-05-19 logging
import sys
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import datetime
import logging

###################################################################
# 1. general 
###################################################################

def log_time(start_time, step_name):
    """
    Record and print the current time and elapsed time.

    Parameters:
    - start_time: The start time of the script
    - step_name: The name of the current step
    """
    current_time = datetime.datetime.now()
    elapsed_time = current_time - start_time
    
    print(f"STEP: {step_name}")
    print(f"Current Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")} Elapsed Time: {elapsed_time}\n")

###################################################################
# 2. Global network built under coexpression context
###################################################################

def pearson_extract_upper_triangle(pcc):
    """
    提取 Pearson 相关系数矩阵的上三角部分，并保留行索引和列索引信息。

    参数：
    - pcc: Pearson 相关系数矩阵（Pandas DataFrame）

    返回：
    - 包含上三角部分的 DataFrame
    """
    # 提取上三角部分的索引
    triu_indices = np.triu_indices_from(pcc.values, k=1)
    
    # 提取上三角部分的值
    upper_tri_values = pcc.values[triu_indices]
    
    # 创建一个 DataFrame，包含行索引和列索引
    upper_tri_df = pd.DataFrame({
        'gene1': pcc.index[triu_indices[0]],
        'gene2': pcc.columns[triu_indices[1]],
        'pcc': upper_tri_values
    })
    
    return upper_tri_df

        
def edges_filter_by_threshold(df, column, threshold):
    """
    根据给定的阈值过滤 DataFrame。

    参数：
    - df: 输入的 DataFrame
    - column: 需要过滤的列名
    - threshold: 阈值

    返回：
    - 过滤后的 DataFrame
    """
    thres_u = df[column].quantile(q=1 - threshold)
    filtered_df = df.query(f'{column} > @thres_u')
    return filtered_df
    
def pearson_to_network(pcc_tri, thres_q_pcc, output_dir, c, ctype, p):
    """
    保存 Pearson 相关性网络。

    参数：
    - pcc_tri: 三角形相关性 DataFrame
    - thres_q_pcc: 相关性阈值
    - output_dir: 输出目录
    - c: 参数 c
    - ctype: 参数 ctype
    - p: 参数 p

    返回：
    - 网络对象
    """
    pcc_edges = edges_filter_by_threshold(pcc_tri, 'pcc', thres_q_pcc)
    # thres_u = pcc_tri['pcc'].quantile(q=1-thres_q_pcc)
    # pcc_tri.query('pcc > @thres_u')
    file = f'{output_dir}/t0_dedup_coex{c}_{ctype}_{p}_{thres_q_pcc:.3f}.tsv.gz'
    pcc_edges.to_csv(file, sep='\t', header=False, index=False)
    # print(file)
    G_int = nx.read_weighted_edgelist(file, delimiter='\t', nodetype=str)
    
    return G_int

def expand_fullconnect_network(pcc, pcc_tri, list_large, add_genes, num_new_edge, output_dir, c, ctype, p, thres_q_pcc):
    """
    扩展网络以连接所有组件。

    参数：
    - G_int: 网络对象
    - pcc: 完整的相关性矩阵
    - pcc_tri: 下三角形的相关性stack DataFrame
    - list_large: 最大连通组件的基因列表
    - add_genes: 需要添加的基因列表
    - num_new_edge: 每个基因添加的新边数
    - output_dir: 输出目录
    - c: 参数 c
    - ctype: 参数 ctype
    - p: 参数 p
    - thres_q_pcc: 相关性阈值

    返回：
    - 扩展后的网络对象
    """
    add_edges = pcc.loc[list_large, add_genes].astype(np.float64)
    mask = add_edges.rank(ascending=False) < num_new_edge
    add_edges = add_edges[mask].stack().reset_index().rename(columns={'level_1': 'gene2', 0: 'pcc'})

    # thres_u = pcc_tri['pcc'].quantile(q=1-thres_q_pcc)
    # pcc_tri.query('(pcc > @thres_u)')
    pcc_edges = edges_filter_by_threshold(pcc_tri, 'pcc', thres_q_pcc)
    all_edges = pd.concat([pcc_edges, add_edges])

    file = f'{output_dir}/t0_dedup_coex{c}_{ctype}_{p}_{thres_q_pcc:.3f}_add{len(add_edges)}.tsv.gz'
    all_edges.to_csv(file, sep='\t', header=False, index=False)
    # print(file)
    G_int = nx.read_weighted_edgelist(file, delimiter='\t', nodetype=str)
    
    return G_int

def check_network_connect(G, mode=None):
    """读取网络并检查连通性; 验证最终网络是否全连通"""
    logging.info(f"Nodes:{G.number_of_nodes():<8} Edges:{G.number_of_edges():<8} Is Connected:{nx.is_connected(G)}\n")
    if mode == 'connect?':
        return nx.is_connected(G)
    if mode == 'fully connect':
        assert nx.is_connected(G), "The network is not fully connected. Consider lowering the threshold or adding more edges."

# 主函数
def global_net(**kwargs):
    """
    处理网络，保存相关性网络，扩展网络以连接所有组件，并验证最终网络是否全连通。

    参数：
    - kwargs: 包含所有参数的字典
    返回：
    - 扩展后的网络对象
    """
    pcc_tri = kwargs['pcc_tri']
    pcc = kwargs['pcc']
    thres_q_pcc = kwargs['thres_q_pcc']
    c = kwargs['c']
    ctype = kwargs['ctype']
    p = kwargs['p']
    output_dir = kwargs['output_dir']

    # 读取Pearson相关性网络并检查连通性
    G_int =  pearson_to_network(pcc_tri, thres_q_pcc, output_dir, c, ctype, p)
    check_network_connect(G_int)
    
    # 获取最大连通组件和需要添加的基因
    connected_subgraphs, connected_subgraphs_size = sub_graph_size(G_int)
    list_large = list(connected_subgraphs[0])
    add_genes = [g for g in pcc.index if g not in list_large]

    # 扩展网络以连接所有组件
    G_int = expand_fullconnect_network(pcc, pcc_tri, list_large, add_genes, num_new_edge=10, output_dir=output_dir, c=c, ctype=ctype, p=p, thres_q_pcc=thres_q_pcc)
    check_network_connect(G_int, 'fully connect')

    return G_int

###################################################################
# 3. AE network built with aberrant nodes and edges under coexpression context
###################################################################

def select_ae_genes(df_score, s, thres_score, ae_gene_max, df_outlier):
    """
    选择 AE 基因。
    参数：
    - df_score: 基因评分数据
    - s: 样本名称
    - thres_score: 评分阈值
    - ae_gene_max: AE基因的最大数量
    - df_outlier: 异常值数据
    返回：
    - ae_g: AE 基因列表
    """
    mask_ae = (df_score < thres_score)
    ae_g = set(df_score[mask_ae][s].dropna().nsmallest(ae_gene_max).index)
    cand_g = df_outlier.query('Sample == @s')['Gene'].values[0]
    return list(ae_g | {cand_g})

def find_high_correlated_genes(series_ae, pcc, pcc_tri, thres_q_pcc, pcc_gene_min, pcc_gene_max):
    """
    找到 AE 基因的高相关基因对。

    参数：
    - series_ae: AE 基因的评分序列
    - pcc: 基因之间的皮尔逊相关系数矩阵
    - thres_q_pcc: 相关性阈值
    - pcc_gene_min: 基因对相关性最小值
    - pcc_gene_max: 基因对相关性最大值

    返回：
    - ae_edges: 包含高相关基因对的 DataFrame
    """
    values = []
    thres_u = pcc_tri['pcc'].quantile(q=1-thres_q_pcc)
    logging.info(f"{'Gene':<10} | {'AE Value':<10} | {'Edge':<10}")
    logging.info("-" * 40)
    for g1 in series_ae.index:
        # mask_self = pcc[g1] != 1  # 排除自身相关性
        mask_self = pcc.index != g1
        num_pcc_g = (pcc[g1][mask_self] > thres_u).sum()  # 高相关基因对的数量
        
        if num_pcc_g < pcc_gene_min:
            take = pcc_gene_min  # 如果数量小于最小值，取最小值
        elif pcc_gene_min <= num_pcc_g <= pcc_gene_max:
            take = num_pcc_g  # 如果在范围内，取实际数量
        else:
            take = pcc_gene_max  # 如果数量大于最大值，取最大值

        series_ae_pcc = pcc.loc[mask_self, g1].astype(np.float32).nlargest(take)  # 获取高相关基因对
        values.append(series_ae_pcc)
        
        logging.info(f"{g1:<10} | {series_ae.loc[g1]:<10.2e} | {series_ae_pcc.shape[0]:<5}")
    logging.info("\n")
    # 构建边列表
    ae_edges = pd.DataFrame(values).stack().reset_index()
    ae_edges.columns = ['gene1', 'gene2', 'pcc']

    return ae_edges


def find_shortest_paths_between_modules(connected_subgraphs, G_int):
    """
    找出所有模块之间的最短路径。

    参数：
    - connected_subgraphs: 连通子图列表
    - G_int: 交互网络

    返回：
    - df_spl: 包含最短路径信息的 DataFrame
    - spl_minimum: 每个模块对的最短路径长度
    """
    values = []
    
    # 遍历所有模块对
    for sub_large in range(len(connected_subgraphs)):
        for sub_small in range(sub_large + 1, len(connected_subgraphs)):
            for g1 in connected_subgraphs[sub_large]:
                if not G_int.has_node(g1):
                    logging.info(g1, 'not found')
                    continue
                for g2 in connected_subgraphs[sub_small]:
                    if not G_int.has_node(g2):
                        logging.info(g2, 'not found')
                        continue
                    try:
                        sp = nx.shortest_path(G_int, source=g1, target=g2)
                        sp_edges = list(zip(sp, sp[1:]))
                        values.append([sub_large, sub_small, g1, g2, len(sp), sp, sp_edges])
                    except nx.NetworkXNoPath:
                        logging.info(f"不存在从 {g1}@{sub_large} 到 {g2}@{sub_small} 的路径。")
                        continue

    # 创建 DataFrame
    df_spl = pd.DataFrame(values, columns=['large', 'small', 'source', 'target', 'spl', 'sp_genes', 'sp_edges'])

    spl_minimum = df_spl.groupby(['large', 'small'])['spl'].min()
    df_spl['shortest'] = df_spl.apply(lambda row: row['spl'] == spl_minimum.loc[row['large'], row['small']], axis=1)
    df_spl = df_spl[df_spl['shortest']].copy()
    
    return df_spl, spl_minimum

def assign_priority_and_filter_paths(df_spl, ae_g, spl_minimum, G_int):
    """
    为路径分配优先级并根据规则过滤路径。

    参数：
    - df_spl: 包含路径信息的 DataFrame
    - ae_g: AE 基因列表
    - spl_minimum: 每个模块对的最短路径长度
    - G_int: 交互网络

    返回：
    - df_spl: 更新后的路径 DataFrame
    """
    # 初始化优先级为 0
    df_spl.loc[:, 'prior'] = 0.0

    # 优先：1. 包含 AE 基因的边设置为 prior
    df_spl.loc[df_spl['source'].isin(ae_g) | df_spl['target'].isin(ae_g), 'prior'] = 1
    df_spl.loc[df_spl['source'].isin(ae_g) & df_spl['target'].isin(ae_g), 'prior'] = 2

    # 有多条路径时的排除规则
    for (sub_large, sub_small) in spl_minimum.index:
        p_mark = df_spl.query('(large == @sub_large) & (small == @sub_small)')

        if (p_mark['prior'] == 2).any():
            # 如果存在优先级为 2 的路径，其他路径优先级设为 -1
            df_spl.loc[p_mark.query('prior != 2').index, 'prior'] = -1
        elif (p_mark['prior'] == 1).any():
            # 如果存在优先级为 1 的路径，其他路径优先级设为 -1
            df_spl.loc[p_mark.query('prior != 1').index, 'prior'] = -1
        else:
            # 计算路径权重乘积
            values, index = [], []
            for idx, row in p_mark.iterrows():
                gs, gt = row['source'], row['target']
                sp_edges = row['sp_edges']
                weights = [G_int.get_edge_data(gs, gt)['weight'] for (gs, gt) in sp_edges]
                product = np.prod(weights)
                values.append(product)
                index.append(idx)

            # 保留权重乘积最大的路径
            mask = pd.Series(data=values, index=index).nlargest(1).index
            df_spl.loc[mask, 'prior'] = 0.5
    
    df_spl = df_spl.query('prior > 0').copy()
    
    return df_spl

def add_edges_to_network(G, new_edges):
    """
    向网络中添加边。

    参数：
    - G: 网络对象
    - new_edges: 要添加的边列表
    """
    assert not nx.is_frozen(G), "network is frozen. cannot add edges"
    G.add_edges_from(new_edges)
        
def ml_model_prediction(X, model_name):
    """
    使用指定模型计算异常值检测分数。
    参数：
    - X: 输入数据
    - model_name: 模型名称
    返回：
    - 预测分数
    """
    # outlier model scores in g1,g2 pair expression space for every sample
    if model_name == "LOF": #"distance-based"# <1min
        clf = LocalOutlierFactor(n_neighbors=20, contamination="auto")
        clf.fit(X)
        decision_scores = clf.negative_outlier_factor_

    elif model_name == "iForest": #"distance-based" # 1min
#         y_pred = isolationforest_mean(X)# 1min
        clf = IsolationForest( n_estimators=100, contamination='auto',random_state = 2024 ) # 100,20;contamination=0.01
        decision_scores = clf.fit(X).decision_function(X)

    elif model_name == 'OCSVM': # One-class classification
        clf = OneClassSVM(gamma='auto')
        decision_scores = clf.fit(X).decision_function(X)
        
    else:
        raise ValueError("Unsupported model name")
    
    positive_ds = - decision_scores
    return positive_ds
    

def residual_linear_regression(y, x):
    """
    计算线性回归的残差。
    参数：
    - x1: 因变量
    - x0: 自变量
    返回：
    - 残差数据
    """
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    residuals = y - predictions
    return residuals

def split_list_into_sublists(lst):
    """
    将列表分割为子列表。
    参数：
    - lst: 输入列表
    返回：
    - 子列表列表
    """
    return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]

def sub_graph_size(G):
    """
    计算子图的大小。
    参数：
    - graph: 网络图
    返回：
    - 子图列表和子图大小
    """
    connected_subgraphs = list(nx.connected_components(G))
    connected_subgraphs_size = [len(subgraph) for subgraph in connected_subgraphs]
    return connected_subgraphs, connected_subgraphs_size

def create_geneset_column(df, col1, col2):
    """
    创建一个新列，该列包含两个基因的排序后的连接字符串。

    参数：
    - df: 输入的 DataFrame
    - col1: 第一个基因列的名称
    - col2: 第二个基因列的名称

    返回：
    - 包含新列
    """
    return df.apply(lambda row: '_'.join(np.sort([row[col1], row[col2]])), axis=1)

# 主函数
# AE are top gene list by outlier method, look up these gene in PCC matrix, find top correlated genes
def ae_net(s, **kwargs):
    """
    构建和分析基因网络。
    参数：
    - s: 样本名称
    - kwargs: 其他参数
    返回：
    - 节点和边列表
    """

    output_dir = kwargs['output_dir'] 
    df_exp = kwargs['df_exp']
    df_score = kwargs['df_score']
    df_outlier = kwargs['df_outlier']
    pcc_tri = kwargs['pcc_tri']
    pcc = kwargs['pcc']
    thres_score = kwargs['thres_score']
    thres_q_pcc = kwargs['thres_q_pcc']
    pcc_gene_min = kwargs['pcc_gene_min']
    pcc_gene_max = kwargs['pcc_gene_max']
    ae_gene_max = kwargs['ae_gene_max']
    G_int = kwargs['G_int']
    file_e = kwargs['file_e']
    # logger_ae_net = kwargs['logger_ae_net']
    
    thres_q_pcc_add = 0.05
    num_clusters = [5, 10, 15]

    # ------------------------------
    # Step 1. Find AE and neighbors
	# ------------------------------
    
    # 选择和排序AE基因
    ae_g = select_ae_genes(df_score, s, thres_score, ae_gene_max, df_outlier)
    series_ae = df_score.loc[ae_g, s].sort_values()
    # print('AE net: get AE genes')
    logging.info('AE net: get AE genes')
    # 找直接邻居，构建边列表
    ae_edges = find_high_correlated_genes(series_ae, pcc, pcc_tri, thres_q_pcc, pcc_gene_min, pcc_gene_max)
	
    file = f'{output_dir}/data/edge_ae_{s}.txt'
    ae_edges.to_csv(file, sep='\t', header=False, index=False)
    
    # ------------------------------
    # Part 2. Build AE Network
	# ------------------------------
 
    # 每AE基因和直接邻居为一个子模块，构建起始网络
    A_int = nx.read_weighted_edgelist(file, delimiter='\t', nodetype=str)
    logging.info('AE net: check AE network connectivity, 1')
    
    connect_state = check_network_connect(A_int,'connect?')
    if connect_state == False:
        # 找出每个模块对的最短路径
        connected_subgraphs, connected_subgraphs_size = sub_graph_size(A_int)
        logging.info(f'Modules:{ ','.join([str(size) for size in connected_subgraphs_size]) }')
        df_spl, spl_minimum = find_shortest_paths_between_modules(connected_subgraphs, G_int)
        
        # spl_minimum = df_spl.groupby(['large', 'small'])['spl'].min()
        # df_spl['shortest'] = df_spl.apply(lambda row: row['spl'] == spl_minimum.loc[row['large'], row['small']], axis=1)
        # df_spl = df_spl[df_spl['shortest']].copy()
        
        df_spl = assign_priority_and_filter_paths(df_spl, ae_g, spl_minimum, G_int)
        logging.info('AE net: shortest paths between modules\n')
        logging.info(df_spl['prior'].value_counts(sort=True).T)
        
        # 增加最短路径的包含的所有边，同时节点数量也随之增加
        spl_edges = sum([gs for gs in df_spl['sp_edges']], [])
        add_edges_to_network(A_int, spl_edges)
        logging.info('AE net: check AE network connectivity, 2')
        check_network_connect(A_int, 'fully connect')
        
    # ------------------------------
    # Part 3. Node and Edge tables
	# ------------------------------
    
    # node attributes: ae score and source group
    nodes = pd.DataFrame({
        'id': list(A_int.nodes()),
        'AE': df_score.loc[list(A_int.nodes()), s].values,
        'group': ['AE' if node in ae_g else 'L1' if node in ae_edges['gene2'].unique() else 'L2' for node in A_int.nodes()]
    })
        
    # node attributes: coexpress cluster group
    mygrid = sns.clustermap(pcc.loc[nodes['id'], nodes['id']], figsize=(5, 5))
    plt.close()
    mygrid.savefig(f'{output_dir}/data/clustermap_{s}.png')
    
    nodes[num_clusters] = hierarchy.cut_tree(mygrid.dendrogram_row.linkage, n_clusters=num_clusters)

    # Edge attributes: Group (AE or ADD or ADD2)
    ae_edges['geneset']  = create_geneset_column(ae_edges, 'gene1','gene2')
    
    A_int_edges = pd.DataFrame(data=A_int.edges(), columns=['gene1', 'gene2'])
    A_int_edges['geneset'] = create_geneset_column(A_int_edges, 'gene1','gene2') 
    A_int_edges['group'] = A_int_edges['geneset'].apply(lambda x: 'AE' if x in ae_edges['geneset'].values else 'ADD')
            
    # except edges already in A_int, get other large weight edges between these nodes
    
    # s_pcc_tri = pd.Series(pcc.loc[nodes['id'], nodes['id']].stack()).reset_index().rename(columns={0: 'pcc'})
    # s_pcc_tri = s_pcc_tri[ s_pcc_tri['gene1'] < s_pcc_tri['gene2'] ]
    s_pcc_tri = pearson_extract_upper_triangle(pcc.loc[nodes['id'], nodes['id']])
    s_pcc_tri['geneset'] = create_geneset_column(s_pcc_tri, 'gene1','gene2')
    cand_edges = s_pcc_tri[ ~s_pcc_tri['geneset'].isin(A_int_edges['geneset']) ]
    cand_edges = edges_filter_by_threshold(cand_edges, 'pcc', thres_q_pcc_add)  
    cand_edges['group'] = 'ADD2'
    
    # adding new edges to A_int
    edges = pd.concat([A_int_edges[['gene1', 'gene2', 'group']], cand_edges[['gene1', 'gene2', 'group']]]).reset_index(drop=True)
    
    logging.info('AE net: run SWEET analysis')
    sweet_edge_score(s, file_e, f'{output_dir}/sweet')
    # get SWEET results
    sweet = pd.DataFrame(np.load(f'{output_dir}/sweet/{s}.txt.npy'), 
                         index = pcc.index.tolist(), columns = pcc.index.tolist(), dtype=np.float16).rename_axis('gene1',axis=0).rename_axis('gene2', axis=1)
    
    # Edge attributes: more 
    value_sweet, value_pcc, value_outscore, value_residual_z = [], [], [], []
    for idx in edges.index:
        g0, g1 = edges.loc[idx, 'gene1'], edges.loc[idx, 'gene2']
        # ML outlier detection model 
        X = df_exp.loc[[g0, g1], :].T
        d_scores = pd.DataFrame(index=X.index)
        for model_name in ['LOF']:
            d_scores[model_name] = ml_model_prediction(X, model_name)
        
        # linear regression
        y, x = df_exp.loc[[g0], :].T, df_exp.loc[[g1], :].T
        residuals = residual_linear_regression(y, x)
        residuals.loc[:, 'z'] = StandardScaler().fit_transform(residuals.values)
        
        # list of attributes value
        value_sweet.append( sweet.loc[g0, g1])
        value_pcc.append( pcc.loc[g0, g1])
        value_outscore.append(d_scores.loc[s, model_name])
        value_residual_z.append(residuals.loc[s, 'z'])

    edges['sweet'] = value_sweet
    edges['pcc'] = value_pcc
    edges['outscore'] = value_outscore
    edges['residual_z'] = value_residual_z
    edges['abs(sweet-pcc)'] = abs(edges['sweet'] - edges['pcc'])
    
    return nodes, edges

###################################################################
# 4. SWEET, edge aberrant score by cohort coexpression 
###################################################################

def sweet_compute_sample_weight(file, save, k=0.1):
    """
    计算样本权重并保存到文件。

    参数：
    - file: 基因表达矩阵文件路径
    - k: 平衡参数
    - save: 输出文件路径

    返回：
    - 权重 DataFrame
    """
    # 读取基因表达矩阵
    df = pd.read_csv(file, sep='\t', index_col=0)
    pat = df.columns
    patlen = len(pat)

    # 计算相关系数矩阵
    corr_matrix = df.corr()
    value = (corr_matrix.sum(axis=1) - 1) / (patlen - 1)

    # 归一化权重
    rmax, rmin = value.max(), value.min()
    dif = rmax - rmin + 0.01
    value = (value - rmin + 0.01) / dif
    value = value * k * patlen

    # 创建权重 DataFrame
    weight_df = pd.DataFrame({'patient': pat, 'sample_weight': value})

    # 保存权重到文件
    weight_df.to_csv(save, sep='\t', index=False)
    # print(f"Sample weight saved to: '{save}'")

def sweet_check_file(expres):
    checkset = set(["", "NA", "Na", "na", "nan", "null"])
    for c in checkset:
        loc = np.where(expres == c)
        if loc[0].size:
            expres[loc] = "0"
            logging.info(f"Notice! There is {c} in the 'gene expression matrix' file and it will be assigned to 0.")
    return expres

def sweet_save_samples(samples, file_p):
    """
    将多个样本保存到文件。

    参数：
    - samples: 样本名称列表
    - file_p: 输出的样本文件路径
    """
    with open(file_p, 'w') as f:
        f.write('\n'.join(samples) + '\n')
    # print(f"Samples saved to: {file_p}")
    
def sweet_save_genes(file_e, file_g):
    """
    从基因表达矩阵文件中提取所有基因，并保存到文件。

    参数：
    - file_e: 基因表达矩阵文件路径
    - file_g: 输出的基因文件路径
    """
    # 加载基因表达矩阵
    df = pd.read_csv(file_e, sep='\t', index_col=0)
    
    # 提取所有基因
    genes = df.index
    
    # 保存基因到文件
    genes.to_series().to_csv(file_g, sep='\t', index=False, header=False)
    # print(f"Gene names saved to: {file_g}")

# 主函数
def sweet_edge_score(s, file_e, save_path):
    """
    计算样本权重并保存到文件，然后计算每个样本的边分数并保存为 NumPy 文件。

    参数：
    - file_e: 基因表达矩阵文件路径
    - file_w: 样本权重文件路径
    - file_p: 感兴趣的样本文件路径
    - file_g: 感兴趣的基因文件路径
    - save_path: 输出路径

    返回：
    - 保存边分数矩阵的文件路径列表
    """
    # save sample names to file
    file_p = f'{save_path}/patient.txt'
    file_w = f'{save_path}/weight.txt'
    file_g = f'{save_path}/gene.txt'
    
    sweet_save_samples([s], file_p)
    # compute weight for all samples, save a file  in the expression file
    sweet_compute_sample_weight(file_e, file_w)
    # save gene names to file
    sweet_save_genes(file_e, file_g)
    
    # open 'samples of interest' file
    patientset = set()
    with open(file_p, mode='r') as rline:
        for nline in rline:
            tem = nline.strip('\n').split('\t')
            patientset.add(tem[0])
    if not patientset:
        logging.info("Warning! There is no sample ID in the 'samples of interest' file.")
        sys.exit()

    # open 'genes of interest' file
    geneset = set()
    with open(file_g, mode='r') as rline:
        for nline in rline:
            tem = nline.strip('\n').split('\t')
            geneset.add(tem[0])
    if not geneset:
        logging.info("Warning! There is no gene ID in the 'genes of interest' file.")
        sys.exit()

    # open 'sample weight' file
    weight = {}
    with open(file_w, mode='r') as rline:
        _ = rline.readline()
        for nline in rline:
            p, w, *_ = nline.strip('\n').split('\t')
            weight.update({p: float(w)})
            
    if not weight:
        logging.info("Warning! There is no sample ID in the 'sample weight' file.")
        sys.exit()

    # open 'gene expression matrix' file
    gene, value = [], []
    with open(file_e, mode='r') as rline:
        pat = rline.readline().strip('\n').split('\t')[1:]
        for nline in rline:
            g, *v = nline.strip('\n').split('\t')
            if g in geneset:
                value += v
                gene.append(g)

    patlen, genelen = len(pat), len(gene)
    if (not patlen):
        logging.info("Warning! The 'gene expression matrix' file is empty")
        sys.exit()

    # check the 'samples of interest' and 'genes of interest' in expression file
    patloc = [l for l, p in enumerate(pat) if p in patientset]
    logging.info( f'patients:{patientset}, index:{patloc}, total samples:{patlen}, total genes:{genelen}\n' )
    if (not genelen) or (len(patloc) != len(patientset)):
        logging.info("Warning! The expression file cannot be mapped to 'samples of interest' or 'genes of interest' file")
        sys.exit()
    if len(set(pat) & weight.keys()) != patlen:
        logging.info("Warning! The sample ID(s) in the expression file cannot be mapped to 'sample weight' file")
        sys.exit()

    gene = np.array(gene)
    value = np.array(value).reshape(genelen, patlen)
    value = sweet_check_file(value)
    value = value.astype(float)
    loc = np.where(np.sum(value, axis=1) == 0)
    if len(loc[0]) != 0:
        tem = ','.join(str(i)for i in gene[loc])
        logging.info('Processing: delete gene(s) with zero expression values in all samples:'+tem)
        value = np.delete(value, loc, 0)
        gene = np.delete(gene, loc)

    agg = np.corrcoef(value)
    # edge score matrix 
    for l in patloc:
        p = pat[l]
        value_s = np.c_[value, value[:, l]]
        value_s = np.corrcoef(value_s)
        value_s = weight[p] * (value_s - agg) + agg
        
        fastfile = f"{save_path}/{p}.txt.npy"
        np.save(fastfile, value_s)
        # logging.info('Done!', p, fastfile)
        
# 
    
def label_aberrant_edges(edges, cutoff_residual_z=2, cutoff_outscore=2, groups=['ADD', 'ADD2', 'AE']):
    """
    Labels aberrant edges in a DataFrame based on given criteria.

    Parameters:
    - edges: DataFrame containing the edges data.
    - cutoff_residual_z: float, cutoff value for the absolute residual_z.
    - cutoff_outscore: float, cutoff value for the outscore.
    - groups: list, groups to be considered.

    Returns:
    - DataFrame with labeled edges.
    """
    # 创建一个掩码，用于筛选出满足条件的边
    mask = (edges['outscore'] > cutoff_outscore) & \
           (abs(edges['residual_z']) > cutoff_residual_z) & \
           (edges['group'].isin(groups))
    
    # 将满足条件的边的label设置为'aberrant'
    edges.loc[mask, 'label'] = 'aberrant'
    # 将未满足条件的边的label设置为'normal'
    edges['label'] = edges['label'].fillna('normal')
    
    # 返回标记后的edges
    return edges

# Example usage:
# edges = label_aberrant_edges(edges)

import matplotlib.pyplot as plt
import seaborn as sns

def create_scatter_plot(edges):
    """
    创建散点图并返回图形对象。

    参数:
    - edges: 包含边缘数据的 DataFrame。

    返回:
    - fig: 图形对象。
    - axs: 轴对象数组。
    """
    n_row, n_col = 1, 3
    fig, axs = plt.subplots(n_row, n_col, sharey=False, figsize=(6 * n_col, 6 * n_row))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # 绘制散点图
    sns.scatterplot(data=edges, x='residual_z', y='outscore', hue='group', style='label', size='abs(sweet-pcc)', alpha=0.8, ax=axs[0])
    sns.scatterplot(data=edges, x='residual_z', y='abs(sweet-pcc)', hue='group', style='label', size='outscore', alpha=0.8, ax=axs[1])
    sns.scatterplot(data=edges, x='abs(sweet-pcc)', y='outscore', hue='group', style='label', alpha=0.8, ax=axs[2])

    return fig, axs