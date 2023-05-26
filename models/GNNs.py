import logging
import os

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv

import Config
from utils import get_edge_norm

logger = logging.getLogger(__name__)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout, num_layers=2):
        super(ResidualGRU, self).__init__()
        # 定义 GRU 层，将输入的 hidden_size 维度向量转换为 hidden_size/2 维度向量
        # num_layers 指定 GRU 层的层数，batch_first=True 表示输入数据的第一维为 batch size，dropout 表示丢弃概率，bidirectional=True 表示双向 GRU
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        # 定义 LayerNorm 层，对 GRU 输出进行归一化
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        self.enc_layer.flatten_parameters()  # 将参数变成一维，加速计算
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class GRUPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = ResidualGRU(hidden_size, dropout=Config.model_args.dropout)

    def forward(self, graphs, return_origin_shape=False):
        graphs_hidden_states = graphs.ndata['h']  # 获取图的节点特征

        graphs_hidden_states_after_padding = []   # 对不同 batch 中的节点数进行 padding，使得它们的节点数都相同
        batch_num_nodes = graphs.batch_num_nodes().detach().cpu().numpy().tolist()  # 获取 batch 中每个图的节点数
        max_node_num = max(batch_num_nodes)
        for index, num_nodes in enumerate(batch_num_nodes):
            start_index = sum(batch_num_nodes[:index])  # 当前图在 graphs_hidden_states 中的起始位置
            end_index = sum(batch_num_nodes[:index + 1])  # 当前图在 graphs_hidden_states 中的结束位置

            padding_embeddings = torch.zeros((max_node_num - num_nodes, self.hidden_size),  # 进行 padding，将多出来的部分用零填充
                                             dtype=graphs_hidden_states.dtype, device=graphs_hidden_states.device)
            graphs_hidden_states_after_padding.append(     # 将 padding 后的节点特征添加到 graphs_hidden_states_after_padding 中
                torch.cat([graphs_hidden_states[start_index:end_index], padding_embeddings], dim=0))

        # 将 graphs_hidden_states_after_padding 转化为张量，大小为 batch_size × max_node_num × hidden_size
        graphs_hidden_states_after_padding = torch.stack(graphs_hidden_states_after_padding).view(-1, max_node_num,
                                                                                                  self.hidden_size)
        graphs_hidden_states = self.gru(graphs_hidden_states_after_padding)   # (13) 经过 GRU 层，得到池化后的节点特征
        if not return_origin_shape:
            return graphs_hidden_states

        # 需要恢复原始形状，即去除 padding 部分的节点特征
        ret_graphs_hidden_states = []
        for index, num_nodes in enumerate(batch_num_nodes):
            ret_graphs_hidden_states.append(graphs_hidden_states[index, :num_nodes, :])
        return torch.cat(ret_graphs_hidden_states, dim=0)


class GraphAttentionPoolingWithGRU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 定义用于预测图节点权重的线性层和激活函数
        self.graph_weight_pred = nn.Sequential(nn.Linear(hidden_size * 2, 1, bias=False), nn.LeakyReLU())
        self.gru_pooling = GRUPooling(hidden_size)   # 定义 GRU Pooling 层

    def forward(self, graphs, attention_query):
        graphs_hidden_states = self.gru_pooling(graphs)  # 通过 GRU Pooling 层获取图的隐藏状态
        max_node_num = max(graphs.batch_num_nodes().detach().cpu().numpy().tolist())
        graphs_node_weight = self.graph_weight_pred(torch.cat(   # 使用线性层和激活函数计算注意力权重
            [graphs_hidden_states, attention_query.view(-1, 1, self.hidden_size).repeat(1, max_node_num, 1)],
            dim=-1))
        # 对节点注意力进行 softmax 规范化，并将形状调整为 (batch_size, max_node_num, 1)
        graphs_node_weight = torch.softmax(graphs_node_weight, dim=1).view(len(graphs_hidden_states), max_node_num, 1)
        # 将节点隐藏状态与其对应的注意力权重相乘，得到加权后的隐藏状态
        graphs_hidden_states = graphs_hidden_states * graphs_node_weight
        return torch.sum(graphs_hidden_states, dim=1).view(-1, self.hidden_size)    # (14)


class feat_nn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert len(x.shape) == 2
        return x[:, :x.shape[-1] // 2]


class GraphAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 定义一个包含LeakyReLU激活函数的线性层，输入为2倍的hidden_size，输出为1维
        self.attention_layer = nn.Sequential(torch.nn.Linear(2 * hidden_size, 1, bias=False), torch.nn.LeakyReLU())
        self.graph_pooling = dgl.nn.GlobalAttentionPooling(self.attention_layer, feat_nn=feat_nn())

    def forward(self, graphs, attention_query):
        batch_num_nodes = graphs.batch_num_nodes().detach().cpu().numpy().tolist()   # 获取图中每个batch的节点数量
        attention_query = attention_query.view(len(batch_num_nodes), self.hidden_size)  # 将attention_query变形为(batch_size, hidden_size)的张量
        attention_query_repeat = []
        for index, num_nodes in enumerate(batch_num_nodes):
            attention_query_repeat.append(attention_query[index].view(1, self.hidden_size).repeat(num_nodes, 1))
        return self.graph_pooling(graphs,   # 将节点特征h与重复后的attention_query按列拼接，然后使用GlobalAttentionPooling进行池化
                                  torch.cat([graphs.ndata['h'], torch.cat(attention_query_repeat, dim=0)], dim=-1))


class GraphSet2SetPooling(nn.Module):
    def __init__(self, hidden_size, set_2_set_n_iters=3, set_2_set_n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.graph_pooling = dgl.nn.Set2Set(hidden_size, set_2_set_n_iters, set_2_set_n_layers)  # Set2Set模块
        self.output_layer = torch.nn.Linear(hidden_size * 2, hidden_size)   # 线性变换层，将Set2Set池化的结果进行线性变换

    def forward(self, graphs, attention_query: None):
        return self.output_layer(self.graph_pooling(graphs, graphs.ndata['h']))


class GraphSortPooling(nn.Module):
    def __init__(self, hidden_size, k=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.graph_pooling = dgl.nn.SortPooling(k)   # 定义 SortPooling 层，用于对每个图的节点进行排序和池化
        self.attention_layer = nn.Sequential(nn.Linear(hidden_size * 2, 1, bias=False), nn.LeakyReLU())  # 定义注意力机制的层

    def forward(self, graphs, attention_query):
        # 对节点进行排序和池化，并改变输出形状为(batch_size, k, hidden_size)
        output = self.graph_pooling(graphs, graphs.ndata['h']).view(graphs.batch_size, self.k, self.hidden_size)
        atten_weight = self.attention_layer(   # 计算注意力权重，将输出和注意力查询向量拼接后作为输入
            torch.cat([output, attention_query.view(graphs.batch_size, 1, self.hidden_size).repeat(1, self.k, 1)],
                      dim=-1)).view(graphs.batch_size, self.k, 1)
        atten_weight = torch.softmax(atten_weight, dim=1)   # 对注意力权重进行softmax操作
        return torch.sum(output * atten_weight, dim=1)   # 对输出和注意力权重进行加权求和，得到最终的图表示


def nodes_with_feature_x(nodes, feature, x):
    return (nodes.data[feature] == x).view(-1)


def nodes_with_nodes_subgraph_type_0(nodes):
    return nodes_with_feature_x(nodes, 'subgraph_type', 0)


def nodes_with_nodes_subgraph_type_1(nodes):
    return nodes_with_feature_x(nodes, 'subgraph_type', 1)


class RGATLayer(nn.Module):
    def __init__(self, hidden_size, num_rels, num_bases=-1, bias=None, activation=None):
        super(RGATLayer, self).__init__()
        self.in_feat = hidden_size
        self.out_feat = hidden_size
        self.hidden_size = hidden_size
        self.num_rels = num_rels   # 关系类型的数量
        self.num_bases = num_bases  # 基矩阵的数量
        self.bias = bias
        self.activation = activation     # 激活函数 relu
        self.dropout = nn.Dropout(Config.model_args.dropout)                        # ada11 把(10)(11)中的LeakyReLU换成SELU试试; ada12 relu; ada20 rrelu+二难+fine-tune
        self.subgraph_attn = nn.Sequential(nn.Linear(hidden_size * 2, 1, bias=False), nn.RReLU())  # 把一个linear层和relu封装在一起，子图注意力机制，用于计算节点在子图中的权重

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        # weight bases in equation (3)
        else:
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))   # 基矩阵的权重 跟着网络的训练进行学习更新
        self.subgraph_proj = nn.Linear(self.in_feat, self.out_feat)
        self.subgraph_gate = nn.Linear(self.in_feat * 2, 1, bias=False)
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))  # 权重系数矩阵
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))   # 偏置项
        self.attn_layer = nn.Sequential(torch.nn.Linear(2 * hidden_size, 1, bias=False), nn.RReLU())   # 注意力机制，用于计算邻居节点的权重
        self.extension_pred_layer = torch.nn.Linear(hidden_size * 2, 1)   # 扩展预测层，用于预测节点的标签

        self.self_loop = torch.nn.Linear(hidden_size, hidden_size)  # 自循环层，用于学习节点的自环特征

    def forward(self, _g, base_nodes_ids=None, exten_nodes_ids=None, exten_edges_ids=None, ):
        graphs_list = dgl.unbatch(_g)  # 将大图拆分成一个图列表
        num_count = 0
        graphs_node_ids = [base_nodes_ids[i][base_nodes_ids[i] != Config.extension_padding_value] for i in  # 获取每个图的节点ID
                           range(len(graphs_list))]
        rel_scores = []
        for i in range(len(graphs_list)):
            rel_socre_i = []
            exten_edges = [[], []]  # 定义一个二元列表存储外部扩展的边
            edge_relation_type = []  # 定义一个列表存储外部扩展的边的关系类型
            for j in range(len(exten_edges_ids[i])):    # 遍历图中的扩展边
                eni = exten_edges_ids[i][j][exten_edges_ids[i][j] != Config.extension_padding_value][:2]  # 获取扩展的边的起始点和终止点
                if len(eni) == 0:
                    continue
                pred_value = self.extension_pred_layer(torch.cat(   # 计算预测的值   （6）
                    [torch.mean(torch.index_select(graphs_list[i].ndata['h'], dim=0, index=eni), dim=0),
                     torch.mean(torch.index_select(graphs_list[i].ndata['answer'], dim=0, index=eni), dim=0)],
                    dim=-1))
                pred_value = torch.sigmoid(pred_value).view(1)   # 对预测值进行Sigmoid激活并调整形状   （7）
                if pred_value > Config.extension_threshold:  # 如果预测值大于设定阈值，则将该边添加到exten_edges和edge_relation_type中
                    exten_edges[0].append(eni[0].view(-1))
                    exten_edges[1].append(eni[1].view(-1))
                    edge_relation_type.append(exten_edges_ids[i][j][2].view(1))
                rel_socre_i.append(pred_value)
            if len(exten_edges[0]) != 0:   # 如果存在外部扩展的边，则将其添加到当前图
                graphs_list[i].add_edges(torch.cat(exten_edges[0]), torch.cat(exten_edges[1]),
                                         {'rel_type': torch.cat(edge_relation_type),
                                          'norm': torch.ones(len(exten_edges[0]), device=exten_edges_ids.device)})
            for j in range(len(exten_nodes_ids[i])):    # 遍历图中的扩展节点
                eni = exten_nodes_ids[i][j][exten_nodes_ids[i][j] != Config.extension_padding_value]
                if len(eni) == 0 or not graphs_list[i].has_edges_between(eni[0], eni[2]):
                    continue
                pred_value = self.extension_pred_layer(torch.cat(   # 预测是否需要将eni添加到原始图中，如果需要，则将eni添加到新图的节点列表中
                    [torch.mean(torch.index_select(graphs_list[i].ndata['h'], dim=0, index=eni), dim=0),
                     torch.mean(torch.index_select(graphs_list[i].ndata['answer'], dim=0, index=eni), dim=0)],
                    dim=-1))
                pred_value = torch.sigmoid(pred_value).view(1)
                if pred_value > Config.extension_threshold:
                    graphs_node_ids[i] = torch.unique(torch.cat([graphs_node_ids[i], eni]))
                rel_socre_i.append(pred_value)
            new_graph = dgl.node_subgraph(graphs_list[i], nodes=graphs_node_ids[i])   # 构建子图，指定节点集合，即需要保留的节点集合
            new_graph.ndata['origin_id'] = graphs_node_ids[i] + num_count   # 新节点的id需要加上原始节点的数量
            num_count += graphs_list[i].num_nodes()
            new_graph.edata['norm'] = torch.tensor(   # 标准化边
                get_edge_norm(new_graph.edata['rel_type'], new_graph, new_graph.edges()),
                device=new_graph.device).view(-1)

            # # 用关系得分初始化边+二难     ada15
            # norms = []
            # for k in range(new_graph.num_edges()):
            #     a = new_graph.edges()[0][k]
            #     b = new_graph.edges()[1][k]
            #     eni = torch.tensor([a, b]).to(new_graph.device)
            #     pred_value = self.extension_pred_layer(torch.cat(
            #         [torch.mean(torch.index_select(new_graph.ndata['h'], dim=0, index=eni), dim=0),
            #          torch.mean(torch.index_select(new_graph.ndata['answer'], dim=0, index=eni), dim=0)],
            #         dim=-1))
            #     pred_value = torch.sigmoid(pred_value).view(1)
            #     norms.append(pred_value)
            #
            # # # 用关系得分+标准化初始化边+二难  sigmoid   ada16
            # # norms1 = get_edge_norm(new_graph.edata['rel_type'], new_graph, new_graph.edges())
            # # norms = [(a + b)/2 for a, b in zip(norms, norms1)]
            # norms = torch.tensor(norms, device=new_graph.device).view(-1)
            # new_graph.edata['norm'] = norms

            # 构建两种子图
            context_subgraph = dgl.node_subgraph(new_graph,    # Vc
                                                 nodes=new_graph.filter_nodes(nodes_with_nodes_subgraph_type_0))
            choice_subgraph = dgl.node_subgraph(new_graph,     # Vo
                                                nodes=new_graph.filter_nodes(nodes_with_nodes_subgraph_type_1))
            # 遍历新图中的节点，计算每个节点对应的子图信息
            subgraph_message = []
            for index in range(new_graph.num_nodes()):
                subgraph = context_subgraph if new_graph.ndata['subgraph_type'][index] == 1 else choice_subgraph
                attn = self.subgraph_attn(torch.cat([subgraph.ndata['h'],
                                                     new_graph.ndata['h'][index].view(1,
                                                                                      self.hidden_size).repeat(
                                                         subgraph.num_nodes(), 1)], dim=-1))
                attn = torch.softmax(attn.view(-1), dim=-1)
                h = torch.sum(subgraph.ndata['h'] * attn.view(-1, 1), dim=0)   # （10）
                subgraph_message.append(h)
            new_graph.ndata['subgraph_message'] = torch.cat(subgraph_message, dim=0).view(new_graph.num_nodes(),
                                                                                          self.hidden_size)
            graphs_list[i] = new_graph
            # 记录关系得分
            rel_scores.append(torch.mean(torch.cat(rel_socre_i, dim=-1)).view(1) if len(rel_socre_i) != 0 else
                              torch.ones(1, device=_g.device))
        g = dgl.batch(graphs_list)   # 打包成一个batch的大图

        # 对边进行注意力计算，返回包含注意力值的字典
        def edge_attn(edges):
            attn = self.attn_layer(torch.cat([edges.src['h'], edges.dst['h']], dim=-1))
            return {'e_attn': attn}

        # 将源节点 src 的隐藏状态与边类型 rel_type 对应的权重矩阵相乘得到消息，然后将其与边的 norm 相乘，返回包含消息和注意力值的字典。
        def message_func(edges):
            w = self.weight[edges.data['rel_type'].cpu().numpy().tolist()]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm'].view(-1, 1)
            return {'msg': msg, 'e_attn': edges.data['e_attn']}

        # 对消息进行汇总，使用 softmax 对注意力值进行归一化，计算子图中每个节点的新隐藏状态 h，并返回字典。
        def reduce_func(nodes):
            alpha = torch.softmax(nodes.mailbox['e_attn'], dim=1)
            subgraph_gate_weight = self.subgraph_gate(
                torch.cat([nodes.data['h'], nodes.data['subgraph_message']], dim=-1))
            subgraph_gate_weight = torch.sigmoid(subgraph_gate_weight).view(-1, 1)
            subgraph_msg = self.subgraph_proj(nodes.data['subgraph_message']) * subgraph_gate_weight
            h = torch.sum(nodes.mailbox['msg'] * alpha, dim=1) + self.self_loop(nodes.data['h']) + subgraph_msg   # （11）
            return {'h': h}

        # 应用偏置项到节点的隐藏状态中，然后返回包含更新隐藏状态的字典。
        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            return {'h': h}

        g.apply_edges(edge_attn)   # 应用边注意力函数
        g.update_all(message_func, reduce_func, apply_func)   # 更新节点表示，执行消息传递、聚合和应用函数
        new_data_map = dict(    # 获取新节点表示，并将其存储到字典中
            [nid.detach().cpu().numpy().tolist(), h] for nid, h in zip(g.ndata['origin_id'], g.ndata['h']))
        whole_g_h = _g.ndata['h']   # 获取整个图的节点表示
        new_whole_g_h = []   # 存储新图节点表示的列表
        for i in range(len(whole_g_h)):
            if i in new_data_map:
                new_whole_g_h.append(new_data_map[i])
            else:
                new_whole_g_h.append(whole_g_h[i])   # 将原节点表示和新节点表示结合起来
        h = torch.stack(new_whole_g_h)
        if self.activation is not None:
            h = self.activation(h)     # relu
        _g.ndata['h'] = h   # 更新整个图的节点表示
        return self.dropout(h), g.ndata['origin_id'], g.batch_num_nodes(), g.batch_num_edges(), torch.cat(rel_scores,
                                                                                                          dim=-1).view(
            -1, 1)


class RGAT(nn.Module):
    def __init__(self, num_layers, hidden_dim, base_num, num_rels=6, activation=torch.relu):
        super(RGAT, self).__init__()
        self.num_layers = num_layers   # 图卷积层数
        self.dropout = nn.Dropout(Config.model_args.dropout)
        self.rgat_layers = nn.ModuleList()   # 图卷积层列表
        self.global_pooling_layer = nn.Linear(hidden_dim * num_layers, hidden_dim)   # 全局池化层，用于合并所有层的节点表示
        for _ in range(num_layers):
            self.rgat_layers.append(
                RGATLayer(hidden_dim, num_rels=num_rels, num_bases=base_num, activation=activation, ))   # 图卷积层
        pooling_type = Config.model_args.pooling_type   # 图池化层
        pooling_classes = {'attention_pooling_with_gru': GraphAttentionPoolingWithGRU,
                           'attention_pooling': GraphAttentionPooling, 'set2set_pooling': GraphSet2SetPooling,
                           'sort_pooling': GraphSortPooling}
        assert pooling_type in pooling_classes
        self.graph_pooling = pooling_classes[pooling_type](hidden_dim)

    def forward(self, graph, base_nodes_ids, exten_nodes_ids, exten_edges_ids, attention_query, ):
        graph.ndata['origin_h'] = graph.ndata['h']   # 保存原始节点表示
        all_rel_scores = []  # 所有层的关系得分列表
        all_h = []   # 所有层的节点表示列表
        for i in range(self.num_layers):   # 计算图卷积
            h, node_ids, batch_num_nodes, batch_num_edges, rel_scores = self.rgat_layers[i](graph, base_nodes_ids,
                                                                                            exten_nodes_ids,
                                                                                            exten_edges_ids, )
            all_rel_scores.append(rel_scores)
            all_h.append(h)
        all_h = torch.cat(all_h, dim=-1)  # with size graph.num_nodes() * (d * num_layers)
        all_rel_scores = torch.cat(all_rel_scores, dim=-1)  # 将所有层的关系得分拼接到一起
        graph.ndata['h'] = all_h   # 将所有层的节点表示赋给图
        # 从图中取出所有层的节点，形成子图
        graph = dgl.node_subgraph(graph, nodes=node_ids)
        graph.set_batch_num_nodes(batch_num_nodes)
        graph.set_batch_num_edges(batch_num_edges)
        # 对子图进行全局池化，得到一个大小为 hidden_dim 的向量
        graph.ndata['h'] = self.global_pooling_layer(graph.ndata['h']) + graph.ndata['origin_h']   # (12)
        # 将全局池化的结果和所有层的关系得分拼接在一起，形成最终输出
        output = torch.cat([self.graph_pooling(graph, attention_query), all_rel_scores], dim=-1)   # (15)
        return output
