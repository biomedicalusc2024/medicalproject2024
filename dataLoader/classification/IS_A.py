import os
import pickle
import warnings
import numpy as np
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ..utils import print_sys, download_file

IS_A_SUBTITLE = ["hard", "easy"]


# tested by tjl 2025/1/22

# this is a function to read a graph from an edgelist.
def read_graph(file, get_connected_graph=True, remove_selfloops=True, get_directed=False):
    G = nx.read_edgelist(file, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    print('Graph created!!!')

    if remove_selfloops:
        # remove the edges with selfloops
        for node in nx.nodes_with_selfloops(G):
            G.remove_edge(node, node)

    if not get_directed:
        G = G.to_undirected()
        if get_connected_graph and not nx.is_connected(G):
            connected_sub_graph = max((G.subgraph(c).copy() for c in nx.connected_components(G)), key=len)
            print('Initial graph not connected...returning the largest connected subgraph..')
            return connected_sub_graph
        else:
            print('Returning undirected graph!')
            return G
    else:
        print('Returning directed graph!')
        return G


def create_train_test_splits_easy(datasetPath, graph, percent_pos, percent_neg):
    # This method creates the train,test splits...from the dataset
    # we remove some percentage of the existing edges(ensuring the graph remains connected)
    # and use them as positive samples
    # then we sample the same amount of non-existing edges, using them as negative samples
    # then do the same for testing set.
    dataset = "IS-A"
    np.random.seed(0)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_pos_train_edges = int(num_edges * percent_pos)
    num_neg_train_edges = int(num_edges * percent_neg)
    num_pos_test_edges = num_edges - num_pos_train_edges
    num_neg_test_edges = num_edges - num_neg_train_edges
    all_edges = [edge for edge in graph.edges]
    all_nodes = [node for node in graph.nodes]
    all_edges_set = set(all_edges)  # for quick access
    train_edges = set(all_edges)
    test_edges = set()
    counter1 = 0
    counter2 = 0
    if not graph.is_directed():
        original_connected_comps = nx.number_connected_components(graph)
        print('Connected components: ', original_connected_comps)

    print('Creating positive test samples..')
    # shuffle the edges and iterate over them creating the test set
    np.random.shuffle(all_edges)
    for edge in tqdm(all_edges):
        node1 = edge[0]
        node2 = edge[1]
        # make sure that the graph remains connected
        graph.remove_edge(node1, node2)
        reachable_from_v1 = nx.connected._plain_bfs(graph, len(graph), edge[0])
        if edge[1] not in reachable_from_v1:
            graph.add_edge(node1, node2)
            counter1 += 1
            continue
        # remove edges from the train_edges set and add them to the test_edges set --positive samples
        if len(test_edges) < num_pos_test_edges:
            test_edges.add(edge)
            train_edges.remove(edge)
            counter2 += 1
        elif len(test_edges) == num_pos_test_edges:
            break

    print("Added: {} number of edges to positive test".format(counter2))
    if len(test_edges) < num_pos_test_edges:
        print("Number of positive test edges could not be reached, because the graph would disconnect")
        print("Sampling the same amount of negative test edges...")
        num_neg_test_edges = len(test_edges)

    if len(train_edges) != num_pos_train_edges:
        # do the same with the positive samples
        num_neg_train_edges = len(train_edges)

    # now create false edges for test and train sets..making sure the edge is not a real edge
    # and not already sampled
    # first for test_set
    print('Creating negative test samples..')
    test_false_edges = set()
    while len(test_false_edges) < num_neg_test_edges:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        # we dont want to sample the same node
        if idx_i == idx_j:
            continue

        sampled_edge = (all_nodes[idx_i], all_nodes[idx_j])
        # check if we sampled a real edge or an already sampled edge
        if sampled_edge in all_edges_set:
            continue
        if sampled_edge in test_false_edges:
            continue
        # everything is ok so we add the fake edge to the test_set
        test_false_edges.add(sampled_edge)

    # do the same for the train_set
    print('Creating negative training samples...')
    train_false_edges = set()
    while len(train_false_edges) < num_neg_train_edges:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        # we don't want to sample the same node
        if idx_i == idx_j:
            continue

        sampled_edge = (all_nodes[idx_i], all_nodes[idx_j])
        # check if we sampled a real edge or an already sampled edge
        if sampled_edge in all_edges_set:
            continue
        if sampled_edge in test_false_edges:
            continue
        if sampled_edge in train_false_edges:
            continue
        # everything is ok so we add the fake edge to the train_set
        train_false_edges.add(sampled_edge)

    # asserts
    assert test_false_edges.isdisjoint(all_edges_set)
    assert train_false_edges.isdisjoint(all_edges_set)
    assert test_false_edges.isdisjoint(train_false_edges)
    assert test_edges.isdisjoint(train_edges)

    # convert them back to lists and return them
    train_pos = list(train_edges)
    train_neg = list(train_false_edges)
    test_pos = list(test_edges)
    test_neg = list(test_false_edges)

    odir = os.path.join(datasetPath, '{}_easy_splits'.format(dataset))
    if not os.path.exists(odir):
        os.makedirs(odir)

    # save the splits
    with open("{}.p".format(os.path.join(odir, '{}_train_pos'.format(dataset))), 'wb') as dump_file:
        pickle.dump(train_pos, dump_file)
    with open("{}.p".format(os.path.join(odir, '{}_train_neg'.format(dataset))), 'wb') as dump_file:
        pickle.dump(train_neg, dump_file)
    with open("{}.p".format(os.path.join(odir, '{}_test_pos'.format(dataset))), 'wb') as dump_file:
        pickle.dump(test_pos, dump_file)
    with open("{}.p".format(os.path.join(odir, '{}_test_neg'.format(dataset))), 'wb') as dump_file:
        pickle.dump(test_neg, dump_file)

    nx.write_edgelist(graph, os.path.join(odir, '{}_train_graph.edgelist'.format(dataset)))
    return train_pos, train_neg, test_pos, test_neg


def return_parents(graph, node, hops=None):
    if hops is not None:
        hops += 1
    return list(graph.predecessors(node)), hops


def create_train_test_splits_hard(datasetPath, graph, num_negative_test, num_negative_train):
    # This method creates the train,test splits...from the dataset
    # we remove some percentage of the existing edges(ensuring the graph remains connected)
    # and use them as positive samples
    # then we sample the same amount of non-existing edges, using them as negative samples
    # then do the same for testing set.
    dataset = "IS-A"
    np.random.seed(0)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    all_edges = [edge for edge in graph.edges]
    all_nodes = [node for node in graph.nodes]
    all_edges_set = set(all_edges)  # for quick access
    # counter1 = 0
    # counter2 = 0
    if not graph.is_directed():
        original_connected_comps = nx.number_connected_components(graph)
        print('Connected components: ', original_connected_comps)

        # if the graph is not directed--> convert it to directed
        graph = graph.to_directed()

    np.random.shuffle(all_edges)
    # shuffle the edges and iterate over them creating the test set
    # create false edges for test and train sets..making sure the edge is not a real edge
    # and not already sampled
    # first for test_set
    print('Creating negative test samples..')
    test_false_edges = set()
    while len(test_false_edges) < num_negative_test:
        print(len(test_false_edges))
        for node in tqdm(all_nodes):
            hop = 1
            parents = []

            # for parents
            # we can skip the og_parent and not append it in the list.
            # this is the first hop
            og_parent, hop = return_parents(graph, node, hop)
            while hop != 6:

                # if a node has more than one parents generate a random number and select a random one
                if len(og_parent) != 0:
                    og_parent, hop = return_parents(graph, og_parent[np.random.randint(0, len(og_parent))], hop)
                    # we must check again if parents exist
                    if len(og_parent) != 0:
                        parents.append(og_parent[np.random.randint(0, len(og_parent))])
                # no more parents
                else:
                    break

            # the first parent in the list is 2 hops away from our focus node.
            if len(parents) > 2:
                # select a random parent +1 hop away and get his children.
                rand_par = parents[int(np.random.uniform(0, len(parents)))]

                # get his children and select a random to node to create a negative example with
                children = list(graph.successors(rand_par))

                # we must exclude ancestors of the focus node from the sampling..so we remove the common elements from these lists
                cleaned_children = [child for child in children if child not in parents]
                if len(cleaned_children) != 0:
                    path_exists = False
                    rev_path_exists = False
                    rand_child = cleaned_children[int(np.random.uniform(0, len(cleaned_children)))]
                    reachable_from_v1 = nx.connected._plain_bfs(graph, len(graph), node)
                    if rand_child in reachable_from_v1:
                        path_exists = True

                    reachable_from_v1 = nx.connected._plain_bfs(graph, len(graph), rand_child)
                    if node in reachable_from_v1:
                        rev_path_exists = True

                    # the focus node might belong in the children so we must check it
                    if node != rand_child and not path_exists and not rev_path_exists:
                        sampled_edge = (node, rand_child)
                        sampled_edge_rev = (rand_child, node)
                    else:
                        continue
                else:
                    continue
                # check if we sampled a real edge or an already sampled one
                if sampled_edge in all_edges_set or sampled_edge_rev in all_edges_set:
                    continue
                if sampled_edge in test_false_edges or sampled_edge_rev in test_false_edges:
                    continue
                # everything is ok so we add the fake edge to the test_set
                if len(test_false_edges) == num_negative_test:
                    # we generated all the false edges we wanted
                    break
                else:
                    test_false_edges.add(sampled_edge)
            else:
                continue

    # do the same for the train_set
    print('Creating negative training samples...')
    train_false_edges = set()
    while len(train_false_edges) < num_negative_train:
        for node in tqdm(all_nodes):
            hop = 1
            parents = []

            # for parents
            # we can skip the og_parent and not append it in the list.
            # this is the first hop
            og_parent, hop = return_parents(graph, node, hop)
            while hop != 6:

                # if a node has more than one parents generate a random number and select a random one
                if len(og_parent) != 0:
                    og_parent, hop = return_parents(graph, og_parent[np.random.randint(0, len(og_parent))], hop)
                    # we must check again if any parents exist
                    if len(og_parent) != 0:
                        parents.append(og_parent[np.random.randint(0, len(og_parent))])
                # no more parents
                else:
                    break

            # the first parent in the list is 2 hops away from our focus node.
            if len(parents) > 2:
                # select a random parent and get his children.
                rand_par = parents[int(np.random.uniform(0, len(parents)))]

                # get his children and select a random to node to create a negative example with
                children = list(graph.successors(rand_par))

                # we must exclude ancestors of the focus node from the sampling..so we remove the common elements from these lists
                cleaned_children = [child for child in children if child not in parents]
                if len(cleaned_children) != 0:
                    path_exists = False
                    rev_path_exists = False
                    rand_child = cleaned_children[int(np.random.uniform(0, len(cleaned_children)))]
                    reachable_from_v1 = nx.connected._plain_bfs(graph, len(graph), node)
                    if rand_child in reachable_from_v1:
                        path_exists = True

                    reachable_from_v1 = nx.connected._plain_bfs(graph, len(graph), rand_child)
                    if node in reachable_from_v1:
                        rev_path_exists = True

                    # the focus node might belong in the children so we must check it
                    if node != rand_child and not path_exists and not rev_path_exists:
                        sampled_edge = (node, rand_child)
                        sampled_edge_rev = (rand_child, node)
                    else:
                        continue
                else:
                    continue
                # check if we sampled a real edge or an already sampled one
                if sampled_edge in all_edges_set or sampled_edge_rev in all_edges_set:
                    continue
                if sampled_edge in test_false_edges or sampled_edge_rev in test_false_edges:
                    continue
                if sampled_edge in train_false_edges or sampled_edge_rev in train_false_edges:
                    continue
                # everything is ok so we add the fake edge to the train_set
                if len(train_false_edges) == num_negative_train:
                    break
                else:
                    train_false_edges.add(sampled_edge)
            else:
                continue

    print("Total number of negative test samples: {}".format(len(test_false_edges)))
    print("Total number of negative train samples: {}".format(len(train_false_edges)))

    # asserts
    assert test_false_edges.isdisjoint(all_edges_set)
    assert train_false_edges.isdisjoint(all_edges_set)
    assert test_false_edges.isdisjoint(train_false_edges)

    # convert them back to lists and return them
    train_neg = list(train_false_edges)
    test_neg = list(test_false_edges)

    odir = os.path.join(datasetPath, '{}_hard_splits'.format(dataset))
    if not os.path.exists(odir):
        os.makedirs(odir)

    # save the splits
    with open("{}.p".format(os.path.join(odir, '{}_train_neg'.format(dataset))), 'wb') as dump_file:
        pickle.dump(train_neg, dump_file)
    with open("{}.p".format(os.path.join(odir, '{}_test_neg'.format(dataset))), 'wb') as dump_file:
        pickle.dump(test_neg, dump_file)

    return train_neg, test_neg


def processData(dataPath, is_hard):
    original_graph = read_graph(file=os.path.join(dataPath,"isa.edgelist"), get_connected_graph=True, remove_selfloops=True, get_directed=False)
    num_nodes = original_graph.number_of_nodes()
    num_edges = original_graph.number_of_edges()
    print('Original Graph: nodes: {}, edges: {}'.format(num_nodes, num_edges))

    percent_pos = 0.5
    percent_neg = 0.5

    train_pos, train_neg_easy, test_pos, test_neg_easy = create_train_test_splits_easy(dataPath, original_graph, percent_pos, percent_neg)

    print('Easy dataset created for the {} dataset.'.format("IS-A"))
    print('Number of positive training samples: ', len(train_pos))
    print('Number of negative training samples: ', len(train_neg_easy))
    print('Number of positive testing samples: ', len(test_pos))
    print('Number of negative testing samples: ', len(test_neg_easy))

    num_neg_test = len(test_pos)
    num_neg_train = len(train_pos)

    if is_hard:
        train_neg_hard, test_neg_hard = create_train_test_splits_hard(dataPath, original_graph, num_neg_test, num_neg_train)

        print('Hard dataset created for the {} dataset.'.format("IS-A"))
        print('Number of positive training samples: ', len(train_pos))
        print('Number of negative hard training samples: ', len(train_neg_hard))
        print('Number of positive testing samples: ', len(test_pos))
        print('Number of negative hard testing samples: ', len(test_neg_hard))


def getIS_A(path, subtitle):
    urls = ["https://raw.githubusercontent.com/SotirisKot/Content-Aware-Node2Vec/master/datasets/relation_instances_edgelists/isa.edgelist"]
    if subtitle not in IS_A_SUBTITLE:
        raise AttributeError(f'Please enter dataset name in IS_A-subset format and select the subsection of IS_A in {IS_A_SUBTITLE}')
    return datasetLoad(urls=urls, subtitle=subtitle, path=path, datasetName="IS_A")


def datasetLoad(urls, subtitle, path, datasetName):
    try:
        datasetPath = os.path.join(path, datasetName)
        datasetFile = os.path.join(datasetPath, "isa.edgelist")
        dataSplitPath = os.path.join(datasetPath, f"IS-A_{subtitle}_splits")
        is_hard = (subtitle == "hard")
        if os.path.exists(dataSplitPath):
            print_sys("Found local copy...")
            return loadLocalFiles(dataSplitPath)
        else:
            if os.path.exists(datasetFile):
                processData(datasetPath, is_hard)
                return loadLocalFiles(dataSplitPath)
            else:
                os.makedirs(datasetPath, exist_ok=True)
                download_file(urls[0], datasetFile)
                processData(datasetPath, is_hard)
                return loadLocalFiles(dataSplitPath)
    except Exception as e:
        print_sys(f"error: {e}")


def loadLocalFiles(dataSplitPath):
    if "easy" in dataSplitPath:
        train_pos = os.path.join(dataSplitPath, 'IS-A_train_pos.p')
        test_pos = os.path.join(dataSplitPath, 'IS-A_test_pos.p')
        train_neg = os.path.join(dataSplitPath, 'IS-A_train_neg.p')
        test_neg = os.path.join(dataSplitPath, 'IS-A_test_neg.p')
    elif "hard" in dataSplitPath:
        easyPath = "hard".join(dataSplitPath.split("easy"))
        train_pos = os.path.join(easyPath, 'IS-A_train_pos.p')
        test_pos = os.path.join(easyPath, 'IS-A_test_pos.p')
        train_neg = os.path.join(dataSplitPath, 'IS-A_train_neg.p')
        test_neg = os.path.join(dataSplitPath, 'IS-A_test_neg.p')
    with open(train_pos, 'rb') as file:
        train_pos_data = pickle.load(file)
        train_pos_label = [1] * len(train_pos_data)
    with open(train_neg, 'rb') as file:
        train_neg_data = pickle.load(file)
        train_neg_label = [-1] * len(train_neg_data)
    with open(test_pos, 'rb') as file:
        test_pos_data = pickle.load(file)
        test_pos_label = [1] * len(test_pos_data)
    with open(test_neg, 'rb') as file:
        test_neg_data = pickle.load(file)
        test_neg_label = [-1] * len(test_neg_data)
    
    trainset = [{"data":data,"label":label} for data,label in zip((train_pos_data+train_neg_data),(train_pos_label+train_neg_label))]
    testset = [{"data":data,"label":label} for data,label in zip((test_pos_data+test_neg_data),(test_pos_label+test_neg_label))]
    
    return trainset, testset