import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def draw_graph(nodes, edges):
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])


    nodes_ls = list(nodes)

    color_lookup = {}
    for node in nodes_ls:
        color_lookup.update({node.uid: node.loss})

    low, *_, high = sorted(color_lookup.values())
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=False)
    node_mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)


    fig = plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G)

    pos_higher = {}
    y_off = 0.15  # offset on the y axis

    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]+y_off)

    nx.draw_networkx_nodes(G, 
                            pos,
                            cmap=plt.get_cmap('bone'), 
                            node_color=[node_mapper.to_rgba(i) for i in color_lookup.values()],
                            node_size = 500
                        )

    labels = {}
    for node in nodes_ls:
        labels.update({node.uid: f"{node.uid}: {node.loss:.2f}"})

    nx.draw_networkx_labels(G, pos_higher, font_color='black', font_size=10, labels=labels)
    nx.draw_networkx_edges(G, pos, arrows=False, edge_color='grey')
    fig.patch.set_alpha(0.0)

    return fig

def draw_confusion_matrix(confusion_matrix):
    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    return ax.get_figure()

def plot_losses(training_losses, color='blue', marker='+'):
    plt.plot(training_losses, color=color, marker=marker, linewidth=1)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    n = 1
    n_losses = len(training_losses)
    if (n_losses > 200):
        n = 100
    elif (n_losses > 100):
        n = 50
    elif (n_losses > 50):
        n = 25
    elif (n_losses > 20):
        n = 10

    plt.xticks(range(len(training_losses))[::n])
    return plt.gcf()

def print_error(msg):
    print(u"\u001b[41m\u001b[1m\u001b[37m ERROR: \u001b[0m {}\n".format(msg))

def print_success(msg):
    print(u"\u001b[42m\u001b[1m\u001b[37m   OK:  \u001b[0m {}".format(msg))

def print_metric(name, value, extra=None):
    print(u"\t{name}: \t\u001b[37m{value}\u001b[0m\t{extra}".format(name=name, value=value, extra="" if extra==None else extra))