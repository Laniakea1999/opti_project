import sys, getopt
from data.topologies.scripts.cities import print_city
from data.topologies.scripts.fully_connected import print_fully_connected
from data.topologies.scripts.ring import print_ring
from data.topologies.scripts.star import print_star
from helpers import *
from data.topologies.scripts import *


if (__name__ == '__main__'):

    argv = sys.argv

    long_args = ['type=', 'cities=', 'suburbs=']
    short_args = 't:n:c:s:'

    usage_string = "usage:\t{name} --type graph_type -n n_nodes".format(name=argv[0])

    graph_func = None
    graph_arg = (None, None)

    try:
        opts, args = getopt.getopt(argv[1:],short_args,long_args)
    except getopt.GetoptError:
        print(usage_string)
        sys.exit(1)
    
    for opt, arg in opts:
        if opt == '-h':
            print(usage_string)
            sys.exit()

        elif opt in ("-n"):
            graph_arg = int(arg)

        elif opt in ("-c", "--cities"):
            graph_arg = (int(arg), graph_arg[1])

        elif opt in ("-s", "--suburbs"):
            graph_arg = (graph_arg[0], int(arg))

        elif opt in ("-t", "--type"):
            if arg == 'city':
                graph_func = print_city
            elif arg == 'star':
                graph_func = print_star
            elif arg == 'ring':
                graph_func = print_ring
            elif arg == 'full':
                graph_func = print_fully_connected
        else:
            print(usage_string)
            sys.exit(1)

    graph_func(graph_arg)