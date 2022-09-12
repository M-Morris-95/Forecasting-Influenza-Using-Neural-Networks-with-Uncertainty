import argparse

def GetParser():
    parser = argparse.ArgumentParser(
        description='M-Morris-95 Forecasting')


    parser.add_argument('--quick', '-q',
                        type=bool,
                        help='run quick version',
                        default=False,
                        required = False)
    
    parser.add_argument('--start_test', '-s',
                        type=bool,
                        help='run test before optimising',
                        default=False,
                        required = False)

    parser.add_argument('--verbose', '-v',
                        type=bool,
                        help='print training info',
                        default=False,
                        required = False)

    parser.add_argument('--show_warnings', '-w',
                        type=bool,
                        help='print warnings',
                        default=False,
                        required = False)

    parser.add_argument('--clustered', '-C',
                        type=bool,
                        help='Do Clustering?',
                        default=False,
                        required = False)

    parser.add_argument('--gamma',
                        type=int,
                        help='distance ahead to forecast',
                        default=14,
                        required=False)

    parser.add_argument('--n_queries',
                        type=int,
                        help='dnumber of queries',
                        default=100,
                        required=False)

    parser.add_argument('--test_season',
                        type=int,
                        help='what test season',
                        default=2015,
                        required=False)

    parser.add_argument('--epochs', '--E',
                        type=int,
                        help='number of epochs',
                        default=150,
                        required=False)

    parser.add_argument('--num_folds', '--n',
                        type=int,
                        help='number of folds in KF',
                        default=5,
                        required=False)

    parser.add_argument('--repeats', '--r',
                        type=int,
                        help='number of repeats',
                        default=3,
                        required=False)

    parser.add_argument('--batch_size', '--B',
                        type=int,
                        help='Batch_Size',
                        default=32,
                        required=False)

    parser.add_argument('--model_type',
                        type=str,
                        help='which model use? FF, LSTM',
                        default='FF',
                        required=False)

    parser.add_argument('--window_size',
                        type=int,
                        help='how much lag should be in the data, 28 or 112?',
                        default=28,
                        required=False)

    return parser