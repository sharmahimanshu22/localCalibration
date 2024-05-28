import argparse


def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        default=None,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--pnratio_train",
        default=None,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--pnratio_calibrate",
        default=None,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--pnratio_test",
        default=None,
        type=float,
        required=True,
    )
    parser.add_argument(
        "--n_train",
        default=None,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_calibrate",
        default=None,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_test",
        default=None,
        type=int,
        required=False,
    )

    parser.add_argument(
        "--model",
        default="NeuralNetwork",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--outdir",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--diminput",
        default=None,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--change",
        default=None,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--dummy",
        default=None,
        type=int,
        required=False,
    )


    return parser


