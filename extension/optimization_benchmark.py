import time

import numpy as np
import matplotlib.pyplot as plt

from autodiff_team29 import Node


def compute_expensive_duplicate_product_of_nodes(n_nodes: int):
    """
    Emulates the execution of the product of n numbers.
    This product is duplicated to test our optimization performance.
    For example, if we have a function (x1*x2*...*xn) + (x1*x2*...*xn).
    Can we save time by avoiding recomputing the second set of nodes?

    """

    # reset state of tracking
    Node.clear_node_registry()
    Node._NODES_COMPUTED_FOR_BENCHMARKING = 0

    # time expensive computation
    start = time.perf_counter()
    np.sum([Node(str(n), n, 1) for n in np.linspace(1,n_nodes,n_nodes)])
    np.sum([Node(str(n), n, 1) for n in np.linspace(1,n_nodes,n_nodes)])
    np.sum([Node(str(n), n, 1) for n in np.linspace(1, n_nodes, n_nodes)])
    np.sum([Node(str(n), n, 1) for n in np.linspace(1, n_nodes, n_nodes)])
    np.sum([Node(str(n), n, 1) for n in np.linspace(1, n_nodes, n_nodes)])
    end = time.perf_counter()
    elapsed_time = end - start

    # return results of computation
    return Node._NODES_COMPUTED_FOR_BENCHMARKING, elapsed_time


def benchmark(number_of_inputs, sample_size, overwrite_setting):
    """
    Benchmark to compare the performance speed with the node lookup enabled or disabled

    """
    # toggle overwrite mode on and off for the nodes
    Node.set_overwrite_mode(enabled=overwrite_setting)

    # store average time it takes to compute the product of n nodes
    average_execution_times = []
    standard_deviation_execution_times = []
    nodes_created = []

    for number in number_of_inputs:

        sample_execution_times = np.empty(sample_size)
        sample_number_nodes_created = np.empty(sample_size)

        for sample_index in range(sample_size):
            # get results of expensive computation
            number_nodes_created, elapsed_time = compute_expensive_duplicate_product_of_nodes(number)

            sample_execution_times[sample_index] = elapsed_time
            sample_number_nodes_created[sample_index] = number_nodes_created

        # compute average and standard deviation execution time for the sample
        average_sample_execution_time = np.average(sample_execution_times)
        std_sample_execution_time = np.std(sample_execution_times)
        average_nodes_created = np.average(sample_number_nodes_created)


        average_execution_times.append(average_sample_execution_time)
        standard_deviation_execution_times.append(std_sample_execution_time)
        nodes_created.append(average_nodes_created)


    return np.array(average_execution_times), np.array(
        standard_deviation_execution_times
    ), np.array(nodes_created)


if __name__ == "__main__":
    # number_of_inputs = (np.linspace(1, 70, 70) ** 2).astype(int)
    number_of_inputs = (np.linspace(1, 500, 500)).astype(int)
    sample_size = 50

    figure, (axis_1, axis_2) = plt.subplots(1,2,figsize=(20,5))

    # overwrite Mode off
    avg_disabled_times, std_disabled_times, number_nodes_created_disabled = benchmark(
        number_of_inputs, sample_size, overwrite_setting=False
    )


    # average Execution Time
    axis_1.plot(
        number_of_inputs,
        avg_disabled_times,
        color="blue",
        linestyle="-.",
        label="Overwriting Disabled",
    )
    axis_1.fill_between(
        number_of_inputs,
        avg_disabled_times + std_disabled_times,
        avg_disabled_times - std_disabled_times,
        alpha=0.5,
        color="skyblue",
        label="+/- Standard Deviation with Overwriting Disabled",
    )
    axis_2.plot(
        number_of_inputs,
        number_nodes_created_disabled,
        color="blue",
        linestyle="-.",
        label="Overwriting Disabled",
    )



    # average time for overwriting on
    avg_enabled_times, std_enabled_times, number_nodes_created_enabled = benchmark(
        number_of_inputs, sample_size, overwrite_setting=True
    )
    axis_1.plot(
        number_of_inputs,
        avg_enabled_times,
        color="red",
        linestyle="-.",
        label="Overwriting Enabled",
    )
    axis_1.fill_between(
        number_of_inputs,
        avg_enabled_times + std_enabled_times,
        avg_enabled_times - std_enabled_times,
        alpha=0.5,
        color="orangered",
        label="+/- Standard Deviation with Overwriting Enabled",
    )

    axis_2.plot(
        number_of_inputs,
        number_nodes_created_enabled,
        color="red",
        linestyle="-.",
        label="Overwriting Enabled",
    )

    # plot formatting
    axis_1.set_xlabel("n")
    axis_1.set_ylabel("Average Execution Time")
    axis_1.set_title(
        "Execution Time by Input Size"
    )

    axis_2.set_xlabel("n")
    axis_2.set_ylabel("Number of Nodes Created")
    axis_2.set_title(
        "Nodes Created by Input Size"
    )


    plt.suptitle("$f(x_1,x_2,...x_n) = (\prod_{i=1}^{n}x_i)+(\prod_{i=1}^{n}x_i)$ for various $n$")

    plt.figtext(
        0.98,
        0.01,
        "Tested on 2022 Apple M1 Max Processor \n 10-core CPU, 32-core GPU, 16-core Neural Engine",
        ha="right",
        fontsize=8,
        fontstyle="italic",
    )

    axis_1.legend()
    axis_2.legend()

    plt.tight_layout()
    figure.show()
