import numpy as np


def print_stats(metrices, print_all=False):
    print(f'Printing summary of stats ->\n')

    if print_all:
        for row in range(metrices.grid_size):
            for col in range(metrices.grid_size):
                success_count = metrices.pix_stat[row][col][0]
                fail_count = metrices.pix_stat[row][col][1]
                score = metrices.pix_score[row][col]
                print(f'Row {row}, Col {col}: Score={score}, Success={success_count}, Fail={fail_count}')

    # Top 10 pixels
    top = 30
    print(f'Top {top} pixels --> \n')
    max_top = np.argsort(metrices.pix_score.reshape(-1))[::-1][:top]
    print_analysis(max_top, model=metrices)

    # Bottom 10 pixels
    bottom = 30
    print(f'\nBottom {bottom} pixels --> \n')
    min_top = np.argsort(metrices.pix_stat[:, :, 1].reshape(-1))[::-1][:bottom]
    print_analysis(min_top, model=metrices)


def print_analysis(top_list, model):
    for pos, i in enumerate(top_list):
        row = i // model.grid_size
        col = i % model.grid_size
        success_count = model.pix_stat[row][col][0]
        fail_count = model.pix_stat[row][col][1]
        neighbor_succ = model.pix_stat[row][col][2]
        neighbor_fail = model.pix_stat[row][col][3]
        score = model.pix_score[row][col]
        print(f'{pos+1:} - Row {row}, Col {col}: Score={score}, Success={success_count}, \
            Fail={fail_count}, Neighbor Success={neighbor_succ}, Neighbor Fail={neighbor_fail}')
