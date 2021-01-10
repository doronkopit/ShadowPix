import numpy as np


def log_statistics(metrices, log_path=None, log_all=False):
    analysis = ""

    analysis += f'\nPrinting summary of stats ->\n'

    if log_all:
        for row in range(metrices.grid_size):
            for col in range(metrices.grid_size):
                success_count = metrices.pix_stat[row][col][0]
                fail_count = metrices.pix_stat[row][col][1]
                score = metrices.pix_score[row][col]
                analysis += f'\nRow {row}, Col {col}: Score={score}, Success={success_count}, Fail={fail_count}\n'

    # Top 10 pixels
    top = 30
    analysis += f'\nTop {top} pixels --> \n'
    max_top = np.argsort(metrices.pix_score.reshape(-1))[::-1][:top]
    analysis += make_analysis(max_top, model=metrices)

    # Bottom 10 pixels
    bottom = 30
    analysis += f'\nBottom {bottom} pixels --> \n'
    min_top = np.argsort(metrices.pix_stat[:, :, 1].reshape(-1))[::-1][:bottom]
    analysis += make_analysis(min_top, model=metrices)

    if log_path is None:
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_path = "log_" + timestr + ".log"

    write_analysis_to_file(path=log_path, analysis=analysis)
    print(f'PixModel statistics are saved in {log_path}')


def make_analysis(top_list, model):
    analysis = ""
    for pos, i in enumerate(top_list):
        row = i // model.grid_size
        col = i % model.grid_size
        success_count = model.pix_stat[row][col][0]
        fail_count = model.pix_stat[row][col][1]
        neighbor_succ = model.pix_stat[row][col][2]
        neighbor_fail = model.pix_stat[row][col][3]
        score = model.pix_score[row][col]
        analysis += f'\n{pos+1:} - Row {row}, Col {col}: Score={score}, Success={success_count}, \
            Fail={fail_count}, Neighbor Success={neighbor_succ}, Neighbor Fail={neighbor_fail}\n'

    return analysis


def write_analysis_to_file(path, analysis):
    with open(path, 'w') as f:
        f.write(analysis)
