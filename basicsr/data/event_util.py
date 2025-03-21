import numpy as np
import torch
import random



def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):

    #Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    #:param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    #:param num_bins: number of bins in the temporal axis of the voxel grid
    #:param width, height: dimensions of the voxel grid
    #:param return_format: 'CHW' or 'HWC'


    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    # print('last stamp:{}'.format(last_stamp))
    # print('max stamp:{}'.format(events[:, 0].max()))
    # print('timestamp:{}'.format(events[:, 0]))
    # print('polarity:{}'.format(events[:, -1]))

    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT  # Normalize timestamps
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # Polarity should be +1 or -1

    # Add weighted interpolation
    tis = ts.astype(int)
    dts = ts - tis
    weights_left = (1.0 - dts)  # Weight for the left interval
    weights_right = dts  # Weight for the right interval

    # Add events to the voxel grid with weighted interpolation
    valid_indices = tis < num_bins  # Check if within the bin range
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + tis[valid_indices] * width * height, pols[valid_indices] * weights_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width + (tis[valid_indices] + 1) * width * height, pols[valid_indices] * weights_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1, 2, 0)

    return voxel_grid
'''
def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param return_format: 'CHW' or 'HWC'
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    # print('DEBUG: voxel.shape:{}'.format(voxel_grid.shape))

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    # print('last stamp:{}'.format(last_stamp))
    # print('max stamp:{}'.format(events[:, 0].max()))
    # print('timestamp:{}'.format(events[:, 0]))
    # print('polarity:{}'.format(events[:, -1]))

    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT #
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins # [True True ... True]
    # print('x max:{}'.format(xs[valid_indices].max()))
    # print('y max:{}'.format(ys[valid_indices].max()))
    # print('tix max:{}'.format(tis[valid_indices].max()))

    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width  ## ! ! !
            + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1,2,0)
'''
def voxel_norm(voxel):
    """
    Norm the voxel

    :param voxel: The unnormed voxel grid
    :return voxel: The normed voxel grid
    """
    nonzero_ev = (voxel != 0)
    num_nonzeros = nonzero_ev.sum()
    # print('DEBUG: num_nonzeros:{}'.format(num_nonzeros))
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = voxel.sum() / num_nonzeros
        stddev = torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        voxel = mask * (voxel - mean) / stddev

    return voxel


def filter_event(x, y, p, t, s_e_index=[0, 6]):
    '''
    s_e_index: include both left and right index
    '''
    t_1 = t.squeeze(1)
    uniqw, inverse = np.unique(t_1, return_inverse=True)
    discretized_ts = np.bincount(inverse)

    # Calculate the average time gap between events to adapt the exposure window
    avg_time_gap = np.mean(np.diff(t_1))  # Average time gap between events
    exposure_window = avg_time_gap * (s_e_index[1] - s_e_index[0])  # Adjust window size based on average gap

    index_exposure_start = np.sum(discretized_ts[0:s_e_index[0]])
    index_exposure_end = np.sum(discretized_ts[0:s_e_index[1] + 1])

    # Dynamically adjust the start and end based on the computed window
    adjusted_start = max(0, int(index_exposure_start - exposure_window))
    adjusted_end = min(len(t), int(index_exposure_end + exposure_window))

    x_1 = x[adjusted_start:adjusted_end]
    y_1 = y[adjusted_start:adjusted_end]
    p_1 = p[adjusted_start:adjusted_end]
    t_1 = t[adjusted_start:adjusted_end]

    return x_1, y_1, p_1, t_1

'''
def filter_event(x,y,p,t, s_e_index=[0,6]):

    #s_e_index: include both left and right index

    t_1=t.squeeze(1)
    uniqw, inverse = np.unique(t_1, return_inverse=True)
    discretized_ts = np.bincount(inverse)
    index_exposure_start = np.sum(discretized_ts[0:s_e_index[0]])
    index_exposure_end = np.sum(discretized_ts[0:s_e_index[1]+1])
    x_1 = x[index_exposure_start:index_exposure_end]
    y_1 = y[index_exposure_start:index_exposure_end]
    p_1 = p[index_exposure_start:index_exposure_end]
    t_1 = t[index_exposure_start:index_exposure_end]
    
    return x_1, y_1, p_1, t_1
'''


