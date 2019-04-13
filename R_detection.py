# ECG的R波检测
# 仅需调用select_R方法


from scipy.signal import find_peaks
import numpy as np
# import matplotlib.pyplot as plt


R_N = 9  # Size of R_Peak_Value_List
R_Peak_Value_List = [0] * R_N
R_Moment_List = [0]
R_Interval_List = [0]
middel_peak = 0
middel_interval = 0
samp_freq = 0
raw_data = []


def select_R(ecg, sampling_frequency, plot=False):
    """
    Call this function to utilize all functions in this file.

    Parameters:
        raw_data(np.array): ECG data.
        samp_freq(int): Sampling frequency.
        plot_result(bool): Whether draw the plot of the result

    Returns:
        R_Moment_List(np.array): An arrary that contain all moments of R
        R_Interval_List(np.array): An array that contain all intervals of R
    """
    global samp_freq, R_Moment_List, raw_data
    samp_freq = sampling_frequency
    raw_data = ecg
    (filter1, filter_result) = filtrate_raw_ECG()
    RPS_1 = R_Primary_Selection(filter_result, 10)
    RPS_2 = R_Primary_Selection(filter_result, 15)
    RSS_1 = R_Senior_Selection(RPS_1, 0.1)
    RSS_2 = R_Senior_Selection(RPS_2, 0.1)
    R_Moment_List = sorted(set(RSS_1).union(RSS_2))
    R_Interval_List = generate_interval_list(R_Moment_List)
    return np.array(R_Moment_List), np.array(R_Interval_List)


def R_Senior_Selection(RSS_in, period):
    """
    There will be a slightly error in R moment due to mean filter.
    Adjusting R_Moment_List by finding local maximum in a samll range
    to eliminate this error.

    Parameters:
        raw_data(np.array): ECG data.
    """
    inspection_range = int(period*samp_freq)
    for index_RML in range(1, len(RSS_in)):
        temp = RSS_in[index_RML]
        max_index = np.argmax(np.array([raw_data[x] for x in
                                       range(temp-inspection_range,
                                             temp+inspection_range)]))
        RSS_in[index_RML] = temp-inspection_range+max_index
    return RSS_in


# def Union(l1, l2):
#     i = 0
#     j = 0
#     l_out = [0]
#     len1 = len(l1)
#     len2 = len(l2)
#     while True:
#         cur_out = l_out[-1]
#         if i
#         while (i < len1-1) and (l1[i] <= cur_out):
#             i = i + 1
#         while (j < len2-1) and (l2[j] <= cur_out):
#             j = j + 1
#         l_out.append(min(l1[i], l2[j]))
#     return l_out


def R_Primary_Selection(filter_result, start_point):
    """
    R波初选，每十秒做一次筛选

    parameters:
        raw_data(np.array): ECG data.
        filter_result(list): The result of filtrate_raw_ECG()
    """
    global R_Peak_Value_List, R_Interval_List
    R_Interval_List = [0]
    RPS_out = [0]
    delay = -1  # Filters delay compensation
    # If the last section of data less than 10 seconds, discard it
    for index in range(start_point*samp_freq, len(raw_data), 10*samp_freq):
        start = index - 10 * samp_freq
        update_middel_peak_and_interval()  # Update threshold
        local_maxima_position, _ = find_peaks(filter_result[start:index],
                                              distance=max(middel_interval*0.8,
                                                           samp_freq/3),
                                              height=middel_peak*0.2)
        R_num = len(local_maxima_position)
        # Make local_maxima_position in line with R_Moment_List in time
        local_maxima_position = (np.array([start]*R_num)+local_maxima_position)
        # Add local_maxima_position to R_Moment_Lis
        RPS_out = (RPS_out +
                   list(local_maxima_position-np.array([delay]*R_num)))
        # Updarte R_Peak_Value_List
        R_Peak_Value_List = [filter_result[x] for x in local_maxima_position]
        R_Interval_List = (np.array(RPS_out[1:]) -
                           np.array(RPS_out[:-1]))
    return RPS_out


def generate_interval_list(gil_in):
    return list(np.append(np.array(gil_in[1:]), [0]) - np.array(gil_in))


def update_middel_peak_and_interval():
    """
    Calculate middel value of past nine R peak value and R interval.
    """
    global middel_peak, middel_interval
    middel_peak = np.median(R_Peak_Value_List)
    middel_interval = np.median(R_Interval_List)


def filtrate_raw_ECG():
    """
    Go through all filters.

    Return:
        (list): A list that contain filtered data
    """
    filter1 = Moving_Mean_Filter(raw_data)
    filter2 = Derivative_Filter_and_Square(filter1)
    filter3 = Moving_Mean_Filter(filter2)
    return filter2, filter3


def Derivative_Filter_and_Square(Der_in):
    """
    A optimized derivative filter.

    Parameter:
        Der_in(float): Input data.

    Return:
        (float): Filter output.
    """
    Der_out = []
    last_item = Der_in[0]
    for item in Der_in:
        temp = item - last_item
        if temp > 0:
            Der_out.append(np.square(temp))
        else:
            Der_out.append(0)
        last_item = item
    return Der_out


def Moving_Mean_Filter(Mov_in, n=3):
    """
    A moving mean filter.

    Parameter:
        Mov_in(float): Input data.

    Return:
        (float): Filter output.
    """
    Mov_in = list(Mov_in)
    # 边缘填充
    filling = [Mov_in[0]]*int(1+(n-1)/2) + Mov_in + [Mov_in[-1]]*int((n-1)/2)
    Mean_Value_Buff = filling[:n]

    Mov_Out = []
    for item in filling[n:]:
        Mean_Value_Buff = Mean_Value_Buff[1:]+[item]
        Mov_Out.append(sum(Mean_Value_Buff)/n)
    return Mov_Out
