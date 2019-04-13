# ECG的R波校正
# 仅需调用select_R方法
import math
import numpy as np
import R_detection
from scipy.signal import find_peaks
from scipy.signal import convolve


ecg = []
rml = []
rml_copy = []
der = []
samp_freq = 0


def refine(ecg_in, R_Moment_List, derivative, fs):
    """

    parameters:
        ecg_in(list):
        R_Moment_List(np.array):
        derivative(np.array):

    """
    global ecg, rml, der, rml_copy, samp_freq
    ecg = np.array(ecg_in)
    rml = R_Moment_List
    rml_copy = rml
    samp_freq = fs
    der = derivative
    CE = Clearness_Evaluation(25, 0.7, 20)
    IE = Interval_Evaluation(0.15)
    # 选出洁净且正常的点
    C_I = np.array(CE) * np.array(IE)
    # print([len(C_I), len(rml)])
    ana_result = Evaluation_Analysis(C_I)
    slution_chosen(ana_result)
    return der, rml_copy


def uniform_der(x, threshold):
    if x > threshold:
        return threshold
    elif x < -threshold:
        return -threshold
    else:
        return x


def Clearness_Evaluation(width, percent, CE_threshold):
    """
    信号洁净度评估

    统计R波附近的信号穿过两条确定水平线(R峰值的正负百分之percent)的次数
    将高于一定次数(CE_threshold)的R波标记为非洁净(0) 其余标记为洁净(1)

    parameters:
        width(int): The range of evaluation
        percent(float): The ratio of the hight of horizontal line to the R peak
        CE_threshold(int): The threshold of evaluation

    return:
        list: [clean R wave as 1, dirty R wave as 0]
    """
    Evaluation_List = [0, 0]
    for moment in rml[2:]:
        R_peak_value = ecg[moment]
        data = ecg[moment-width: moment+width]
        translation_positive = (np.array(data) -
                                np.array([R_peak_value*percent]*len(data)))
        translation_negative = (-np.array(data) +
                                np.array([R_peak_value*percent]*len(data)))
        n = 0
        # print([moment, R_peak_value, len(data), len(translation)])
        last = translation_positive[0]
        for current in translation_positive[1:]:
            if (last*current) < 0:
                n = n + 1
            last = current
        last = translation_negative[0]
        for current in translation_negative[1:]:
            if (last*current) < 0:
                n = n + 2
            last = current
        if n < CE_threshold:
            Evaluation_List.append(1)
        else:
            Evaluation_List.append(0)
    return Evaluation_List


def Interval_Evaluation(IE_threshold):
    """
    信号间期评估

    分析间期序列的导数 将导数高于一定值(IE_threshold)的R波标记为异常(0)
    其余标记为正常(1)

    parameters:
        IE_threshold(float): The threshold of evaluation

    return:
        list: [normal R wave as 1, abnormal R wave as 0]
    """
    Evaluation_List = []
    for value in der:
        if abs(value) < IE_threshold:
            Evaluation_List.append(1)
        else:
            Evaluation_List.append(0)
    return Evaluation_List


def Evaluation_Analysis(EA_in):

    # 生成[1, ... 1]
    kernel_size = 5
    kernel = np.array([1]*kernel_size)

    # 连续正常点组的中点标记为1 其他点为0
    continual_detection = convolve(EA_in, kernel)[kernel_size//2:
                                                  -kernel_size//2]
    dict = [0] * kernel_size + [1]
    continual_detection = [dict[x] for x in continual_detection]

    # 得到01和10边界 分别标记为1 -1 其他点为0
    kernel = np.array([1, -1])  # 注意卷积时第一步会将卷积核翻转
    jump_detection = convolve(continual_detection, kernel)[:-1]

    # 将一个异常组打包为一个异常list，其中包括：异常组前连续正常点组的间期均值、异常组起点、
    # 异常组后连续正常点组的间期均值和异常组终点。多个异常list组成EA_out
    EA_out = []
    index = 0
    mean_s = 0
    start = 0
    print(len(jump_detection))
    while index < (len(jump_detection)-kernel_size):
        # 01和01边界是基于连续正常点组的中点标记的 此处从中点反推出连续正常点组的边缘
        if jump_detection[index] == -1:
            start = index + kernel_size//2 - 1
            mean_s = get_mean(-1, start, kernel_size)
        if jump_detection[index] == 1:
            end = index - kernel_size//2
            mean_e = get_mean(1, end, kernel_size)
            EA_out.append([mean_s, start, mean_e, end])
        index = index + 1
    return EA_out


def get_mean(type, index, size):
    """
    计算连续正常点的平均间期
    """
    if type == -1:
        return (rml[index]-rml[index-size])/size
    else:
        return (rml[index+size]-rml[index])/size


def slution_chosen(sc_in):
    global rml_copy
    for item in sc_in[1:]:
        abnomal_gap_length = rml[item[3]] - rml[item[1]]
        mean_interval = 0.5 * (item[0]+item[2])
        # n_max = 1.25 * abnomal_gap_length / mean_interval
        # n_min = 0.83 * abnomal_gap_length / mean_interval
        # N = get_n(n_max, n_min)
        N = [int(round(abnomal_gap_length / mean_interval))]
        if N:
            delete_start = np.argwhere(rml_copy == rml[item[1]])[0][0]
            delete_end = np.argwhere(rml_copy == rml[item[3]])[0][0]
            rml_copy = np.delete(rml_copy, range(delete_start+1, delete_end))
            plan_A(delete_start, abnomal_gap_length, N[0])
            # if N[0] < 6:
            #     plan_A(delete_start, abnomal_gap_length, N[0])
            #     pass
            # else:
            #     pass


def get_n(max, min):
    n_out = []
    for i in range(int(1+min), int(max)+1):
        n_out.append(i)
    return n_out


def plan_A(start, gap_length, n):
    global rml_copy
    assume_interval = gap_length / n
    for i in range(1, n):
        assume_R_location = rml_copy[start] + assume_interval * i
        tolerance = assume_interval * 0.2
        locs, _ = find_peaks(ecg[int(round(assume_R_location-tolerance)):
                                 int(round(assume_R_location+tolerance))],
                             prominence=0.25)
        R_num = len(locs)
        # Make locs in line with R_Moment_List in time
        locs = (np.array([int(round(assume_R_location-tolerance))]*R_num)+locs)

        # if R_num > 0:
        #     closest_distance = abs(locs[0]-assume_R_location)
        #     closest_location = locs[0]
        #     for item in locs[1:]:
        #         current_distance = abs(item-assume_R_location)
        #         if current_distance < closest_distance:
        #             closest_distance = current_distance
        #             closest_location = item
        #     rml_copy = np.insert(rml_copy, start+i, closest_location)

        if R_num > 0:
            largest_rank = cal_rank(locs[0], assume_R_location, tolerance)
            closest_location = locs[0]
            for item in locs[1:]:
                current_rank = cal_rank(item, assume_R_location, tolerance)
                if current_rank > largest_rank:
                    largest_rank = current_rank
                    closest_location = item
            rml_copy = np.insert(rml_copy, start+i, closest_location)
        else:
            rml_copy = np.insert(rml_copy, start+i, assume_R_location)


def cal_rank(x, assume_R_location, tolerance):
    return ecg[x]/(1+math.exp(10*(abs(x-assume_R_location)/tolerance)))


def plan_B():
    pass
