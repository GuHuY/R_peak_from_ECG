import numpy as np
import matplotlib.pyplot as plt
import R_detection
import R_refine1
import wfdb
import time
from scipy.interpolate import interp1d

samp_freq = 0  # ECG记录的采样频率


def synthesize(position):
    """
    预处理全过程（读取数据，R波检测，生成散点图(A)，平滑，求导(B)，插值并采样（C），
    贴标签）

    parameter:
        position(str): The position of record
        plot(None or list): [[start, end](second int), [sig1, sig2,...](str)]

    returns:
        list: Preprocessed data with label
    """
    global samp_freq
    time_start = time.time()
    (ecg, annotation, sf, name) = Read_ECG_and_ANN(position)
    samp_freq = sf
    # (R, A) = R_A(ecg)  [:10000*samp_freq]
    (R, A, R_Moment_List) = R_A(ecg)
    #
    b1 = B(A)  # 用于评估间期变化是否异常
    #
    (eva, refine_out) = R_refine1.refine(R[1], R_Moment_List, b1[1], samp_freq)
    A = Zero_Exception(A)
    a = smoothing(A, 3)
    b = B(a)
    c = C(b)
    seg = segment(c[1])
    result = add_labels_into_results(seg, annotation)
    print(['Name: ' + name,
           'ECG: ' + str(len(R[1])),
           'A: ' + str(len(A[1])),
           'Segments: ' + str(len(seg)),
           'Labels: ' + str(len(annotation)),
           'Time: ' + str(float('%.3f' % (time.time()-time_start)))])
    # return result, R, A
    return result, R, A, eva, refine_out


# def image_test(R, A, plot):
def image_test(R, A, plot, E, F, G):
    """
    测试用
    """
    a1 = smoothing(A, 3)
    a2 = smoothing(A, 7)

    b1 = B(A)
    b2 = B(a1)
    b3 = B(a2)

    c1 = C(b2)

    pack_list = []
    for item in plot[1]:
        if item == 'ECG':
            pack_list.append([R, 'mV', 'ECG', 1, A])
        if item == 'A':
            pack_list.append([A, 'ms', 'A', 1, None])
        if item == 'SMO_3':
            pack_list.append([a1, 'ms', 'Smoothing n=3', 1, None])
        if item == 'SMO_7':
            pack_list.append([a2, 'ms', 'Smoothing n=7', 1, None])
        if item == 'DER':
            pack_list.append([b1, '', 'Derivative', 1, ''])
        if item == 'DER_S3':
            pack_list.append([b2, '', 'Derivative S3', 1, ''])
        if item == 'DER_S7':
            pack_list.append([b3, '', 'Derivative S7', 1, ''])
        if item == 'SAM':
            pack_list.append([c1, '', 'sampling', 1, '0'])
        #
        if item == 'EVA':
            pack_list.append([E, '', 'evaluation', 1, ''])
        if item == 'EVA2':
            pack_list.append([G, '', 'evaluation2', 1, ''])
        if item == 'ANA':
            pack_list.append([R, '', 'refined', 1, F])

    draw_plot(pack_list, plot[0])


def Read_ECG_and_ANN(position):
    """
    根据文件地址导入：ECG数据，注释，采样频率，记录名

    parameter:
        position(str): The position of record

    returns:
        list: ECG record
        list: Annotation ('0' stand for no apnea, '1' stand for apnea)
        int: The sampling frequency of ECG record
        str: The name of ECG record
    """
    temp = wfdb.rdrecord(position)
    ecg_data = temp.p_signal
    samp_freq = temp.fs
    reco_name = temp.record_name
    temp = wfdb.rdann(position, 'apn')
    anno_data = temp.symbol
    converter = {'N': 0, 'A': 1}
    return ([x[0] for x in ecg_data],
            [converter[x] for x in anno_data if x in ['N', 'A']],
            samp_freq,
            reco_name)


def R_A(ecg):
    """
    R波检测，散点图A生成, 规范所有单位为毫秒

    parameter:
        ecg(list): Raw ECG record

    returns:
        list: [[X of ECG record], [Y of ECG record]]
        list: [[X of A], [Y of A]]

    """
    R = [np.linspace(0, int(len(ecg)*(1000/samp_freq)),
                     len(ecg),
                     endpoint=False),
         ecg]
    (R_Moment_List, R_Interval_List) = R_detection.select_R(ecg, samp_freq)
    A = [R_Moment_List*(1000/samp_freq), R_Interval_List*(1000/samp_freq)]
    # return R, A
    return R, A, R_Moment_List


def Zero_Exception(Z_in):

    Value = Z_in[1]
    Time = Z_in[0]
    i_list = []
    for i in range(len(Value)):
        if Value[i] == 0:
            i_list.append(i)
    return [np.delete(Time, i_list), np.delete(Value, i_list)]


def smoothing(A, n=3):
    """
    均值滤波器

    parameter:
        B_in(list): The 2nd output of R_A()

    return:
        list: [[X of smoothing result], [Y of smoothing result]]
    """
    Smoothing = R_detection.Moving_Mean_Filter(A[1], n)
    return [A[0], Smoothing]


def B(B_in):
    """
    导数滤波器

    parameter:
        B_in(list): The output of smoothing()

    return:
        list: [[X of derivative result], [Y of derivative result]]
    """
    Derivative = [0]
    for i in range(1, len(B_in[0])):
        Derivative.append((B_in[1][i]-B_in[1][i-1])/(B_in[0][i]-B_in[0][i-1]))
    return [B_in[0], np.array(Derivative)]


def C(B, frequency=6):
    """
    插值后采样

    parameters:
        B(list): The output of B()
        frequency(int): The sampling frequency apply on B

    return:
        list: [[X of sampling result], [Y of sampling result]]
    """
    x_before_samp, ind = np.unique(B[0], return_index=True)
    y_before_samp = B[1][ind]
    # fl为插值结果 linear cubic
    fl = interp1d(x_before_samp, y_before_samp, kind='linear')
    # 采样间隔设置
    x_after_samp = np.linspace(min(x_before_samp),
                               max(x_before_samp//1000*1000),
                               int(max(x_before_samp)//1000*frequency),
                               endpoint=False)
    # 采样
    y_after_samp = fl(x_after_samp)
    return [[int(round(x)) for x in x_after_samp], y_after_samp]


def segment(seg_in, frequency=6):
    """
    将采样结束后的数据分割，每个片段对应60s

    parameters:
        seg_in(list): The 2nd item in the output list of C()
        frequency(int): The sampling frequency apply on B

    return:
        list: [[C[1] in 1st min], [C[1] in 2nd min], ...]
    """
    seg_out = []
    seg_num = len(seg_in)//(frequency*60)
    for i in range(seg_num):
        n = i*frequency*60
        seg_out.append(seg_in[n:n+frequency*60])
    return seg_out


def add_labels_into_results(segment, label):
    """
    将注释(是否有呼吸暂停)添加到对应片段的结尾处

    parameters:
        segment(list): The output of segment()
        label(list): The list of annotation

    return:
        list: [[segment[0][0], ..., segment[0][360], label[0]],
               [segment[1][0], ..., segment[1][360], label[1]], ...]
    """
    list_len = min(len(segment), len(label))
    add_out = []
    for i in range(list_len):
        add_out.append(list(segment[i])+[label[i]])
    return add_out


def draw_plot(pack_list, period=[0, 1]):
    """
    根据输入的pack_list绘图

    parameters:
        pack_list(list): pack_list的每一项必须依次为数据集(lsit),单位(str),
                         标签(str),标签位置(int),辅助线(None 或 str)
        period(list): period必须有切只有两个参数,第一个参数为所有图的时间起点,第二个
                      参数为所有图的时间终点,以秒为单位.
    """
    # period储存以秒为单位的绘图起止时间点，而R_Moment_List以采样序数为单位，
    # 故第一步是将period中的值乘以采样频率得到采样序数，从而得到在R_Moment_List
    # 中的起止点。将R_Moment_List起止点之间的值保存在R_M_S中，并令R_M_S从0开始。

    start = period[0]*1000
    end = period[1]*1000

    plot_amount = len(pack_list)
    plot_sequence = 0
    plt.figure()
    for pack in pack_list:
        plot_sequence = plot_sequence + 1
        plt.subplot(plot_amount, 1, plot_sequence)
        (i, j) = get_border(pack[0][0], start, end)
        X = pack[0][0][i:j]
        Y = pack[0][1][i:j]
        # X = X - np.array([start]*len(X))

        # X轴范围
        plt.xlim(start, end)
        # X,Y轴单位
        if plot_sequence == plot_amount:
            plt.xlabel('Time/ms')
        plt.ylabel(pack[1])

        # 绘图
        plt.plot(X, Y, label=pack[2])
        print(["i:j", i, j], ["time :", pack[0][0][i], pack[0][0][j]])

        # ECG辅助线（R波标记）
        if type(pack[4]) == list:
            (i, j) = get_border(pack[4][0], start, end)
            plt.plot(pack[4][0][i:j],
                     [pack[0][1][int(round(t/1000*samp_freq))]
                      for t in pack[4][0][i:j]],
                     "x",
                     label='R wave')
        # 导数辅助线
        F = np.linspace(start, end, (end-start)//1000)
        if pack[4] == '0':
            plt.plot(F, [0 for x in F], ".")

        # 标签位置x
        plt.legend(loc=pack[3])
    plt.show()


def get_border(list_in, start, end):

    """
    截取从小到大排序的数组中，指定范围的下标上下界

    parameters:
        list_in(list): Input list
        start(float): Low border
        end(float): High border

    returns：
        i(int): Beginning order
        j(int): Terminating order
    """

    i = None
    j = None
    for i in range(len(list_in)):
        if list_in[i] >= start:
            break
    for j in range(len(list_in[i:])):
        if list_in[j] >= end:
            break
    return i, j
