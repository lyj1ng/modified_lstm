import numpy as np

from lstm import LstmParam, LstmNetwork


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """

    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def train_lstm(xdata, ydata, epochs=100, batch_size=16):
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = len(xdata[0])  # 50
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    for cur_iter in range(epochs):
        print("iter", "%2s" % str(cur_iter), end=": ")
        for batch in range(int(len(xdata) / batch_size)+1):
            y_list = ydata[batch * batch_size:(batch + 1) * batch_size]
            input_val_arr = xdata[batch * batch_size:(batch + 1) * batch_size]

            for ind in range(len(y_list)):
                lstm_net.x_list_add(input_val_arr[ind])
            y_pred = [round(lstm_net.lstm_node_list[ind].state.h[0], 2) for ind in range(len(y_list))]

            # print("y_pred = [" +
            #       ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
            #       "]", end=", ")

            # print(y_pred, end=' ')

            loss = lstm_net.y_list_is(y_list, ToyLossLayer)
            # print("loss:", "%.3e" % loss)
            lstm_param.apply_diff(lr=0.1)
            lstm_net.x_list_clear()
        y_list = ydata
        input_val_arr = xdata
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_net.x_list_clear()


if __name__ == "__main__":
    # example_0()
    import math

    i = 0
    list1 = []  # 定义一个空list
    while (i < 360):
        list1.append(i)  # 把数据增加到列表末
        i = i + 3.6;  # 因为我要一个周期里有100个点，所以点间距为3.6度
    # 以上生成了100个点的角度数据
    list2 = [x * math.pi / 180 for x in list1]  # 把角度转成弧度

    list3 = [math.sin(x) for x in list2]  # 求出正弦值,并放大100倍

    list4 = [round(x,2) for x in list3]  # 取整

    list4 *= 2
    y = []
    x = []
    for i in range(500):
        start_idx = np.random.randint(1, len(list4)//2)
        xx = list4[start_idx:start_idx+10]
        yy = list4[start_idx+11]
        x.append(xx)
        y.append(yy)

    train_lstm(x, y, 1000)
    print(y)
