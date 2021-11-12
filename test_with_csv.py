import numpy as np
from lstm_ll import LstmParam, LstmNetwork
from sklearn.model_selection import train_test_split

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


def train_lstm(xdata, ydata, epochs=100, validaion_data=None):
    """
    :param xdata: input
    :param ydata: label
    :param epochs: 迭代轮数
    :param batch_size: time step
    :param validaion_data: 验证集
    :return: model
    """
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)
    end_node = ' ' if validaion_data else '\n'
    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = len(xdata[0][0])  # 50
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    for cur_iter in range(epochs):
        print("iter", "%2s" % str(cur_iter), end=": ")
        total_loss = 0
        for time_step in range(len(xdata)):
            # for batch in range(int(len(xdata) / batch_size) + 1):
            #     y_list = ydata[batch * batch_size:(batch + 1) * batch_size]
            #     input_val_arr = xdata[batch * batch_size:(batch + 1) * batch_size]
            y_list = ydata[time_step]
            input_val_arr = xdata[time_step]

            for ind in range(len(input_val_arr)):
                lstm_net.x_list_add(input_val_arr[ind])
            # y_pred = [round(lstm_net.lstm_node_list[ind].state.h[0], 2) for ind in range(len(y_list))]

            # print("y_pred = [" +
            #       ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
            #       "]", end=", ")

            # print(y_pred, end=' ')

            loss = lstm_net.y_list_is(y_list, ToyLossLayer)
            # print("loss:", "%.3e" % loss)
            lstm_param.apply_diff(lr=0.001)
            lstm_net.x_list_clear()
            total_loss += loss
        print("training loss:", "%.3e" % loss, end=end_node)
        if validaion_data:
            validation(lstm_net, validaion_data[0], validaion_data[1])
    return lstm_net


def validation(model, x, y):
    total_loss = 0
    for y_list, input_val_arr in zip(y, x):
        for ind in range(len(input_val_arr)):
            model.x_list_add(input_val_arr[ind])
        loss = model.y_list_is(y_list, ToyLossLayer)
        total_loss += loss

        model.x_list_clear()
    print("validation loss:", "%.3e" % total_loss)


def predict(model, x):
    res = []
    for time_batch in x:
        for xx in time_batch:
            model.x_list_add(xx)
        y_pred = round(model.lstm_node_list[-1].state.h[0], 2)
        model.x_list_clear()
        res.append(y_pred)
    return res


if __name__ == "__main__":
    vectors = []
    with open('C:\\Users\\forev\\PycharmProjects\\crowd_feature_extraction\\vector.csv', 'r') as fp:
        lines = fp.readlines()[1:]
        for line in lines:
            vector = line.split(',')[1:]
            vectors.append([float(i) for i in vector])
    vectors = np.array(vectors)
    # print(vectors)
    x = []
    y = []
    step, window = 5, 10
    for i in range(window, len(vectors), step):
        x.append(vectors[i-window:i])
        y.append(int(i>len(vectors)//2))
    # print(x)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
    # lstmNet = train_lstm(x, y, epochs=200, )
    lstmNet = train_lstm(X_train, y_train, epochs=50, validaion_data=(X_test, y_test))
    result = predict(lstmNet, X_test)
    print(result, '\n', y_test)
