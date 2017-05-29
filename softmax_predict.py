
import xlrd
import xlwt
import tensorflow as tf


def sel_max(data, col_cnt):
    ret_ind = []
    for i in range(col_cnt):
        if data[i][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)

    return ret_ind


def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


learning_rate = 0.01
x_test = []
y_test = []


print("Loading Test data...")
book_x = xlrd.open_workbook("Test_data.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
COLS_TEST = 16906

for i in range(COLS_TEST):
    x_test.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))


x = tf.placeholder("float", [None, 477])
y = tf.placeholder("float", [None, 2])  # 0-9 digits recognition => 10 classes

W1 = tf.get_variable("W1", shape=[477, 300], initializer=xaver_init(477, 300))
W2 = tf.get_variable("W2", shape=[300, 120], initializer=xaver_init(300, 120))
W3 = tf.get_variable("W3", shape=[120, 2], initializer=xaver_init(120, 2))

b1 = tf.Variable(tf.zeros([300]))
b2 = tf.Variable(tf.zeros([120]))
b3 = tf.Variable(tf.zeros([2]))

L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))  # Softmax
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))  # Softmax
activation = tf.add(tf.matmul(L2, W3), b3)  # Softmax

t1 = tf.nn.softmax(activation)

# Initializing the variables
init = tf.initialize_all_variables()

sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

print ('load learning...')
saver.restore(sess, 'model.ckpt')

print ('Prediction test data...')
ret = sess.run(t1, feed_dict={x: x_test})
prec_data = sel_max(ret, COLS_TEST)

book_out1 = xlwt.Workbook(encoding="utf-8")
sheet_out1 = book_out1.add_sheet("sheet1")

for i in range(COLS_TEST):
    sheet_out1.write(i, 0, 1-prec_data[i])

book_out1.save("softmax_Predictor.xls")
print ('Finished successfully')

