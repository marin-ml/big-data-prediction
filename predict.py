
import xlrd
import xlwt
import tensorflow as tf


def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def sel_max(data, col_cnt):
    ret_ind = []
    for i in xrange(col_cnt):
        if data[i][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)

    return ret_ind


learning_rate = 0.01
x_test = []
y_test = []


print "Loading Test data ..."
book_x = xlrd.open_workbook("Test_data.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
ROWS_TEST = sheet_x1.nrows

for i in xrange(ROWS_TEST):
    x_test.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))


x = tf.placeholder("float", [None, 476])
y = tf.placeholder("float", [None, 2])  # 0-9 digits recognition => 10 classes

W1 = tf.get_variable("W1", shape=[476, 2], initializer=xaver_init(476, 2))

b1 = tf.Variable(tf.zeros([2]))

activation = tf.add(tf.matmul(x, W1), b1)  # Softmax

t1 = tf.nn.softmax(activation)

# Initializing the variables
init = tf.initialize_all_variables()

sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

print ('load learning ...')
saver.restore(sess, 'model_bin.ckpt')

print ('Prediction test data ...')
ret = sess.run(t1, feed_dict={x: x_test})
prec_data = sel_max(ret, ROWS_TEST)

book_out1 = xlwt.Workbook(encoding="utf-8")
sheet_out1 = book_out1.add_sheet("sheet1")

for i in xrange(ROWS_TEST):
    sheet_out1.write(i, 0, prec_data[i])

book_out1.save("Data_prediction.xls")
print ('Finished successfully')

