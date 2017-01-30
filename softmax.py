
import xlrd
import tensorflow as tf


def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def acc(d1, d2, col_cnt):
    cnt = 0
    for i in xrange(col_cnt):
        if d1[i] == d2[i]:
            cnt += 1

    return float(cnt)/col_cnt


def sel_max(data, col_cnt):
    ret_ind = []
    for i in xrange(col_cnt):
        if data[i][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)

    return ret_ind

def expand(d):
    s = [0, 0]
    s[d] = 1
    return s

learning_rate = 0.01
x_training = []
y_training = []
x_verification = []
y_verification = []

COLS1 = COLS3 = COLS4 = COLS2 = 0

print "Loading Training data 1..."
book_x = xlrd.open_workbook("Training1.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)
COLS1 = 65000

for i in xrange(COLS1):
    x_training.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_training.append(expand(int(sheet_y.cell(i, 0).value)))

print "Loading Training data 2..."
book_x = xlrd.open_workbook("Training2.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)
COLS2 = 65000

for i in xrange(COLS2):
    x_training.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_training.append(expand(int(sheet_y.cell(i, 0).value)))

print "Loading Training data 3..."
book_x = xlrd.open_workbook("Training3.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)
COLS3 = 19999

for i in xrange(COLS3):
    x_training.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_training.append(expand(int(sheet_y.cell(i, 0).value)))

print "Loading Verification data..."
book_x = xlrd.open_workbook("Verification.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)
COLS4 = 16906

for i in xrange(COLS4):
    x_verification.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_verification.append(expand(int(sheet_y.cell(i, 0).value)))

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

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Gradient Descent
#
# # Initializing the variables
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

print ('load learning...')
saver.restore(sess, 'model.ckpt')

cols = COLS1 + COLS2 + COLS3
# Training cycle
for step in range(5000):
    sess.run(optimizer, feed_dict={x: x_training, y: y_training})
    if step % 5 == 0:
        ret = sess.run(t1, feed_dict={x: x_training})
        ret1 = sel_max(ret, cols)
        acc1 = 100 - acc(ret1, sel_max(y_training, cols), cols)*100

        ret2 = sess.run(t1, feed_dict={x: x_verification})
        ret3 = sel_max(ret2, COLS4)
        acc2 = 100 - acc(ret3, sel_max(y_verification, COLS4), COLS4)*100
        print step, sess.run(cost, feed_dict={x: x_training, y: y_training}), acc1, acc2, [ret1.count(0), ret1.count(1)]

        saver.save(sess, 'model.ckpt')

print ("Optimization Finished!")



