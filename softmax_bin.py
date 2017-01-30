
import xlrd
import tensorflow as tf


def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def acc(d1, d2):
    cnt = 0
    for ii in xrange(d1.__len__()):
        if d1[ii] == d2[ii]:
            cnt += 1

    return float(cnt)/d1.__len__()


def sel_max(data):
    ret_ind = []
    for ii in xrange(data.__len__()):
        if data[ii][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)

    return ret_ind


def expand(d):
    s = [0, 0]
    s[d] = 1
    return s

learning_rate = 0.00001
x_training = []
y_training = []
x_verification = []
y_verification = []

print "Loading Training data 1..."
book_x = xlrd.open_workbook("Training1.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)

for i in xrange(sheet_x1.nrows):
    x_training.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_training.append(expand(int(sheet_y.cell(i, 0).value)))

print "Loading Training data 2..."
book_x = xlrd.open_workbook("Training2.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)

for i in xrange(sheet_x1.nrows):
    x_training.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_training.append(expand(int(sheet_y.cell(i, 0).value)))

print "Loading Training data 3..."
book_x = xlrd.open_workbook("Training3.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)

for i in xrange(sheet_x1.nrows):
    x_training.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_training.append(expand(int(sheet_y.cell(i, 0).value)))

print "Loading Verification data..."
book_x = xlrd.open_workbook("Verification.xls")
sheet_x1 = book_x.sheet_by_index(0)
sheet_x2 = book_x.sheet_by_index(1)
sheet_y = book_x.sheet_by_index(2)

for i in xrange(sheet_x1.nrows):
    x_verification.append(sheet_x1.row_values(i) + sheet_x2.row_values(i))
    y_verification.append(expand(int(sheet_y.cell(i, 0).value)))

x = tf.placeholder("float", [None, 476])
y = tf.placeholder("float", [None, 2])

W1 = tf.get_variable("W1", shape=[476, 2], initializer=xaver_init(476, 2))
b1 = tf.Variable(tf.zeros([2]))
activation = tf.add(tf.matmul(x, W1), b1)
t1 = tf.nn.softmax(activation)

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Gradient Descent

# Initializing the variables
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

print ('load learning...')
saver.restore(sess, 'model_bin.ckpt')

# Training cycle
for step in range(5000):
    sess.run(optimizer, feed_dict={x: x_training, y: y_training})
    if step % 5 == 0:
        ret = sess.run(t1, feed_dict={x: x_training})
        ret1 = sel_max(ret)
        acc1 = acc(ret1, sel_max(y_training))*100

        ret2 = sess.run(t1, feed_dict={x: x_verification})
        ret3 = sel_max(ret2)
        acc2 = acc(ret3, sel_max(y_verification))*100
        print step, sess.run(cost, feed_dict={x: x_training, y: y_training}), acc1, acc2, [ret1.count(0), ret1.count(1)]

        saver.save(sess, 'model_bin.ckpt')

print ("Optimization Finished!")
