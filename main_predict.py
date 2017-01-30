
import xlrd
import xlwt
import func
import tensorflow as tf

filename = "test.xlsx"

def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


print "Reading Type List file..."
type_list = func.load_type("Type.xls")

print "Reading Data file..."
book_in = xlrd.open_workbook(filename)
sheet_in = book_in.sheet_by_index(0)

print "Standardization Data file..."
x_test = []
for i in xrange(sheet_in.nrows):
    row_data = sheet_in.row_values(i)
    r = func.get_real(type_list, row_data)
    x_test.append(r[1:478])

print "Configuring Model..."
x = tf.placeholder("float", [None, 477])
y = tf.placeholder("float", [None, 2])
W1 = tf.get_variable("W1", shape=[477, 2], initializer=xaver_init(477, 2))
b1 = tf.Variable(tf.zeros([2]))
activation = tf.add(tf.matmul(x, W1), b1)
t1 = tf.nn.softmax(activation)
init = tf.initialize_all_variables()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

print ('load learning...')
saver.restore(sess, 'model_bin.ckpt')

print ('Prediction test data...')
ret = sess.run(t1, feed_dict={x: x_test})
prec_data = func.sel_max(ret)

book_out1 = xlwt.Workbook(encoding="utf-8")
sheet_out1 = book_out1.add_sheet("sheet1")

for i in xrange(sheet_in.nrows):
    sheet_out1.write(i, 0, prec_data[i])

book_out1.save("Data_prediction.xls")
print ('Finished successfully')