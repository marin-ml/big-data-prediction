
import xlrd
import xlwt
import func


def conv_data(f_in, f_out, type_data):
    book_in = xlrd.open_workbook(f_in)
    sheet_in = book_in.sheet_by_index(0)
    book_out = xlwt.Workbook(encoding="utf-8")
    sheet_out1 = book_out.add_sheet("sheet1")
    sheet_out2 = book_out.add_sheet("sheet2")
    sheet_out_y = book_out.add_sheet("sheet3")

    for i in range(sheet_in.nrows):
        if i % 10 == 0:
            print i
        row_data = sheet_in.row_values(i)
        out_data = func.get_real(type_data, row_data)
        sheet_out_y.write(i, 0, func.bigger0(row_data[1]))
        for j in xrange(1, out_data.__len__()):
            if j < 240:
                sheet_out1.write(i, j - 1, out_data[j])
            else:
                sheet_out2.write(i, j - 240, out_data[j])

    book_out.save(f_out)


print "Reading Type List file ..."
type_list = func.load_type("Type.xls")

conv_data("test1.xlsx", "test.xls", type_list)

# print "Converting Training data 1 file ..."
# conv_data("c1.xlsx", "Training1.xls", type_list)

# print "Converting Training data 2 file ..."
# conv_data("c2.xlsx", "Training2.xls", type_list)
#
# print "Converting Training data 3 file ..."
# conv_data("c3.xlsx", "Training3.xls", type_list)

# print "Converting Verification data file ..."
# conv_data("c4.xlsx", "Verification.xls", type_list)

# print "Converting  Test data file ..."
# conv_data("c5.xlsx", "Test_data.xls", type_list)

