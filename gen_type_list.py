
import xlrd
import xlwt


print "Reading Data file!"
book_in = xlrd.open_workbook("Data set external prepared.xlsx")
sheet_in1 = book_in.sheet_by_index(0)

print "Reading Type file!"
COLS = 578
ROWS = 186908
book_type = xlrd.open_workbook("Attribute List.xlsx")
sheet_in2 = book_type.sheet_by_index(0)

book_out1 = xlwt.Workbook(encoding="utf-8")
sheet_out1 = book_out1.add_sheet("sheet1")

row_ind = 0
for j in range(COLS):
    MAX_VAL = 0
    MIN_VAL = 999999999
    list_data = []
    list_count = 0
    print j

    type = sheet_in2.cell(j, 1).value
    sheet_out1.write(j, 0, sheet_in2.cell(j, 0).value)
    sheet_out1.write(j, 1, type)

    if type == 2:
        for i in range(ROWS - 1):
            cell_value = sheet_in1.cell(i + 1, j).value
            if cell_value == '':
                cell_value = 0
            MAX_VAL = max(MAX_VAL, int(cell_value))
            MIN_VAL = min(MIN_VAL, int(cell_value))

        sheet_out1.write(j, 2, MIN_VAL)
        sheet_out1.write(j, 3, MAX_VAL)

    elif type == 3:
        list_data = ['']
        list_count = 1

        for i in range(ROWS - 1):
            cell_value = sheet_in1.cell(i + 1, j).value
            comp_ind = 0
            for k in range(list_count):
                if cell_value == list_data[k]:
                   comp_ind = 1

            if comp_ind == 0 and list_count < 254:
                list_data.append(cell_value)
                list_count += 1
                sheet_out1.write(j, list_count+1, cell_value)

        sheet_out1.write(j, 2, list_count-1)

book_out1.save("Type.xls")
print "Generated Type List file successfully!"



