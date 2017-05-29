
import xlrd


def load_type(fname):
    book_type = xlrd.open_workbook(fname)
    sheet_in = book_type.sheet_by_index(0)

    type_list = []
    for j in range(sheet_in.nrows):
        type_val = int(sheet_in.cell(j, 1).value)
        list_data = [type_val]
        if type_val == 2:
            list_data.append(int(sheet_in.cell(j, 2).value))
            list_data.append(int(sheet_in.cell(j, 3).value))
        elif type_val == 3:
            cnt = int(sheet_in.cell(j, 2).value)
            list_data.append(cnt)
            for i in range(cnt):
                list_data.append(sheet_in.cell(j, i + 3).value)

        type_list.append(list_data)

    return type_list


def kill_space(char):
    if char == '':
        return -1
    else:
        return char


def bigger0(char):
    if char > 0:
        return 1
    else:
        return 0


def get_real(typ, d_in):
    d_out = []
    for i in range(d_in.__len__()):
        type_data = typ[i][0]
        if type_data == 1:
            if d_in[i] == '' or d_in[i] == ' ':
                d_out.append(-1)
            else:
                d_out.append(d_in[i])
        elif type_data == 2:
            if d_in[i] == '':
                d_out.append(-1)
            elif typ[i][1] == typ[i][2]:
                d_out.append(0)
            else:
                d_out.append(10 + int((d_in[i]-typ[i][1])/(typ[i][2]-typ[i][1])*500))
        elif type_data == 3:
            if d_in[i] == '':
                d_out.append(-1)
            else:
                cnt_list = typ[i][1]
                ret = 0
                for j in range(cnt_list):
                    if d_in[i] == typ[i][j+2]:
                        ret = j+1
                d_out.append(ret)

    return d_out


def sel_max(data):
    ret_ind = []
    for i in range(data.__len__()):
        if data[i][0] == 1:
            ret_ind.append(0)
        else:
            ret_ind.append(1)

    return ret_ind
