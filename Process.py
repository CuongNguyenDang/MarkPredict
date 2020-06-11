# ***************************************************************************************
# Dự đoán điểm thi THPT và gợi ý trường-ngành học cho học sinh dựa trên điểm tổng kết   *
# Sai số theo RMSE trên tập test: khoảng 110 (<180, chấp nhận được)                     *
#                                                                                       *
# Nguyễn Đăng Cương - 1811640                                                           *    
#                                                                 Bài tập tết 2020 BKAIC*
# ***************************************************************************************



import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

#Global variable
#=====================================================================================
Ten = 'Ten'
Toan = 'Toán'
Van = 'Ngữ văn'
Li = 'Vật lí'
Hoa = 'Hóa học'
Sinh = 'Sinh học'
Su = 'Lịch sử'
Dia = 'Địa lý'
GDCD = 'GDCD'
Anh = 'Ngoại ngữ'

gpa_columns = [Ten, Toan, Van, Li, Hoa, Sinh, Su, Dia, GDCD, Anh]
exam_columns = [ '_' + x for x in gpa_columns]
exam_columns[0] = Ten

subs = ['toan', 'van', 'li', 'hoa', 'sinh', 'su', 'dia', 'gdcd', 'anh']
eps = 0.5
factor = {}
comb = {
        'A00' : 0,                # A00 Toán lí hóa
        'A01' : 0,                # A01 toán lí anh
        'A02' : 0,                # A02 toán lí sinh
        'B00' : 0,                # B00 toán hóa sinh
        'B07' : 0,                # B07 toán hóa anh
        'C00' : 0,                # C00 văn sử địa
        'D01' : 0,                # D01 toán văn anh
        'D07' : 0,                # D07 toán hóa anh
        'D09' : 0,                # D09 toán sử anh
}
pd.set_option('display.max_colwidth', None)

#function
#=====================================================================================
def LoadData(list_file):
    """Load list file excel"""
    data = pd.DataFrame()
    for file_name in list_file:
        subdata = pd.read_excel(file_name,encoding = 'utf-8',sheet_name = None, skiprows = 0)
        subdata = pd.concat(subdata,ignore_index = True, sort = False)
        
        if 'Họ tên' in subdata.columns:
            subdata = subdata.rename(columns = {'Họ tên': Ten})
        if 'Họ và tên' in subdata.columns:
            subdata = subdata.rename(columns = {'Họ và tên': Ten})
        if 'Hóa' in subdata.columns:
            subdata = subdata.rename(columns = {'Hóa': Hoa})
        if 'Lí' in subdata.columns:
            subdata = subdata.rename(columns = {'Lí': Li})
        if 'Sinh' in subdata.columns:
            subdata = subdata.rename(columns = {'Sinh': Sinh})
        if 'Văn' in subdata.columns:
            subdata = subdata.rename(columns = {'Văn': Van})
        if 'Sử' in subdata.columns:
            subdata = subdata.rename(columns = {'Sử': Su})
        if 'Địa' in subdata.columns:
            subdata = subdata.rename(columns = {'Địa': Dia})
        if 'Ng.ngữ' in subdata.columns:
            subdata = subdata.rename(columns = {'Ng.ngữ': Anh})

        data = pd.concat([data,subdata],ignore_index = True, sort = False)
    
    
    data = data.dropna(how='all', axis=1)   #drop empty columns
    data = data.drop(columns = ['STT']) #drop column 'STT'
    data = data.fillna(0)                   #replace value NaN by 0

    data.columns = gpa_columns
    data.index = data.Ten.tolist()         #set primary key by Name
    data = data.loc[~data.index.duplicated(keep='first')]   #remove rows with duplicate index

    return data

def split_dataset(data):
    """Split dataset to training_set and test_set"""
    N = len(data)     #num of rows
    trainSize = int(N*0.8)
    #testSize = N - trainSize
    train = pd.DataFrame(columns = data.columns)
    listIndex = [1]
    while len(listIndex) < trainSize:
        idx = rd.randint(0,N-1)
        if idx not in listIndex:
            train = train.append(data.loc[idx])
            listIndex.append(idx)
            data = data.drop(index = idx)
    train = train.reset_index(drop = True)
    test = data.reset_index(drop = True)

    return (train,test)

def LinearRegression(train,label):
    #   y = b0 + b1 * x

    x = train
    y = label

    #ignore label = 0
    list_idx = np.where(y == 0)[0].tolist()
    x = x.drop(list_idx)
    y = y.drop(list_idx)

    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return (b_0, b_1) 

def add_comb(_comb, mark):
    for item in _comb:
        comb[item] += mark
    return 1

def calc(b,x):
    ret = b[0] + b[1] * x
    if ret > 0: 
        return round(ret,2)
    else:
        return 0
    
def Prediction(toan, van, li, hoa, sinh, su, dia, gdcd, anh):
    #Write to html file
    #=====================================================================================
    input_score = [toan, van, li, hoa, sinh, su, dia, gdcd, anh]

    result_file = open('templates/result.html',"w",encoding = 'utf-8')
    result_file.writelines('{% extends "index.html" %}')
    result_file.writelines('{% block result %}')

    result_file.writelines('<tr>')
    result_file.writelines('<td>Điểm dự đoán</td>')
    j = 0
    
    #reset mark data
    for i in comb:
        comb[i] = 0

    for i in subs:
        output_score = calc(factor[i],input_score[j])
        if j == 0: #Toan
            add_comb(['A00','A01','A02','B00','B07','D01','D07','D09'],output_score)
        elif j == 1: # Van
            add_comb(['C00','D01'],output_score)
        elif j == 2: # Li
            add_comb(['A00','A01','A02'],output_score)
        elif j == 3: # Hoa
            add_comb(['A00','B00','B07','D07'],output_score)
        elif j == 4: # Sinh
            add_comb(['A02','B00'],output_score)
        elif j == 5: # Su
            add_comb(['C00','D09'],output_score)
        elif j == 6: # Dia
            add_comb(['C00'],output_score)
        elif j == 8: # Anh
            add_comb(['A01','B07','D01','D07','D09'],output_score)

        j = j + 1
        result_file.writelines('<td>' + str(output_score) + '</td>')
    result_file.writelines('</tr>')

    result_file.writelines('{% endblock %}')

    #Predict university
    #==================================================================================
    #print(comb)
    data_uni = pd.read_excel('diemchuanOfficial.xlsx',encoding = 'utf-8')
    _comb = data_uni.columns[2]
    _mark = data_uni.columns[3]
    for i in range(len(data_uni)):
        target_comb = data_uni.loc[i][_comb]
        target_mark = data_uni.loc[i][_mark]
        my_mark = comb[target_comb]
        if target_mark - my_mark > eps:
            data_uni = data_uni.drop(i)
    
    #display setting
    data_uni.sort_values(by = _mark, ascending = False)
    data_uni = data_uni.reset_index(drop = True)
    data_uni = data_uni.reset_index()
    data_uni = data_uni.rename(columns = {'index': 'STT'})
    result_uni = data_uni.head(10)
    

    result_file.writelines('{% block uni %}')
    if (len(result_uni) == 0):
        result_file.writelines("""
        Không tìm được trường phù hợp!
        """)
    else:
        result_file.writelines(result_uni.to_html(index = False))
    result_file.writelines('{% endblock %}')

    result_file.close()
    return 0
    #=====================================================================================

#main function
#=====================================================================================
data_exam = LoadData(['diem_tn_NK.xls','diem_tn_NQ.xls', 'diem_tn_PL.xls', 'diem_tn_PR.xls'])
data_gpa = LoadData(['so_diem_tong_ket_khoi_khoi_12_NK.xls',
                'so_diem_tong_ket_khoi_khoi_12_NQ.xls', 'so_diem_tong_ket_khoi_khoi_12_PL.xls', 
                'so_diem_tong_ket_khoi_khoi_12_PR.xls'])
data_exam.columns = exam_columns
data_gpa = data_gpa.drop(columns = 'Ten')
data = pd.concat([data_gpa,data_exam], axis = 1, join = 'inner')
data = data.reset_index(drop = True)

(train,test) = split_dataset(data)

#training 
for subject_lower,subject_upper in zip(subs,gpa_columns[1:]):
    factor.setdefault(subject_lower,LinearRegression(train[subject_upper], train['_'+subject_upper]))

#print(factor)

#calc error (RMSE)
error = 0
for subject_lower,subject_upper in zip(subs,gpa_columns[1:]):
    calc_test = calc(factor[subject_lower],test[subject_upper].all())
    error += sum(test['_' + subject_upper] - calc_test) ** 2

error = (error / len(test)) ** 0.5
print('\n\t\tRMSE : ',error,'\n')

