'''
代码名称： gen_case_1.py
所需编译器： python3.6
所需开源代码包： NumPy, Pandas
说明： 打印出运行情况&完成工件数&加工序列，并将其输出为excel文件
        （"Case_1_result.xls"和"Case_3_result_1.xls"文件的数据的生成来源）
        参数调节（决定组数，决定有无故障情况——通过调节故障率是否为0）在主函数部分。
'''

import numpy as np
import pandas as pd
from datetime import datetime

class CNC:
    def __init__(self, reload_time, work_time, loc, num):
        self.loc = loc
        self.is_empty = True
        self.end_time = 0 # 当前state结束时间
        self.reload_time = reload_time
        self.work_time = work_time
        self.is_breakdown = False
        self.number = num
        self.record = np.array([self.number, 0, 0]) # 记录每次工序的内容
    def error_state_updating(self, time):
        if self.is_breakdown:
            if time > self.end_time:
                self.is_breakdown = False
    def re_initialize_record(self):
        if self.record[2] != 0:
            # 可以打印
            self.print_to_data(data)
        self.record = np.array([self.number, 0, 0])
        # 重新更新时间记录
    def print_to_data(self, data):
        # 将内容打印到data
        data.append(self.record)

class RGV:
    def __init__(self, move_time, cncs, wash_time):
        self.loc = 1
        self.total_time = 0
        self.move_time = move_time
        self.cncs = cncs
        self.wash_time = wash_time
        self.completed = 0
    def move_to(self, id):
        # 移动到id对应的cnc处
        move_time = self.move_time[abs(self.loc-(int(id/2)+1))]
        self.total_time += move_time
        # print(f"移动到{id+1}#处：loc：[{self.loc}->{self.cncs[id].loc}], 移动时间 = {move_time}")
        self.loc = self.cncs[id].loc
    def add_workpiece(self, id):
        # empty->加件
        # print(f"给{id+1}#加件")
        order_list.append(id)
        self.cncs[id].is_empty = False
        self.cncs[id].record[1] = self.total_time   # 记载上料开始时间
        self.total_time += self.cncs[id].reload_time
        if 0.24 <= np.random.rand() < 0.24+err_rate:
            self.error_happen(id)
        else:
            self.cncs[id].end_time = self.total_time+self.cncs[id].work_time
    def change_workpiece(self, id):
        # waiting->换件+清洗
        # print(f"给{id+1}#换件+清洗")
        order_list.append(id)
        self.cncs[id].is_empty = False
        self.cncs[id].record[2] = self.total_time
        self.cncs[id].re_initialize_record()
        self.cncs[id].record[1] = self.total_time   # 记载上料开始时间
        self.total_time += self.cncs[id].reload_time
        if 0.24 <= np.random.rand() < 0.24+err_rate:
            self.error_happen(id)
        else:
            self.cncs[id].end_time = self.total_time+self.cncs[id].work_time
        self.total_time += self.wash_time
        self.completed += 1
        # print(f"{i+1}#已加工完成")
    def error_happen(self, error_ind):
        error_time_minute = 10+np.random.rand()*10
        print(f"{error_ind+1}# 发生了 {error_time_minute} 分钟的故障")
        self.cncs[error_ind].is_breakdown = True
        self.cncs[error_ind].is_empty = True # 拆下物件清空
        err_happen_time = self.total_time + self.cncs[error_ind].work_time*np.random.rand()
        fix_time = error_time_minute*60
        err_data.append([error_ind+1, err_happen_time, err_happen_time+fix_time])
        self.cncs[error_ind].end_time = err_happen_time + fix_time
    def find_best_obj(self):
        # 寻找最优对象
        min_val = 1000
        min_ind = 0
        for i in range(len(self.cncs)):
            go_time = 0
            # 非空
            if self.cncs[i].end_time > self.total_time:
                # 未加工完的
                go_time += max(self.move_time[abs(self.loc-(int(i/2)+1))], self.cncs[i].end_time-self.total_time)+self.cncs[i].reload_time
            else:
                # 已经加工完了
                go_time += self.move_time[abs(self.loc-(int(i/2)+1))]+self.cncs[i].reload_time
            if go_time < min_val:
                min_val = go_time
                min_ind = i
        return min_ind

def workpieces_in_time(work_time):
    # CNC对象初始化
    cnc1 = CNC(cnc_reload_time_1, work_time_val, loc=1, num=1)
    cnc2 = CNC(cnc_reload_time_2, work_time_val, loc=1, num=2)
    cnc3 = CNC(cnc_reload_time_1, work_time_val, loc=2, num=3)
    cnc4 = CNC(cnc_reload_time_2, work_time_val, loc=2, num=4)
    cnc5 = CNC(cnc_reload_time_1, work_time_val, loc=3, num=5)
    cnc6 = CNC(cnc_reload_time_2, work_time_val, loc=3, num=6)
    cnc7 = CNC(cnc_reload_time_1, work_time_val, loc=4, num=7)
    cnc8 = CNC(cnc_reload_time_2, work_time_val, loc=4, num=8)

    cncs = list([cnc1, cnc2, cnc3, cnc4, cnc5, cnc6, cnc7, cnc8])

    rgv = RGV(move_time, cncs, wash_time)

    while rgv.total_time < work_time:
        aim_id = rgv.find_best_obj()
        rgv.move_to(aim_id)
        for cnc in cncs:
            cnc.error_state_updating(rgv.total_time)
            # cnc 状态更新
        if rgv.cncs[aim_id].is_empty:
            rgv.add_workpiece(aim_id)
        else:
            if rgv.total_time < rgv.cncs[aim_id].end_time:
                rgv.total_time = rgv.cncs[aim_id].end_time
            rgv.change_workpiece(aim_id)


if __name__ == '__main__':



    # 【参数列表】-- begin

    # for 1#，3#，5#，7#
    cnc_reload_time_1 = 27
    # for 2#，4#，6#，8#
    cnc_reload_time_2 = 32
    work_time_val = 545
    move_time = [0, 18, 32, 46]
    wash_time = 25
    err_rate = 0.01
    # 【参数列表】-- end



    data = []
    err_data = []

    beg = datetime.now()

    order_list = []
    workpieces_in_time(8 * 60 * 60)
    data = np.array(data)
    print('工件完成数', len(data))
    print(data[:, 0])

    if err_rate > 0:
        # ATTENTION：***如果报错了 就是工作运行期间没有故障**
        is_err_str = 'err'
        data_err_df = pd.DataFrame(err_data)
        data_err_df.columns = ['故障cnc编号', '故障开始时间', '故障结束时间']
        data_err_df.index += 1
        # create and writer pd.DataFrame to excel
        writer = pd.ExcelWriter('Save_Excel_case1_' + is_err_str + '_how.xlsx')
        data_err_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
        writer.save()
    else:
        is_err_str = 'no_err'

    en = datetime.now()

    # 生成excel表格
    # 注意组号
    data_df = pd.DataFrame(data)
    # change the index and column name
    data_df.columns = ['加工编号', '上料时间', '下料时间']
    data_df.index += 1
    # create and writer pd.DataFrame to excel
    writer = pd.ExcelWriter('Save_Excel_case1_' + is_err_str + '.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
    writer.save()

    print(f"Time cost: {en.microsecond-beg.microsecond} μs")