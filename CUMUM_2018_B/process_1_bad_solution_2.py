'''
代码名称： process_1_bad_solution_2.py
所需编译器： python3.6
所需开源代码包： NumPy, matplotlib
说明： 打印出运行情况&完成工件数&加工序列
        基于遗传退火算法的单工序无故障加工效果检验
          该算法复杂，计算量大（速度慢），参数难定，无动态性能（无法处理故障情况），且随机性很强。
          用于和本论文的主要模型进行对比。
      参数调节（决定组数，决定有无故障情况——通过调节故障率是否为0）在主函数部分。
'''
import numpy as np
import matplotlib.pyplot as plt
import random
import time

class CNC:
    def __init__(self, reload_time, work_time, loc):
        self.loc = loc
        self.is_empty = True
        self.end_time = 0 # 当前state结束时间
        self.reload_time = reload_time
        self.work_time = work_time

class RGV:
    def __init__(self, move_time, cncs, wash_time):
        self.loc = 1
        self.total_time = 0
        self.move_time = move_time
        self.cncs = cncs
        self.wash_time = wash_time
    def move_to(self, id):
        # 移动到id对应的cnc处
        move_time = self.move_time[abs(self.loc-(int(id/2)+1))]
        self.total_time += move_time
        # print(f"移动到{id+1}#处：loc：[{self.loc}->{self.cncs[id].loc}], 移动时间 = {move_time}")
        self.loc = self.cncs[id].loc
    def add_workpiece(self, id):
        # empty->加件
        # print(f"给{id+1}#加件")
        self.cncs[id].is_empty = False
        self.total_time += self.cncs[id].reload_time
        self.cncs[id].end_time = self.total_time+self.cncs[id].work_time
    def change_workpiece(self, id):
        # waiting->换件+清洗
        # print(f"给{id+1}#换件+清洗")
        self.cncs[id].is_empty = False
        self.total_time += self.cncs[id].reload_time
        self.cncs[id].end_time = self.total_time+self.cncs[id].work_time
        self.total_time += self.wash_time
        # print(f"{i+1}#已加工完成")
    def fast2remain(self):
        # 结束的回合，要把已经加载的料全部取下来
        min_val = 1000
        min_ind = 0
        for i in range(len(self.cncs)):
            go_time = 0
            if not self.cncs[i].is_empty:
                # 非空
                if self.cncs[i].end_time > self.total_time:
                    # 未加工完的
                    go_time += max(self.move_time[abs(self.loc-(int(i/2)+1))], self.cncs[i].end_time-self.total_time)-self.cncs[i].reload_time
                else:
                    # 已经加工完了
                    go_time += self.move_time[abs(self.loc-(int(i/2)+1))]-self.cncs[i].reload_time
            else:
                go_time = 10000 # 10^6很大，表示不用去了
            if go_time < min_val:
                min_val = go_time
                min_ind = i
        if not self.cncs[min_ind].is_empty:
            self.move_to(min_ind)
            if self.cncs[min_ind].end_time > self.total_time:
                self.total_time = self.cncs[min_ind].end_time
                self.change_workpiece(min_ind)
            else:
                self.change_workpiece(min_ind)
            self.cncs[min_ind].is_empty = True

def single_process_time(working_list):
    # CNC对象初始化
    cnc1 = CNC(cnc_reload_time_1, work_time_val, loc=1)
    cnc2 = CNC(cnc_reload_time_2, work_time_val, loc=1)
    cnc3 = CNC(cnc_reload_time_1, work_time_val, loc=2)
    cnc4 = CNC(cnc_reload_time_2, work_time_val, loc=2)
    cnc5 = CNC(cnc_reload_time_1, work_time_val, loc=3)
    cnc6 = CNC(cnc_reload_time_2, work_time_val, loc=3)
    cnc7 = CNC(cnc_reload_time_1, work_time_val, loc=4)
    cnc8 = CNC(cnc_reload_time_2, work_time_val, loc=4)
    cncs = list([cnc1, cnc2, cnc3, cnc4, cnc5, cnc6, cnc7, cnc8])
    rgv = RGV(move_time, cncs, wash_time)
    for i in working_list:
        i = int(i)
        cnc = cncs[i]
        rgv.move_to(i)
        if cnc.is_empty:
            rgv.add_workpiece(i)
        elif cnc.end_time > rgv.total_time:
            rgv.total_time = cnc.end_time
            rgv.change_workpiece(i)
        else:
            rgv.change_workpiece(i)
    for j in range(8):
        # 8次之内清理所有物品
        rgv.fast2remain()
    # print(rgv.total_time/60/60)
    return rgv.total_time
    # return in second form

def get_eval(lines):
    # 计算一堆数据的适应度
    evals = np.zeros(len(lines)) # 按行数初始化适应度列
    for i, line in enumerate(lines):
        evals[i] = single_process_time(line)
    evals = (np.max(evals)-evals)
    evals = evals/np.sum(evals)
    return evals
    # 返回占其和的百分比

def random_gene(l_eval):
    # 根据随机数产生计数分布矩阵，即这些随机数在这些区间中出现的次数
    length = len(l_eval)
    cnts = np.zeros(length) # 计数点
    rands = np.random.rand(length) # 随机点
    for r in rands:
        split = 0
        for k, eval in enumerate(l_eval):
            split += eval
            if r < split:
                cnts[k] += 1
                break
    return cnts

def new_rand_lines(line_cnt, original_lines):
    # 输入各序列出现次数（来自random_gene）输出被选择的新的序列
    next_to_fill = 0
    new_lines_ = np.zeros(original_lines.shape)
    for n, times in enumerate(line_cnt):
        if times != 0:
            for i in range(int(times)):
                new_lines_[next_to_fill] = original_lines[n]
                next_to_fill += 1
    return new_lines_

def random_pairing(in_lines):
    # 随机配对
    randed_lines = np.random.permutation(in_lines)
    single_line_width = len(randed_lines[0])
    pair_num = int((len(in_lines)/2))
    randed_pairs = np.zeros((pair_num,2,single_line_width))
    for i in range(pair_num):
        randed_pairs[i,0] = randed_lines[2*i]
        randed_pairs[i,1] = randed_lines[2*i+1]
    return randed_pairs

def cross_group_genes(pairs):
    lines = np.zeros((len(pairs)*2,line_width))
    for i, pair in enumerate(pairs):
        for j, line in enumerate(pair):
            lines[2*i+j]=line
    f = get_eval(lines)
    f_max_group = f.max()  # 种群最大适应度
    f_avg_group = f.mean()  # 种群适应度之和
    size = int(len(f) / 2)
    original_lines = np.zeros((len(pairs)*2,line_width)) # pairs -> lines
    for k in range(size):
        original_lines[2 * k] = pairs[k, 0]
        original_lines[2 * k + 1] = pairs[k, 1]
        pair_max_f = np.maximum(f[2*k],f[2*k+1])
        if pair_max_f >= f_avg_group:
            if np.random.rand() < (f_max_group-pair_max_f)/(f_max_group-f_avg_group):
                new_pair = cross_single_gene([lines[2*k], lines[2*k+1]])
                lines[2*k] = new_pair[0]
                lines[2*k+1] = new_pair[1]
        else:
            new_pair = cross_single_gene([lines[2 * k], lines[2 * k + 1]])
            lines[2 * k] = new_pair[0]
            lines[2 * k + 1] = new_pair[1]
    var_lines,f_after_variation = variation(lines)
    return var_lines, original_lines, f_after_variation, f

def variation(ready_lines):
    # 变异
    f_ = get_eval(ready_lines)
    f_max_group_ = f_.max()
    f_avg_group_ = f_.mean()
    for l, line in enumerate(ready_lines):
        if f_[l] >= f_avg_group_:
            if np.random.rand() < var_rate*(f_max_group_-f_[l])/(f_max_group_-f_avg_group_):
                ready_lines[l] = variate_line(line)
        else:
            if np.random.rand() < var_rate:
                ready_lines[l] = variate_line(line)
    return ready_lines, f_

def variate_line(v_line):
    size = len(v_line)
    for ii in range(20):
        # 20是变异次数
        v_line[np.random.randint(0,size)] = np.random.randint(0, 8)
    return v_line

def aneal(n_lines, o_lines, f_a, f_b, t):
    # 退火算法
    detas = (np.array(f_a)-np.array(f_b))
    for m, deta in enumerate(detas):
        if deta<0:
            p = np.minimum(1, np.exp(-np.abs(deta)/t))
            if np.random.rand() >= p:
                n_lines[m] = o_lines[m]
    n_f = get_eval(n_lines)
    b_line = n_lines[np.where(n_f == np.max(n_f))]
    return n_lines, b_line[0]

def cross_single_gene(pair):
    # 将一对基因给交叉
    line_width = len(pair[0]) # 进行随机配对的序列的长度
    rnd_point_num = int(line_width/5)
    inds_to_cross = []
    if rnd_point_num%2 == 0:
        rand_ind = np.sort(random.sample(range(line_width), rnd_point_num))
    else:
        rand_ind = np.sort(random.sample(range(line_width), rnd_point_num-1))
    for i in range(int(len(rand_ind)/2)):
        if rand_ind[2*i+1]-rand_ind[2*i] <= 6:
            # 说明这一个区域可交换
            tmp = pair[0][rand_ind[2*i]:rand_ind[2*i+1]+1]
            pair[0][rand_ind[2 * i]:rand_ind[2 * i + 1] + 1] = pair[1][rand_ind[2*i]:rand_ind[2*i+1]+1]
            pair[1][rand_ind[2 * i]:rand_ind[2 * i + 1] + 1] = tmp
    return pair

if __name__ == '__main__':
    # 【参数列表】-- begin

    # for 1#，3#，5#，7#
    cnc_reload_time_1 = 28
    # for 2#，4#，6#，8#
    cnc_reload_time_2 = 31
    work_time_val = 560
    move_time = [0, 20, 33, 46]
    wash_time = 25
    line_width = 213
    batch_size = 150
    cycles = 80 # 循环次数
    temp = 500 # 初始温度
    rate = 0.95
    var_rate = 0.5
    # 【参数列表】-- end

    begin_time = time.time()

    # fit func : single_process_time()
    # 随机生成初始序列
    test_lines = []

    rs = int(line_width/5)
    for p in range(batch_size):
        test_line = []
        for w in range(rs):
            gen = random.sample(range(8), 5)    # 每次产生5个，产生380/5次
            test_line.extend(gen)
        test_line.extend(np.random.randint(0, 8, line_width%5).tolist())
        test_lines.append(test_line)
    test_lines = np.array(test_lines)
    for v in range(len(test_lines)):
        test_lines[v,:8] = np.arange(8)

    best_lines = np.zeros((cycles, line_width))
    np.seterr(invalid='ignore')
    for z in range(cycles):
        lines_eval = get_eval(test_lines)
        rand_cnt = random_gene(lines_eval) # 得到序列数
        test_lines = new_rand_lines(rand_cnt, test_lines)
        rand_pairs = random_pairing(test_lines)
        new_lines_, old_lines_, f_after, f_before = cross_group_genes(rand_pairs)
        test_lines, best_lines[z] =  aneal(new_lines_, old_lines_, f_after,f_before, temp)
        temp = temp*rate
    end_time = time.time()
    ans = []
    for best_line in best_lines:
        ans.append(single_process_time(best_line))
    ans = np.array(ans)
    print('最好结果')
    print(min(ans)/3600)
    print(best_lines[np.where(ans==min(ans))[0][0]])

    print('【主程序运行耗时：', (end_time - begin_time), '】')

    plt.plot(np.arange(cycles) + 1, ans)
    plt.xlabel('迭代次数')
    plt.ylabel('210件工件加工时间')
    plt.show()
