#!/usr/bin/env python3
# coding: utf-8
# File: hmm_cut.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-26

class HmmCut:
    def __init__(self):
        base_dir = r'D:\codes\tf_models\fork_github\WordSegment'
        trans_path = base_dir + r'\model\prob_trans.model'
        emit_path = base_dir + r'\model\prob_emit.model'
        start_path = base_dir + r'\model\prob_start.model'
        print(trans_path)
        self.prob_trans = self.load_model(trans_path)
        self.prob_emit = self.load_model(emit_path)
        self.prob_start = self.load_model(start_path)

    '''加载模型'''
    def load_model(self, model_path):
        f = open(model_path, 'r')
        a = f.read()
        word_dict = eval(a)
        f.close()
        return word_dict

    '''verterbi算法求解'''
    def viterbi(self, obs, states, start_p, trans_p, emit_p):  # 维特比算法（一种递归算法）
        #
        # 算法的局限在于训练语料要足够大，需要给每个词一个发射概率,.get(obs[0], 0)的用法是如果dict中不存在这个key,则返回0值
        V = [{}] # 用于记录每个观测值是某个tag的概率
        path = {}
        #起始概率
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(obs[0], 0)  # 在位置0，以y状态为末尾的状态序列的最大概率 初始tag概率*tag-word概率
            path[y] = [y]

        #后面观测序列路径
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}
            for y in states: #循环每个tag
                # 上一次到tag y0路径概率 * tag y0- tag y的概率 *   tag y-当前obs字符概率，只选上一次到y0路径概率大于0的tag
                # hmm的精髓，计算一个词是某个tag的概率公式：路径上节点概率相乘，节点概率是转移概率和发射概率相乘，相对于crf是相加
                state_path = ([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emit_p[y].get(obs[t], 0), y0) for y0 in states if V[t - 1][y0] > 0])
                if state_path == []: # 如果上一次路径概率<=0 说明没有路径了
                    (prob, state) = (0.0, 'S')
                else:
                    (prob, state) = max(state_path) # 选择一个概率最大的路径
                V[t][y] = prob
                newpath[y] = path[state] + [y] # 记录这条路径

            path = newpath  # 记录状态序列
        (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])  # 在最后一个位置，以y状态为末尾的状态序列的最大概率
        return (prob, path[state])  # 返回概率和状态序列

    # 分词主控函数
    def cut(self, sent):
        prob, pos_list = self.viterbi(sent, ('B', 'M', 'E', 'S'), self.prob_start, self.prob_trans, self.prob_emit)
        seglist = list()
        word = list()
        for index in range(len(pos_list)):
            if pos_list[index] == 'S':
                word.append(sent[index])
                seglist.append(word)
                word = []
            elif pos_list[index] in ['B', 'M']:
                word.append(sent[index])
            elif pos_list[index] == 'E':
                word.append(sent[index])
                seglist.append(word)
                word = []
        seglist = [''.join(tmp) for tmp in seglist]

        return seglist

    #测试
def test():
    sent = []
    sent.append('维特比算法viterbi的简单实现-python版')
    sent.append('''目前在自然语言处理技术中，中文处理技术比西文处理技术要落后很大一段距离，许多西文的处理方法中文不能直接采用，就是因为中文必需有分词这道工序。中文分词是其他中文信息处理的基础，搜索引擎只是中文分词的一个应用。''')
    sent.append('北京大学学生前来应聘')
    sent.append('新华网驻东京记者报道')
    sent.append('我们在野生动物园玩')
    sent.append('我们在野生動物園玩')
    cuter = HmmCut()
    seglist = cuter.cut(sent[5])
    print(seglist)

if __name__ == "__main__":
    test()