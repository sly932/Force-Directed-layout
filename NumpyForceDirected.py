import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random
import time
import argparse
import os
import sys
BLOCK = '█'
BLOCKS_NUM = 50
# 定义力导向算法类
class ForceDirectedAlgorithm:
    def __init__(self,args):
        self.args = args
        if args.load or args.load_node:
            nodefile = os.path.join("saved_data","nodes_epoch_latest.txt")
            nodes = readNodes(nodefile)
        elif args.withmass:
            nodes = generateMassNodes(args.node_num,args.minpos,args.maxpos,args.minmass,args.maxmass)
        elif not args.withmass:
            nodes = generateNodes(args.node_num,args.minpos,args.maxpos)
            
        if args.load or args.load_edge:
            edgefile = os.path.join("saved_data","edges_epoch_latest.txt")
            edges = readEdges(edgefile)
        else:
            edges = generateEdges(args.node_num,args.max_edge_num,args.total_edge_num)  
        print(f"node nums:{len(nodes)}")      
        if args.load or args.load_delta:
            self.delta_sum = self.loadDelta_sum()
        else:
            self.delta_sum = []
        self.rundir = mkdir(args.dir)
        self.nodes = np.array(nodes)  # 节点坐标
        self.edges = np.array(edges)  # 边连接
        
    def loadDelta_sum(self):
        sum_file = os.path.join('saved_data','delta_sum.txt')
        delta_sum = []
        with open(sum_file,'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                delta_sum.append(float(line))
        return delta_sum 
    """def loadConfig(self,file=os.path.join('saved_data','config.conf')):
        dic = {}
        with open(file,'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                dic[line.split()[0]] = line.split()[1]
        self.k_r = float(dic["k_r"])  # 万有引力常数
        self.k_s = float(dic["k_s"])
        self.max_len = int(dic["max_len"])
        self.eval_step = int(dic["eval_step"])
        self.steps = int(dic["steps"])  # 迭代次数
        self.dis = float(dic["dis"])
        self.step = int(dic["step"])
        sum_file = os.path.join('saved_data','delta_sum.txt')
        self.delta_sum = []
        with open(sum_file,'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.delta_sum.append(float(line))"""
        
    #打印进度条
    def printprogress(self,cur,total,total_sec,epoch_sec,block=BLOCK,progress_len = BLOCKS_NUM):
        hour = int(total_sec // 3600)
        minute = int(total_sec // 60 % 60)
        second = int(total_sec % 60)
        progress = float(cur+1) / total * 100
        block_num = int(progress * progress_len / 100)
        bar = block * block_num + ' ' * (progress_len - block_num) + '|'
        print(f"\rProgressing:{cur+1}|{total}|{bar}{progress:.1f}% {hour:02d}:{minute:02d}:{second:02d} {epoch_sec:.2f}sec/epoch",end=' ')
    
    #存储网络图
    def printNodes(self,args,step,ifsvg=False):
        #s = str(step)
        #step = (4-len(s))*'0' + s
        file = os.path.join(self.rundir,"iter",f"epoch_{step:06d}")
        if args.light:
            plt.figure()
            nodes = self.nodes
            edges = self.edges
            for edge in edges:
                edge = edge - 1
                if edge[0] > edge[1]:
                    continue
                x = [nodes[edge[0]][0],nodes[edge[1]][0]]
                y = [nodes[edge[0]][1],nodes[edge[1]][1]]
                plt.plot(x,y,zorder=1)
            plt.scatter(nodes[:,0], nodes[:,1],color='royalblue', edgecolor='black',linewidth=args.scatter_linewidth,zorder=2)
        else:
            plt.figure(figsize=(args.width,args.height))
            nodes = self.nodes
            edges = self.edges
            for edge in edges:
                edge = edge - 1
                if edge[0] > edge[1]:
                    continue
                x = [nodes[edge[0]][0],nodes[edge[1]][0]]
                y = [nodes[edge[0]][1],nodes[edge[1]][1]]
                plt.plot(x,y,linewidth=args.linewidth,zorder=1)
            
            sizes = [args.scatter_size for _ in range(len(nodes))]
            mass = args.maxmass - args.minmass
            if args.withmass:
                sizes = [args.scatter_size*(n[2]-args.minmass)/mass for n in nodes]
            for i,node in enumerate(nodes):
                plt.scatter(node[0], node[1],color='royalblue', edgecolor='black',linewidth=args.scatter_linewidth,s=sizes[i],zorder=2)
            plt.tick_params(labelsize=args.labelsize)
        
        svg = file + '.svg'
        jpg = file + '.jpg'
        if ifsvg:
            plt.savefig(svg,format='svg')
        else:
            plt.savefig(jpg)
        plt.close()
    
    #存储网络每个训练轮次的变化量
    def printDeltaSum(self,step,sum):
        file = os.path.join(self.rundir,'saved_data','delta_sum.txt')
        with open(file,'w') as f:
            for s in sum:
                f.write(f"{s}\n")
        path = os.path.join(self.rundir,'delta_sum',f"DeltaSum.svg")
        plt.figure()
        plt.plot(sum)
        plt.savefig(path,format='svg')
        plt.close()
       
    #保存edges与nodes
    def savedata(self,step,name):
        nodefile = os.path.join(self.rundir,'saved_data',f"nodes_epoch_{name}.txt")
        edgefile = os.path.join(self.rundir,'saved_data',f"edges_epoch_{name}.txt")
        with open(nodefile,'w') as f:
            for node in self.nodes:
                f.write(f"{node[0]} {node[1]}\n")
        with open(edgefile,'w') as f:
            for node in self.edges:
                f.write(f"{node[0]} {node[1]}\n")
        config = os.path.join(self.rundir,'saved_data',"config.conf")
        args = vars(self.args)
        with open(os.path.join(self.rundir,'saved_data','config.conf'),'w') as f:
            keys = args.keys()
            for key in keys:   
                f.write(f"{key} {args[key]}\n")
            
    def execute(self):
        args = self.args 
        step = args.cur_step
        delta_sum = self.delta_sum
        
        plt.figure()
        start_time = time.time()
        last_time = start_time
        
        self.printNodes(args,0) 
        while True:
            delta_step = 0
            if step >= args.steps:
                break
            """plt.cla()
            plt.scatter(self.nodes[:,0],self.nodes[:,1])
            for edge in self.edges:
                edge = edge - 1
                if edge[0] > edge[1]:
                    continue
                x = [self.nodes[edge[0]][0],self.nodes[edge[1]][0]]
                y = [self.nodes[edge[0]][1],self.nodes[edge[1]][1]]
                plt.plot(x,y)
            plt.show(block=False)
            plt.pause(1)
            plt.close()"""  
            
            if not args.notdraw:
                self.printNodes(args,step)     
            if not args.notsave:
                self.savedata(step=step,name="latest") 
            if step%args.eval_step == 0 and not args.notdraw: 
                self.printNodes(args,step,ifsvg=True)
                self.printDeltaSum(step,delta_sum) 
            
            cur_time = time.time()
            self.printprogress(step,args.steps,cur_time-start_time,epoch_sec=cur_time-last_time)
            last_time = cur_time
            step += 1
            
            # 计算斥力
            repulsion = np.zeros_like(self.nodes)
            dn = self.nodes[:,np.newaxis,:2] - self.nodes[np.newaxis,:,:2]
            dist = np.linalg.norm(dn,axis=2)
            dist[dist==0] = 1e-6
            if args.withmass:
                m = self.nodes[:,np.newaxis,2] * self.nodes[np.newaxis,:,2]
                repulsion = args.k_r * dn *m[:,:,np.newaxis] / dist[:,:,np.newaxis]**3
            else:
                repulsion = args.k_r * dn / dist[:,:,np.newaxis]**3
            repulsion = np.sum(repulsion,axis=1)
            
            # 计算弹簧作用力
            spring = np.zeros_like(self.nodes[:,:2])
            for edge in self.edges:
                if edge[1]-edge[0] < 0:
                    continue
                edge = edge - 1
                if args.withmass:
                    dx, dy, _ = self.nodes[edge[1]] - self.nodes[edge[0]]
                else :
                    dx, dy = self.nodes[edge[1]] - self.nodes[edge[0]]
                distance = math.sqrt(dx ** 2 + dy ** 2)
                if distance == 0:
                    continue
                force = args.k_s * (distance - args.ideal_dis)
                force_x, force_y = force * dx / distance, force * dy / distance
                spring[edge[0]][0] += force_x
                spring[edge[0]][1] += force_y
                spring[edge[1]][0] -= force_x
                spring[edge[1]][1] -= force_y
              
            # 更新节点坐标
            delta = repulsion + spring
            cur_max_len = args.max_move / (step + 0.1)
            dist = np.sum(delta**2,axis=1)
            idx = dist > cur_max_len
            rate = np.ones(len(delta))
            rate[idx] = np.sqrt(cur_max_len / dist[idx])
            delta[idx] = delta[idx] * rate[idx,np.newaxis]
            self.nodes[:,:2] += delta
            if not args.notsave:
                delta_sum.append(np.sum(delta))
        self.printNodes(args,1) 
        return self.nodes

def generateNodes(node_num,minval,maxval):
    #generate nodes
    print("generating nodes without mass...",end='')
    nodes = []
    for i in range(node_num):
        inter = maxval - minval
        x, y = random.random()*inter+minval,random.random()*inter+minval
        nodes.append((x,y))
    print("done")
    return nodes

def generateMassNodes(node_num,minval,maxval,minmass,maxmass):
    #generate nodes
    print("generating nodes with mass...",end='')
    nodes = []
    for i in range(node_num):
        inter = maxval - minval
        x, y = random.random()*inter+minval,random.random()*inter+minval
        m = random.random()*(maxmass-minmass)+minmass
        nodes.append((x,y,m))
    print("done")
    return nodes

def generateEdges(node_num,max_edge_num,total_edge_num):
    total_edge_num = min(total_edge_num,node_num*max_edge_num)
    edges = []
    nums = [random.randint(0,total_edge_num) for _ in range(node_num-1)]
    nums.append(total_edge_num)
    nums.sort()
    nums = [nums[0] if i==0 else nums[i]-nums[i-1] for i in range(len(nums))]
    for x in range(node_num):
        edge_num = nums[x] if nums[x]>4 else 4
        for i in range(edge_num):
            while True:
                y = random.randint(0,node_num)
                if y > x:
                    break
            edges.append((x,y))
    return edges

def readNodes(file):
    print("reading nodes...",end='')
    nodes = []
    with open(file,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            a,b = float(line.split()[0]),float(line.split()[1])
            nodes.append((a,b))
    print("done")
    return nodes

def readEdges(file):
    #reading edges
    print("reading edges...",end='')
    edges = []
    with open(file,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            a,b = int(line.split()[0]),int(line.split()[1])
            edges.append((min((a,b)),max(a,b)))
    print("done")
    return edges

def renameImg(path=os.path.join('imgs','iter')):
    files = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.svg')]
    for file in files:
        name = os.path.basename(file)
        epoch = int(name.split('.')[0].split('_')[1])
        newname = f"{name.split('_')[0]}_{epoch:06d}.{name.split('.')[1]}"
        newfile = os.path.join(path,newname)
        os.rename(file,newfile)

def mkdir(dir):
  if dir == None:
    i = 1
    while True:
        rundir = os.path.join('imgs',str(i))
        i += 1
        if not os.path.exists(rundir):
          break
  else:
    rundir = os.path.join('imgs',dir)
    if os.path.exists(rundir):
      i = 1
      while True:
        rundir = os.path.join('imgs',dir+'(' + str(i) + ')')
        i += 1
        if not os.path.exists(rundir):
          break
  os.makedirs(rundir)
  os.makedirs(os.path.join(rundir,'iter'))
  os.makedirs(os.path.join(rundir,'delta_sum'))
  os.makedirs(os.path.join(rundir,'saved_data'))
  return rundir
# 测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force Directed")
    parser.add_argument("--dir",type=str,default=None, help="running dir")
    parser.add_argument("--notdraw",action="store_true", help="do not draw")
    parser.add_argument("--notsave",action="store_true", help="do not save data")
    parser.add_argument("--light",action="store_true", help="draw figure with default settings")
    parser.add_argument("--load",action="store_true",help="load saved data")
    parser.add_argument("--load_delta",action="store_true",help="load saved delta_sum")
    parser.add_argument("--load_node",action="store_true",help="load saved data")
    parser.add_argument("--load_edge",action="store_true",help="load saved data")
    parser.add_argument("--cur_step",type=int,default=0,help="beginning step") 
    parser.add_argument("--steps",type=int,default=3000,help="total steps")
    parser.add_argument("--eval_step",type=int,default=100,help="step interval of evaluation")
    parser.add_argument("--k_s",type=float,default=0.7,help="coefficient of spring")
    parser.add_argument("--k_r",type=float,default=20,help="coefficient of repulsion")
    parser.add_argument("--max_move",type=float,default=60,help="maximum movement distance")
    parser.add_argument("--ideal_dis",type=float,default=5,help="ideal distance of two nodes")
    parser.add_argument("--withmass",action="store_true",help="flag to generate node with mass")
    parser.add_argument("--minmass",type=float,default=2,help="minimum mass")
    parser.add_argument("--maxmass",type=float,default=10,help="maximum mass")
    parser.add_argument("--node_num",type=int,default=200,help="number of nodes")
    parser.add_argument("--minpos",type=float,default=0,help="minimum position of nodes")
    parser.add_argument("--maxpos",type=float,default=1000,help="maximum position of nodes")
    parser.add_argument("--total_edge_num",type=int,default=500,help="total number of edges")
    parser.add_argument("--max_edge_num",type=int,default=100,help="maximum number of edges per node")
    
    parser.add_argument("--width",type=int,default=8,help="figure width")
    parser.add_argument("--height",type=int,default=5,help="figure height")
    parser.add_argument("--scatter_size",type=int,default=300,help="the size of node")
    parser.add_argument("--scatter_linewidth",type=int,default=1,help="the width of contour of node")
    parser.add_argument("--linewidth",type=int,default=2,help="the width of edge")
    parser.add_argument("--labelsize",type=int,default=16,help="the size of figure label")
    args = parser.parse_args()
    print(args)      
    if args.withmass and args.minmass <= 0:
        print(f"illegal minimum mass {args.minmass}")
        sys.exit()
    
    algorithm = ForceDirectedAlgorithm(args)
    positions = algorithm.execute()
  