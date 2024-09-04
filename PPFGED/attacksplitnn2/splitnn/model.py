import torch
import cv2
# import torch
import PIL
from PIL import Image
import os
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
# def eigValPct(eigVals, percentage):
#     sortArray=np.sort(eigVals)[::-1] # 特征值从大到小排序
#     pct = np.sum(sortArray)*percentage
#     tmp = 0
#     num = 0
#     for eigVal in sortArray:
#         tmp += eigVal
#         num += 1
#         if tmp>=pct:
#             return num
#
#
# def im_PCA(dataMat, percentage=0.9):
#     meanVals = np.mean(dataMat, axis=0)
#     meanRemoved = dataMat - meanVals
#     # 这里不管是对去中心化数据 or 原始数据计算协方差矩阵，结果都一样，特征值大小会变，但相对大小不会改变
#     # covMat = np.cov(dataMat, rowvar=False)
#     # 标准的计算需要除以(dataMat.shape[0]-1)，不算也不会影响结果，理由同上
#     covMat = np.dot(np.transpose(meanRemoved), meanRemoved)
#
#     eigVals, eigVects = np.linalg.eig(np.mat(covMat))
#     k = eigValPct(eigVals, percentage)  # 要达到方差的百分比percentage，需要前k个向量
#     # print('K =', k)
#
#     eigValInd = np.argsort(eigVals)[::-1]  # 对特征值eigVals从大到小排序
#     eigValInd = eigValInd[:k]
#     redEigVects = eigVects[:, eigValInd]  # 主成分
#     lowDDataMat = meanRemoved * redEigVects  # 将原始数据投影到主成分上得到新的低维数据lowDDataMat
#     reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 得到重构数据reconMat
#     return lowDDataMat, reconMat

#LDP
def k_random_response(value, values,epsilon):
    """
    the k-random response
    :param value: current value
    :param values: the possible value
    :param epsilon: privacy budget
    :return:
    """
    if not isinstance(values, list):
        raise Exception("The values should be list")
    if value not in values:
        raise Exception("Errors in k-random response")
    p = np.e ** epsilon / (np.e ** epsilon + len(values) - 1)
    if np.random.random() <= p:
        return value
    values.remove(value)
    return values[np.random.randint(low=0, high=len(values))]

# 数据中心化
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal


# 协方差矩阵
def Cov(dataMat):
    meanVal = np.mean(data, 0)  # 压缩行，返回1*cols矩阵，对各列求均值
    meanVal = np.tile(meanVal, (rows, 1))  # 返回rows行的均值矩阵
    Z = dataMat - meanVal
    Zcov = (1 / (rows - 1)) * Z.T * Z
    return Zcov


# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    k = Percentage2n(D, p)  # 确定k值
    # print("保留99%信息，降维后的特征个数：" + str(k) + "\n")#20230608-4/4
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector


# 得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat


# PCA算法
def PCA(data, p):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    # covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    # 得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)
    # 得到降维后的数据
    lowDataMat = getlowDataMat(dataMat, V)
  #  print(lowDataMat)
  #   print('p1 type:',type(lowDataMat))
  #   print('p1 shape:',lowDataMat.shape)
  #   print(lowDataMat.shape[0])
  #   print(lowDataMat.shape[1])
    if lowDataMat.shape[1]==1:
        p = lowDataMat
    else:
        i = 0
        j = 0
        q = []
        p = []

        while i < lowDataMat.shape[0]:
            while j < lowDataMat.shape[1]:
                c = lowDataMat[i,j]
                b = lowDataMat[i].tolist()[0]
                # print('测试开始')
                # print(c)
                # print(b)

                e = k_random_response(c, b, 1)
                q.append(e)
                # print(q)
                # print('测试结束')
                j = j + 1
            p.append(q)
            i = i + 1
        # print(i)
            j = 0
            q = []
        p = pd.DataFrame(p)
        p = np.array(p)
        p = np.mat(p)


    # 重构数据
    # reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    # print('p2 type:',type(p))
    # print('p2 shape:',p.shape)
    reconDataMat = Reconstruction(p, V, meanVal)
    return lowDataMat,p,reconDataMat




def PrintError(data, recdata):
    sum1 = 0
    sum2 = 0
    D_value = data - recdata # 计算两幅图像之间的差值矩阵
    # 计算两幅图像之间的误差率，即信息丢失率
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i],data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    # print('丢失信息量：', sum2)
    # print('原始信息量：', sum1)
    # print('信息丢失率：', sum2/sum1)20230608-3


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Client(torch.nn.Module):
    def __init__(self, client_model):
        super().__init__()
        """class that expresses the Client on SplitNN

        Args:
            client_model (torch model): client-side model

        Attributes:
            client_model (torch model): cliet-side model
            client_side_intermidiate (torch.Tensor): output of
                                                     client-side model
            grad_from_server
        """

        self.client_model = client_model
        self.client_side_intermidiate = None
        self.grad_from_server = None

    def forward(self, inputs):
        """client-side feed forward network

        Args:
            inputs (torch.Tensor): the input data

        Returns:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model which the client sent
                                                   to the server
        """

        self.client_side_intermidiate = self.client_model(inputs)
        # send intermidiate tensor to the server
        intermidiate_to_server = self.client_side_intermidiate.detach()\
            .requires_grad_()

        return intermidiate_to_server

    def client_backward(self, grad_from_server):
        """client-side back propagation

        Args:
            grad_from_server: gradient which the server send to the client
        """
        self.grad_from_server = grad_from_server
        self.client_side_intermidiate.backward(grad_from_server)

    def train(self):
        self.client_model.train()

    def eval(self):
        self.client_model.eval()


class Server(torch.nn.Module):
    def __init__(self, server_model):
        super().__init__()
        """class that expresses the Server on SplitNN

        Args:
            server_model (torch model): server-side model

        Attributes:
            server_model (torch model): server-side model
            intermidiate_to_server:
            grad_to_client
        """
        self.server_model = server_model

        self.intermidiate_to_server = None
        self.intermidiate_to_server1 = None
        self.grad_to_client = None

    def forward(self, intermidiate_to_server):
        """server-side training

        Args:
            intermidiate_to_server (torch.Tensor): the output of client-side
                                                   model

        Returns:
            outputs (torch.Tensor): outputs of server-side model
        """
        self.intermidiate_to_server1 = intermidiate_to_server
        outputs = self.server_model(intermidiate_to_server)

        return outputs

    def server_backward(self):
        # self.intermidiate_to_server1=self.intermidiate_to_server1.detach().requires_grad_()
        # print('原始参数尺寸1：',self.intermidiate_to_server1.shape)#torch.Size([64, 64, 16, 16])
        # print('原始参数类型1：', type(self.intermidiate_to_server1))
        # print('原始参数类型1：', self.intermidiate_to_server1.dtype)
        # print('原始参数尺寸2：',self.intermidiate_to_server1.grad.shape)#torch.Size([64, 64, 16, 16])
        # print('原始参数类型2：', type(self.intermidiate_to_server1.grad))
        # print('原始参数类型2：', self.intermidiate_to_server1.grad.dtype)
        # print('原始参数尺寸3：',self.intermidiate_to_server1.grad.clone().shape)#torch.Size([64, 64, 16, 16])
        # print('原始参数类型3：', type(self.intermidiate_to_server1.grad.clone()))
        # print('原始参数类型3：', self.intermidiate_to_server1.grad.clone().dtype)
        self.grad_to_client = self.intermidiate_to_server1.grad.clone()
        return self.grad_to_client

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()


class SplitNN(torch.nn.Module):
    def __init__(self, client, server,
                 client_optimizer, server_optimizer,
                 ):
        super().__init__()
        """class that expresses the whole architecture of SplitNN

        Args:
            client (attack_splitnn.splitnn.Client):
            server (attack_splitnn.splitnn.Server):
            clietn_optimizer
            server_optimizer

        Attributes:
            client (attack_splitnn.splitnn.Client):
            server (attack_splitnn.splitnn.Server):
            clietn_optimizer
            server_optimizer
        """
        self.client = client
        self.server = server
        self.client_optimizer = client_optimizer
        self.server_optimizer = server_optimizer

        self.intermidiate_to_server = None
        self.intermidiate_to_server1 = None

    def forward(self, inputs):
        # execute client - feed forward network
        self.intermidiate_to_server = self.client(inputs)
        # for i in range(self.intermidiate_to_server.shape[0]):
        #     b = self.intermidiate_to_server[i].view(-1, 128, 128)
        #     print("img_tensor:", b)
        #     #     print("img_tensor:",b[0])
        #     bb = Image.fromarray((b[0] * 255).numpy().astype(np.uint8))
        #     print(0)
        #     plt.imshow(bb)
        #     print(1)

        # execute server - feed forward netwoek
        # print('原始参数尺寸：',self.intermidiate_to_server.shape)
        # print('原始参数类型：', type(self.intermidiate_to_server))
        # print('原始参数类型：', self.intermidiate_to_server.dtype)
        data = 0
        for epoch in range(64):

            # a = self.intermidiate_to_server  # 随机2行3列数据，01之间 torch.Size([64, 64, 16, 16])   #torch.Size([64, 64, 16, 16])
            # print('self.intermidiate_to_server.shape[0]',self.intermidiate_to_server.shape[0])#64
            for i in range(self.intermidiate_to_server.shape[0]):
                # print('self.intermidiate_to_server[i].shape',self.intermidiate_to_server[i].shape)#torch.Size([64, 16, 16])
                # print('self.intermidiate_to_server[i].shape', self.intermidiate_to_server[i].shape)#self.intermidiate_to_server[i].shape torch.Size([64, 4, 4])

                # b = self.intermidiate_to_server[i].view(-1, 128, 128)#20230627性能下降
                b = self.intermidiate_to_server[i].view(-1, 32, 32)
                # print('原始维度尺寸：', b.shape)  # torch.Size([1, 128, 128])
                # print("原始维度数值:", b)
                #     print("img_tensor:",b[0])
                bb = Image.fromarray((b[0] * 255).detach().numpy().astype(np.uint8))
                root = './test_data'  # 保存地址
                root1 = root + "/" + str(data)
                if not os.path.exists(root1):  # 如果文件夹不存在，则创建该文件夹
                    os.makedirs(root1)
                path = root1 + "/" + str(i) + ".jpg"  # 保存地址

                try:
                    bb.save(path, quality=95)
                    # print('图片保存成功，保存在' + root + "\n")
                except:
                    print('图片保存失败')

                path = os.path.abspath(path)

                # img = cv2.imread('./test_data/0.jpg')
                img = cv2.imread(path)
                blue = img[:, :, 0]
                # print('blue shape :',blue.shape)20230608-2
                # blue = img[:,:]
                # print(blue.shape)

                #1

                # pca = PCA(n_components=100).fit(blue)
                # # 降维
                # x_new = pca.transform(blue)
                # # 还原降维后的数据到原空间
                # recdata = pca.inverse_transform(x_new)
                # x = torch.Tensor(recdata)


                #2


                dataMat = np.mat(blue)
                # lowDDataMat, reconMat = im_PCA(dataMat, 0.9)
                # print('原始数据', blue.shape, '降维数据', lowDDataMat.shape, '重构数据', reconMat.shape)

                lowDDataMat,p, reconMat = PCA(dataMat, 0.99)
                # print('原始数据', blue.shape, '降维数据', lowDDataMat.shape,'降维扰动数据', p.shape, '重构数据', reconMat.shape)#20230608-1


                # print('dataMat shape:', dataMat.shape)
                # print('dataMat:', dataMat)
                # print('reconMat shape:', reconMat.shape)
                # print('reconMat:', reconMat)
                PrintError(np.array(blue, dtype='double'), np.array(reconMat, dtype='double'))
                x = torch.Tensor(reconMat)
                # print('x shape :', x.shape)torch.Size([128, 128])



                # x = torch.tensor(reconMat,dtype=torch.float64).to(device)
                # print(type(x))

                # print('x shape:', x.shape)
                # print('x:', x)
                reconMat1 = x.unsqueeze(0)
                # print('reconMat1 shape :', reconMat1.shape)torch.Size([1, 128, 128])
                # print('reconMat1 shape:', reconMat1.shape)
                # print('reconMat1:', reconMat1)

                # reconMat2 = reconMat1.view(64, 16, 16)#20230627性能下降
                reconMat2 = reconMat1.view(64, 4, 4)
                # print('reconMat2 shape:', reconMat2.shape)torch.Size([1, 64, 16, 16])
                # print('reconMat2:', reconMat2)

                reconMat3 = torch.unsqueeze(reconMat2, dim=0)
                # print('reconMat3 shape :', reconMat3.shape)
            if epoch == 0:
                self.intermidiate_to_server1 = reconMat3
            else:
                self.intermidiate_to_server1 = torch.cat((self.intermidiate_to_server1, reconMat3), 0)

            data = data + 1
            # print('self.intermidiate_to_server1 shape :', self.intermidiate_to_server1.shape)torch.Size([64, 16, 16])
        # print('PCA参数尺寸：',_inputs.shape)
        # print('PCA参数类型：',type(_inputs))
        # print('PCA参数类型：', _inputs.dtype)
        # self.intermidiate_to_server1=self.intermidiate_to_server

        self.intermidiate_to_server1.requires_grad = True
        outputs = self.server(self.intermidiate_to_server1)
        # outputs = self.server(self.intermidiate_to_server1)
        # print('输出参数：', outputs.shape)


        return outputs

    def backward(self):
        # execute server - back propagation
        grad_to_client = self.server.server_backward()
        # execute client - back propagation
        self.client.client_backward(grad_to_client)

    def zero_grads(self):
        self.client_optimizer.zero_grad()
        self.server_optimizer.zero_grad()

    def step(self):
        self.client_optimizer.step()
        self.server_optimizer.step()

    def train(self):
        self.client.train()
        self.server.train()

    def eval(self):
        self.client.eval()
        self.server.eval()
