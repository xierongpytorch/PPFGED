import torch
import cv2
# import torch
import PIL
from PIL import Image
import os
from ..attacker import AbstractAttacker
from sklearn.decomposition import PCA
import numpy as np
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
    print("保留99%信息，降维后的特征个数：" + str(k) + "\n")
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
    print('丢失信息量：', sum2)
    print('原始信息量：', sum1)
    print('信息丢失率：', sum2/sum1)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.random.manual_seed(42)
# # print(device)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Black_Box_Model_Inversion(AbstractAttacker):
    def __init__(self, splitnn, attacker_model, attacker_optimizer):
        """Class that implement black box model inversion

        Args:
            splitnn (SplitNN): target splitnn model
            attacker_model (torch model): model to attack target splitnn
            attacker_optimizer (torch optimizer): optimizer for attacker_model

        Attributes:
            splitnn (SplitNN):
            attacker_model (torch model):
            attacker_optimizer (torch optimizer):
        """
        super().__init__(splitnn)
        self.attacker_model = attacker_model
        self.attacker_optimizer = attacker_optimizer

    def fit(self, dataloader_for_attacker, epoch):

        for i in range(epoch):
            # for data, _ in dataloader_for_attacker:
            for data in dataloader_for_attacker:
                data=data['image']#CPU
                # data=data['image'].to(device)#GPU
                self.attacker_optimizer.zero_grad()

                target_outputs = self.splitnn.client(data)

                attack_outputs = self.attacker_model(target_outputs)

                loss = ((data - attack_outputs)**2).mean()

                loss.backward()
                self.attacker_optimizer.step()

            print(f"epoch {i}: reconstruction_loss {loss.item()}")

    def attack(self, dataloader_target):
        attack_results = []

        # for data, _ in dataloader_target:
        for data in dataloader_target:
            data = data['image']
            # data = data['image'].to(device)
            target_outputs = self.splitnn.client(data)

            d = 0
            for epoch in range(15):

                # a = self.intermidiate_to_server  # 随机2行3列数据，01之间 torch.Size([64, 64, 16, 16])   #torch.Size([64, 64, 16, 16])
                for i in range(target_outputs.shape[0]):
                    b = target_outputs[i].view(-1, 128, 128)
                    # print('原始维度尺寸：', b.shape)  # torch.Size([1, 128, 128])
                    # print("原始维度数值:", b)
                    #     print("img_tensor:",b[0])
                    bb = Image.fromarray((b[0] * 255).detach().numpy().astype(np.uint8))
                    root = './test_data1'  # 保存地址
                    root1 = root + "/" + str(d)
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
                    # blue = img[:,:]
                    # print(blue.shape)

                    # 1

                    # pca = PCA(n_components=64).fit(blue)
                    # # 降维
                    # x_new = pca.transform(blue)
                    # # 还原降维后的数据到原空间
                    # recdata = pca.inverse_transform(x_new)
                    # x = torch.Tensor(recdata)

                    # 2

                    dataMat = np.mat(blue)
                    # lowDDataMat, reconMat = im_PCA(dataMat, 0.9)
                    # print('原始数据', blue.shape, '降维数据', lowDDataMat.shape)
                    lowDDataMat, p, reconMat = PCA(dataMat, 0.9)
                    print('原始数据', blue.shape, '降维数据', lowDDataMat.shape, '降维扰动数据', p.shape, '重构数据', reconMat.shape)

                    # print('dataMat shape:', dataMat.shape)
                    # print('dataMat:', dataMat)
                    # print('reconMat shape:', reconMat.shape)
                    # print('reconMat:', reconMat)
                    PrintError(np.array(blue, dtype='double'), np.array(reconMat, dtype='double'))
                    x = torch.Tensor(reconMat)

                    # x = torch.tensor(reconMat,dtype=torch.float64).to(device)
                    # print(type(x))

                    # print('x shape:', x.shape)
                    # print('x:', x)
                    reconMat1 = x.unsqueeze(0)
                    # print('reconMat1 shape:', reconMat1.shape)
                    # print('reconMat1:', reconMat1)

                    reconMat2 = reconMat1.view(64, 16, 16)
                    # print('reconMat2 shape:', reconMat2.shape)
                    # print('reconMat2:', reconMat2)

                    reconMat3 = torch.unsqueeze(reconMat2, dim=0)
                if epoch == 0:
                    target_outputs = reconMat3
                else:
                    target_outputs = torch.cat((target_outputs, reconMat3), 0)

                d = d + 1
            # print('PCA参数尺寸：',_inputs.shape)
            # print('PCA参数类型：',type(_inputs))
            # print('PCA参数类型：', _inputs.dtype)
            # self.intermidiate_to_server1=self.intermidiate_to_server

            target_outputs.requires_grad = True

            # print(target_outputs.shape)
            recreated_data = self.attacker_model(target_outputs)
            attack_results.append(recreated_data)

        return torch.cat(attack_results)
