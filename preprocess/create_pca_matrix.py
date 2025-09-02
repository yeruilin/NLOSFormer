import os
from scipy.io import loadmat,savemat
import numpy as np

def find_files(directory,ext):
    obj_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                obj_files.append(os.path.join(root, file))
    return obj_files
                    
class PCAMatrix:
    def __init__(self, kernel_dir,init_ker,l=128, crop_l=79, b=42):
        self.kernel_dir =kernel_dir
        self.l=l # 原始图片大小
        self.crop_l=crop_l # 裁剪到这么大的图片作为核
        self.b=b # 降维之后的维度

        self.start=int(l/2-(crop_l-1)/2)
        
        if type(kernel_dir)==type([]):
            self.files = []
            for dir in kernel_dir:
                temp=find_files(dir,'mat')
                self.files=self.files+temp
        else:
            self.files = find_files(kernel_dir,'mat')

        self.init_ker=init_ker

        print(len(self.files))

    def generateMatrix(self):
        num=len(self.files)
        mat=np.zeros((num,self.crop_l*self.crop_l))

        init_kernel=self._readkernel(self.init_ker).flatten(order="F")

        for i in range(num):
            kernel=self._readkernel(self.files[i])
            mat[i,:]=kernel.flatten(order="F") # 不知为何这里需要用列优先展开结果才对

        # PCA:左奇异矩阵可以用于行数的压缩，右奇异矩阵可以用于列数即特征维度的压缩
        mean = np.mean(mat, 0)
        normal_mat = mat - mean
        U, S, Vt = np.linalg.svd(mat.T,False) 
        # S按照顺序排列的一维向量

        # print(U.shape,S.shape,Vt.shape) # (79*79, 200) (200,) (200, 200)

        print(np.sum(S[:self.b])/np.sum(S)) # 判断是否找到足够的主成分

        matrix=U[:,:self.b]

        reduced_init_kernel=np.dot(init_kernel-mean,matrix) # 降维，右乘V
        restored_sample=np.dot(reduced_init_kernel,matrix.T)+mean # 重新恢复，右乘V^T，并加上mean
        return matrix,mean,reduced_init_kernel,restored_sample # [N,b]的矩阵，将未降维的数据[M,N]乘以它，就可以压缩得到b的压缩维度
    
    def _readkernel(self, file):
        try:
            # 读取实测图片
            kernel=loadmat(file)['psf']
            kernel=np.squeeze(kernel)
            kernel=kernel[self.start:self.start+self.crop_l,self.start:self.start+self.crop_l]
            kernel=kernel/np.sum(kernel)
        
            return kernel
        
        except Exception as e:
            print(file)
            print(e)
            return None

if __name__=="__main__":
    datadir=["/data/NLOSFormer/TrainingData/ThermalNLOSData/augment/15/"]#"/home/yrl/Infrared/alpha_data_tilt/calib/"
    init_ker="/data/NLOSFormer/TrainingData/ThermalNLOSData/augment/15/18.mat" # angle0_alpha5
    dim=2
    p=PCAMatrix(datadir,init_ker,128,79,dim) # l=79,b=42,核大小和降维后大小
    pca_matrix,mean,reduced_kernel,restored_sample=p.generateMatrix() # [l*l,b]
    # (f"pca_matrix{dim}.mat",{"matrix":pca_matrix,"mean":mean})
    # savemat("pca_matrix.mat",{"matrix":pca_matrix,"mean":mean})
    # savemat("initial_kernel.mat",{"reduced_kernel":reduced_kernel,"restored_sample":restored_sample})


