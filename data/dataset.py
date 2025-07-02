import pickle
import torch.utils.data as data
from typing import Any, Callable, Optional, Tuple
import os
import numpy as np
__all__ = ('TennesseeEastman','TE_challenge',
           )

class TennesseeEastman(data.Dataset):
    """`TEDataset. 21 faults with one normal condition (22 classes), 2500/500 train/500 samples per class 

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform has not been applied to the 2D array.
        transform (callable, optional): A function/transform 
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    Usage:
    import dataset
    trainset = dataset.TennesseeEastman('data',size='standard',train=True,normalized=args['normalized'])
    testset = dataset.TennesseeEastman('data',size='standard',train=False,normalized=args['normalized'])
    # pytorch loader
    train_loader = DataLoader(trainset,batch_size='standard', shuffle=True)
    test_loader = DataLoader(testset, batch_size='standard', shuffle=False)
    # numpy array
    x_train, y_train = trainset.data, trainset.targets
    x_test, y_test = testset.data, testset.targets  
    """

    def __init__(
            self,
            root: str,
            size: str,
            train: bool = True,
            normalized = False,
            # transform: Optional[Callable] = None,
            # target_transform: Optional[Callable] = None,
    ) -> None:
        if size == 'large':
            self.root = os.path.join(root, 'TennesseeEastman_large')
        if size == 'standard':
            self.root = os.path.join(root, 'TennesseeEastman_standard')
        if size == 'standard_r':
            self.root = os.path.join(root, 'TennesseeEastman_standard_r')
        # self.transform = transform
        # self.target_transform = target_transform
        self.train = train  # training set or test set
        
        if self.train:
            file_name = 'TE_train'
        else:
            file_name = 'TE_test'

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        file_path = os.path.join(self.root, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f)
            self.data = entry['data']
            self.targets = entry['targets']

        if normalized:
            mean_ = np.array([0.25842208, 0.601262 , 0.565033  , 0.46127933, 0.49006358,
                    0.47829297, 0.41303924, 0.54983675, 0.61351615, 0.26339066,
                    0.54946023, 0.45080534, 0.40963298, 0.49352625, 0.503039  ,
                    0.37110674, 0.42217657, 0.4601355 , 0.46213433, 0.5465192 ,
                    0.83463174, 0.7007445 , 0.57617944, 0.5277977 , 0.41079056,
                    0.53652924, 0.52512354, 0.7739826 , 0.5759207 , 0.51219463,
                    0.40818158, 0.56241256, 0.5230139 , 0.7522604 , 0.5449328 ,
                    0.6019936 , 0.47727355, 0.44834647, 0.5226504 , 0.5386391 ,
                    0.51857436, 0.09808339, 0.22562303, 0.2989817 , 0.2680645 ,
                    0.51585126, 0.4227921 , 0.45080933, 0.5030598 , 0.5097167 ,
                    0.42817998, 0.14817655],np.float32)
            std_ = np.array([0.14532694, 0.11032472, 0.08916038, 0.09917132, 0.09768971,
                    0.09561172, 0.16109727, 0.08129608, 0.06458396, 0.13695784,
                    0.1461436 , 0.12895611, 0.15507314, 0.13876586, 0.14529936,
                    0.15133666, 0.10853891, 0.11738647, 0.17385288, 0.10945982,
                    0.07829959, 0.08052967, 0.1278781 , 0.09211544, 0.12313268,
                    0.10580523, 0.08960835, 0.13752691, 0.13089171, 0.09885687,
                    0.1255515 , 0.10698746, 0.09153594, 0.14290157, 0.13262388,
                    0.1312648 , 0.14476539, 0.10450402, 0.12679891, 0.13768515,
                    0.15336737, 0.09684499, 0.10031538, 0.20122908, 0.16501582,
                    0.12952238, 0.13572538, 0.12894315, 0.14531347, 0.17305142,
                    0.10940921, 0.10491683],np.float32)
            self.data = (self.data - mean_)/std_

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        # if self.transform is not None:
        #     data = self.transform(data)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)

class TE_challenge(data.Dataset):
    def __init__(
            self,
            root: str = 'data',
            train: bool = True,
            normalized = False,
            # transform: Optional[Callable] = None,
            # target_transform: Optional[Callable] = None,
    ) -> None:
        
        if train:            
            with open(os.path.join(root, 'TE_challenge','training'), 'rb') as f:
                entry = pickle.load(f)
                self.data = entry['data'].astype(np.float32)
                self.targets = entry['targets'].astype(np.int64)
        else:
            with open(os.path.join(root, 'TE_challenge','testing'), 'rb') as f:
                entry = pickle.load(f)
                self.data = entry['data'].astype(np.float32)

        if normalized:
            mean_ = np.array([0.86525279, 0.4848608 , 0.57071007, 0.89024389, 0.31661584,
                            0.3638963 , 0.83161989, 0.48864017, 0.28474732, 0.68250754,
                            0.37184247, 0.52390441, 0.83191496, 0.50922531, 0.46405147,
                            0.70963894, 0.49670272, 0.34762098, 0.51472577, 0.22100091,
                            0.53016744, 0.08732263, 0.7500555 , 0.41204334, 0.34948371,
                            0.54917318, 0.50793903, 0.51615337, 0.70286961, 0.33118116,
                            0.44634853, 0.48559026, 0.44110287, 0.55673136, 0.45287741,
                            0.4419828 , 0.56783558, 0.4113721 , 0.39488698, 0.47016514,
                            0.52658243, 0.85224781, 0.52844261, 0.02504429, 0.05343293,
                            0.9512636 , 0.678528  , 0.18044935, 0.68489327, 0.30713847,
                            0.5857508 ],np.float32)
            std_ = np.array([0.16100965, 0.11555204, 0.13002022, 0.04655812, 0.0808387 ,
                    0.11383163, 0.14385814, 0.13743503, 0.05636559, 0.1440101 ,
                    0.14362261, 0.13467256, 0.14184617, 0.14085732, 0.14464693,
                    0.13015504, 0.1407629 , 0.13410173, 0.13360171, 0.09738677,
                    0.04902142, 0.18377842, 0.16413715, 0.15815317, 0.14283748,
                    0.15388108, 0.16988248, 0.19964187, 0.12591586, 0.14294453,
                    0.10321026, 0.16656117, 0.17473866, 0.17773265, 0.15773381,
                    0.20061808, 0.19015855, 0.23686074, 0.14122578, 0.20932298,
                    0.16388074, 0.07944342, 0.21233973, 0.1041019 , 0.16471124,
                    0.12085955, 0.14899814, 0.07281286, 0.10007884, 0.10669552,
                    0.11156318],np.float32)
            self.data = (self.data - mean_)/std_

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        # if self.transform is not None:
        #     data = self.transform(data)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)

class TE_identification(data.Dataset):
    def __init__(
            self,
            root: str = 'data',
            train: bool = True,
            normalized = False,
            train_mode = 'normal',
            # transform: Optional[Callable] = None,
            # target_transform: Optional[Callable] = None,
    ) -> None:
        
        if train:     
            if train_mode == 'normal':
                with open(os.path.join(root, 'TE_idt','training_normal'), 'rb') as f:
                    entry = pickle.load(f)
                    self.data = entry['data'].astype(np.float32)
                    self.targets = entry['targets'].astype(np.int64)
            elif train_mode == 'all':
                with open(os.path.join(root, 'TE_idt','training_all'), 'rb') as f:
                    entry = pickle.load(f)
                    self.data = entry['data'].astype(np.float32)
                    self.targets = entry['targets'].astype(np.int64)
        else:
            with open(os.path.join(root, 'TE_idt','testing'), 'rb') as f:
                entry = pickle.load(f)
                self.data = entry['data'].astype(np.float32)
                self.targets = entry['targets'].astype(np.int64)
            
        if normalized:
            mean_ = np.array([0.8425227,0.5678867,0.5763916,0.89458144,0.32054517,
                              0.3663834,0.8068733,0.46543568,0.22362293,0.6681432,
                              0.36425006,0.5233977,0.8063752,0.51094943,0.4629864,
                              0.69721895,0.49643487,0.2638841,0.5334526,0.23371643,
                              0.19283044,0.119212836,0.34669545,0.5534244,0.038942166,
                              0.07760323,0.93406415,0.66405845,0.5163864,0.6805957,
                              0.0,0.24140666,0.47175452 ],np.float32)
            
            std_ = np.array([0.21381602,0.13166049,0.13139495,0.057063043,0.09027075,
                             0.122565545,0.18778099,0.14290985,0.071499355,0.17972773,
                             0.18095987,0.13364756,0.18450508,0.13997853,0.14668596,
                             0.17259853,0.14129703,0.19372427,0.1437435,0.12164309,
                             0.060138397,0.24241512,0.19034958,0.20831609,0.1385507,
                             0.21875075,0.1418301,0.18661071,0.13772465,0.12241695,
                             1e-8,0.15311077,0.14710413],np.float32)
            self.data = (self.data - mean_)/std_

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        # if self.transform is not None:
        #     data = self.transform(data)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)
    
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import MinMaxScaler

# with open('data/TE_challenge/training.csv', 'rb') as f:
#     x = np.loadtxt(f,str,delimiter=',')
# x = np.array(x, np.float32)
# y = np.array(x[:,0], np.int64)
# x = x[:,1:]

# ind = y<16
# y = y[ind]
# x = x[ind,:]
# xmeas = x[:,:22]
# xmv = x[:,-12:-1]
# x = np.hstack((xmeas,xmv))
# x = MinMaxScaler().fit_transform(x)
# mean_ = np.mean(x,axis=0)
# std_ = np.std(x,axis=0)
# std_[-3] += 1e-6
# print(",".join(str(i) for i in mean_))
# print(",".join(str(i) for i in std_))

# x_train_normal = x[y==0,:]
# y_train_normal = y[y==0]

# sss = StratifiedShuffleSplit(n_splits=5,test_size=0.25,random_state=0)
# sss = sss.split(list(range(len(y))), y)
# for _ in range(1):
#     train_idx, valid_idx = next(sss)
# x_train_all = x[train_idx,:]
# y_train_all = y[train_idx]
# y_test = y[valid_idx]
# x_test = x[valid_idx,:]
# # x_test = x_test[y_test!=0,:]
# # y_test = y_test[y_test!=0]


# normal_data = {'data':x_train_normal,'targets':y_train_normal}
# with open('data/TE_idt/training_normal','wb') as f:
#     pickle.dump(normal_data,f)
    
# all_data = {'data':x_train_all,'targets':y_train_all}
# with open('data/TE_idt/training_all','wb') as f:
#     pickle.dump(all_data,f)
    
# test_data = {'data':x_test,'targets':y_test}
# with open('data/TE_idt/testing','wb') as f:
#     pickle.dump(test_data,f)