from scipy import io, spatial
import os
import numpy as np

class DataLoader:
    def __init__(self, path: str, binary:bool = True):
        self.path = path
        self.res101 = io.loadmat(os.path.join(path,'res101.mat'))
        if binary:
            self.att_splits = io.loadmat(os.path.join(path,'binaryAtt_splits.mat'))
        else:
            self.att_splits = io.loadmat(os.path.join(path,'att_splits.mat'))
        # print(self.att_splits['all_class_names'])
        # print(self.att_splits.items())
        # print(self.att_splits['train_loc'])

        # print(self.res101.items())

        
    def load(self):
        train_loc = 'train_loc'
        val_loc = 'val_loc'
        test_loc = 'test_unseen_loc'
 
        feat = self.res101['features']
        # Shape -> (dxN)
        X_train = feat[:, np.squeeze(self.att_splits[train_loc]-1)]
        X_val = feat[:, np.squeeze(self.att_splits[val_loc]-1)]
        X_test = feat[:, np.squeeze(self.att_splits[test_loc]-1)]

        labels = self.res101['labels']
        labels_train = np.squeeze(labels[np.squeeze(self.att_splits[train_loc]-1)])
        labels_val = np.squeeze(labels[np.squeeze(self.att_splits[val_loc]-1)])
        labels_test = np.squeeze(labels[np.squeeze(self.att_splits[test_loc]-1)])
        
        train_labels_seen = np.unique(labels_train)
        val_labels_unseen = np.unique(labels_val)
        test_labels_unseen = np.unique(labels_test)

        train_labels_seen = np.array([x-1 for x in train_labels_seen])

        val_labels_unseen= np.array([x-1 for x in val_labels_unseen])

        test_labels_unseen = np.array([x-1 for x in test_labels_unseen])
        
        i=0
        for labels in train_labels_seen:
            labels_train[labels_train == labels] = i    
            i+=1
        
        j=0
        for labels in val_labels_unseen:
            labels_val[labels_val == labels] = j
            j+=1
        
        k=0
        for labels in test_labels_unseen:
            labels_test[labels_test == labels] = k
            k+=1
        
        sig = self.att_splits['att']
        # Shape -> (Number of attributes, Number of Classes)
        train_sig = sig[:, train_labels_seen]
        val_sig = sig[:, val_labels_unseen]
        test_sig = sig[:, test_labels_unseen]


        sorted_ind =  np.concatenate([val_labels_unseen , train_labels_seen], axis=0)
        sorted_ind=np.sort(sorted_ind)
        train_sig_fin = sig[:, sorted_ind]

        testClasses = test_labels_unseen
        trainClasses =  np.concatenate([train_labels_seen , val_labels_unseen], axis=0)

        sorted_ind_all =  np.concatenate([val_labels_unseen , train_labels_seen], axis=0)
        sorted_ind_all =  np.concatenate([sorted_ind_all , test_labels_unseen], axis=0)
        sorted_ind_all=np.sort(sorted_ind_all)



        # Loads features
        X = self.res101['features'].transpose()
        Y_temp = self.res101['labels']
        Y_temp=Y_temp.transpose()
        Y_temp=Y_temp[0]
        Y = np.array([x-1 for x in Y_temp])
        Y =Y.astype(np.int32).transpose()
        #Y1 =np.array(Y1.tolist())
        ATTR =  self.att_splits['att']
        att =  self.att_splits['att']
        att=np.transpose(att)
        noExs = X.shape[0]


        trainDataX = []
        trainDataLabels = []
        trainDataAttrs = []

        testDataX = []
        testDataLabels = []
        testDataAttrs = []



        for ii in range(0,noExs):
            if(Y[ii] in trainClasses):
                trainDataX = trainDataX + [X[ii]]
                trainDataLabels = trainDataLabels + [Y[ii]]
                trainDataAttrs = trainDataAttrs + [att[Y[ii]]]
            elif(Y[ii] in testClasses):
                #print(str(Y[ii])  + "  is in   " + str(testClasses))
                testDataX = testDataX + [X[ii]]
                testDataLabels = testDataLabels + [Y[ii]]
                testDataAttrs = testDataAttrs + [att[Y[ii]]]
            else:
                print('Fatal Error... Please check code/data')
            
            
        trainDataX = np.array(trainDataX)
        trainDataLabels = np.array(trainDataLabels)
        trainDataAttrs = np.array(trainDataAttrs)

        testDataX = np.array(testDataX)
        testDataLabels = np.array(testDataLabels)
        testDataAttrs = np.array(testDataAttrs)

        return (trainDataX, trainDataLabels, trainDataAttrs, testDataX, testDataLabels, testDataAttrs)