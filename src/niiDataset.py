import glob
import os
import nibabel as nb
import numpy as np
import tensorflow as tf
import nibabel as nb
import re
import skimage
import copy
import metrics as me

def normalize(data):
    min = np.min(data)
    max = np.max(data)
    if not min == max:
        return min,max,(data - min)/(max - min)
    else:
        return min,max,data*0

def onehotLabel(labels):
    return (np.arange(labels.max()+1) == labels[...,None]).astype(int)

class NiiDataset(object):
    def __init__(self , searchPath , params):
        self.params = params

        data_paths = glob.glob(str(os.path.join(searchPath,"**/ct.nii.gz")),recursive=True)
        data_paths += glob.glob(str(os.path.join(searchPath,"**/0.nii.gz")),recursive=True)
        
        label_paths = glob.glob(str(os.path.join(searchPath,"**/label_ctv.nii.gz")),recursive=True)
        label_paths += glob.glob(str(os.path.join(searchPath,"**/1.nii.gz")),recursive=True)

        length = len(data_paths)
        ratio = [7,3]
        train_size = int(length* ratio[0]/sum(ratio))
        test_size = int(length* ratio[1]/sum(ratio))
        self.train_paths = (data_paths[:train_size],label_paths[:train_size])
        self.test_paths = (data_paths[train_size:],label_paths[train_size:])

    def load_train(self):
        self.train_niis = []
        for i in range(len(self.train_paths[0])):
            self.train_niis.append((nb.load(self.train_paths[0][i]),nb.load(self.train_paths[1][i])))

    def load_test(self):
        self.test_niis = []
        for i in range(len(self.test_paths[0])):
            self.test_niis.append((nb.load(self.test_paths[0][i]),nb.load(self.test_paths[1][i])))

    def get_train_dataset(self):
        max_s = self.params['max_scans']
        w = h = self.params['train_img_size']

        datas = []
        labels = []
        for i,nii in enumerate(self.train_niis):
            data = nii[0].get_data()
            label = nii[1].get_data()

            depth = np.shape(data)[-1]

            if depth < max_s :
                zero = np.zeros([np.shape(data)[0],np.shape(data)[1],max_s-depth])
                data = np.dstack((zero,data ))
                label = np.dstack((zero,label ))

            depth = np.shape(data)[-1]

            # depth = self.scans.shape[0]
            
            depth_low  = depth//2-max_s//2

            data = np.moveaxis(data,-1,0)
            min , max, data = normalize(data)
            data = skimage.transform.resize(
                image=data,
                output_shape=(depth,w, w )
            )
            datas.append(data[depth_low:depth_low+max_s,:,:])

            label = np.moveaxis(label,-1,0)
            n_classes = label.max()+1
            label = (np.arange(n_classes) == label[...,None]).astype(bool)

            label = skimage.img_as_bool(
                skimage.transform.resize(
                    image=label,
                    output_shape=(depth,w, w,n_classes)
                )
            )

            label = label.astype(int)
            labels.append(label[depth_low:depth_low+max_s,:,:])
            print("preprocess train data %s"%i)
            
            # self.depth_low  = depth//2-max_s//2
            # self.width_low = np.shape(data)[0]//2-w//2

            # datas.append(data[self.width_low:self.width_low+w,self.width_low:self.width_low+w,self.depth_low:self.depth_low+max_s])
            # labels.append(label[self.width_low:self.width_low+w,self.width_low:self.width_low+w,self.depth_low:self.depth_low+max_s])
        
        datas = np.asarray(datas,dtype=np.float32)
        labels = np.asarray(labels , dtype= np.int32)

        # self.train_min,self.train_max,datas = normalize(datas)
        # labels = onehotLabel(labels)
        # datas = np.expand_dims(datas,-1)

        # datas = np.moveaxis(datas, -2, 1)
        # labels = np.moveaxis(labels, -2, 1)

        datas = np.expand_dims(datas,-1)

        dataset = tf.data.Dataset.from_tensor_slices((datas,labels)).shuffle(
            buffer_size=24
        ).repeat().padded_batch(
            batch_size=self.params['batch_size'],
            padded_shapes=(
                [max_s, w, h, 1], [max_s, w, h, self.params['num_classes']]
            )
        )
        return dataset

    def get_test_dataset(self):
        max_s = self.params['max_scans']
        w = h = self.params['train_img_size']

        datas = []
        labels = []

        for i,nii in enumerate(self.test_niis):
            data = nii[0].get_data()
            label = nii[1].get_data()

            depth = np.shape(data)[-1]
            
            if depth < max_s :
                zero = np.zeros([np.shape(data)[0],np.shape(data)[1],max_s-depth])
                data = np.dstack((zero,data ))
                label = np.dstack((zero,label ))

            depth = np.shape(data)[-1]
            depth_low  = depth//2-max_s//2

            data = np.moveaxis(data,-1,0)
            min , max, data = normalize(data)
            data = skimage.transform.resize(
                image=data,
                output_shape=(depth,w, w )
            )
            datas.append(data[depth_low:depth_low+max_s,:,:])

            label = np.moveaxis(label,-1,0)
            n_classes = label.max()+1
            label = (np.arange(n_classes) == label[...,None]).astype(bool)

            label = skimage.img_as_bool(
                skimage.transform.resize(
                    image=label,
                    output_shape=(depth,w, w,n_classes)
                )
            )

            label = label.astype(int)
            labels.append(label[depth_low:depth_low+max_s,:,:])
            print("preprocess test data %s"%i)
        
        datas = np.asarray(datas,dtype=np.float32)
        labels = np.asarray(labels , dtype= np.int32)

        datas = np.expand_dims(datas,-1)

        dataset = tf.data.Dataset.from_tensor_slices((datas,labels)).padded_batch(
            # we have different sized test scans so we need batch 1
            batch_size=1,
            padded_shapes=(
                [max_s, None, None, 1],
                [max_s, None, None, self.params['num_classes']]
            )
        )
        return dataset

    def get_test_dataset_noneresize(self):
        max_s = self.params['max_scans']
        w = h = self.params['train_img_size']

        datas = []
        labels = []
        for nii in self.test_niis:
            data = nii[0].get_data()
            label = nii[1].get_data()

            depth = np.shape(data)[-1]
            
            if depth < max_s :
                zero = np.zeros([np.shape(data)[0],np.shape(data)[1],max_s-depth])
                data = np.dstack((zero,data ))
                label = np.dstack((zero,label ))

            depth = np.shape(data)[-1]
            depth_low  = depth//2-max_s//2

            data = np.moveaxis(data,-1,0)
            min , max, data = normalize(data)
            datas.append(data[depth_low:depth_low+max_s,:,:])

            label = np.moveaxis(label,-1,0)
            n_classes = label.max()+1
            label = (np.arange(n_classes) == label[...,None]).astype(int)
            labels.append(label[depth_low:depth_low+max_s,:,:])
            print("preprocess test data %s"%i)
        
        datas = np.asarray(datas,dtype=np.float32)
        labels = np.asarray(labels , dtype= np.int32)

        datas = np.expand_dims(datas,-1)

        dataset = tf.data.Dataset.from_tensor_slices((datas,labels)).padded_batch(
            # we have different sized test scans so we need batch 1
            batch_size=1,
            padded_shapes=(
                [max_s, None, None, 1],
                [max_s, None, None, self.params['num_classes']]
            )
        )
        return dataset


    def save_test_pred(self,prediction,savepath):
        dsc_s = 0
        dsc_ss = 0
        avd_s = 0
        hd_s = 0
        for i,pred in enumerate( prediction ):
            max_s = self.params['max_scans']

            nii_data = self.test_niis[i][0]
            depth = np.shape(nii_data.get_data())[-1]

            pred = pred['truth']
            pred = pred.astype(bool)
            pred = skimage.img_as_bool(
                 skimage.transform.resize(
                    image=pred,
                    output_shape=(max_s,512, 512 )
                )
            )
            pred = pred.astype('int16')
            pred = np.moveaxis(pred,0,-1)

            template = np.zeros([512,512,depth]).astype('int16')
            


            if depth < max_s:
                depth_low  = max_s//2 - depth//2
                template = pred[:,:,depth_low:depth_low+depth]
                gt = self.test_niis[i][1].get_data()
                pred = pred[:,:,depth_low:depth_low+depth]


            else:
                depth_low  = depth//2-max_s//2
                template[:,:,depth_low:depth_low+max_s] = pred
                gt = self.test_niis[i][1].get_data()[:,:,depth_low:depth_low+max_s]

            dsc_label_1 = me.dice_with_class(pred,gt,1)
            dsc_label_1g = me.dice_with_class(template,self.test_niis[i][1].get_data(),1)
            avd = me.avd_with_class(pred,gt,1)
            hd = me.hd_with_class(pred,gt,1)

            patient = re.match(r".*/(\w+)/.*?.nii.gz",self.test_paths[0][i]).group(1)
            print("%s\t\tdsc:%.4f\tdsc_g:%.4f\tavd:%.4f\thd:%.4f"%(patient,dsc_label_1,dsc_label_1g,avd,hd))

            save_nii = nb.Nifti1Image(template, nii_data.affine)
            nb.save(save_nii,os.path.join(savepath,"%s_pred.nii.gz"%patient))

            dsc_s +=dsc_label_1
            dsc_ss += dsc_label_1g
            avd_s += avd
            hd_s += hd
        l = len(self.test_niis)
        print("%.4f,%.4f,%.4f,%.4f" % (dsc_s/l,dsc_ss/l,avd_s/l,hd_s/l))

    def input_fun(self,train):
        if train:
            dataset = self.get_train_dataset()
        else:
            dataset = self.get_test_dataset()
        iterator = tf.data.Iterator.from_structure(
            dataset.output_types,
            dataset.output_shapes
        )
        dataset_init_op = iterator.make_initializer(dataset)
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_init_op)
        next_element = iterator.get_next()

        # extremely hack way of getting tf.estimator to return labels at pred time
        # see https://github.com/tensorflow/tensorflow/issues/17824
        features = {'x': next_element[0], 'y': next_element[1]}
        return features, next_element[1]



