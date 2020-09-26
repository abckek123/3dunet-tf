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
import math

def normalize(data):
    min = np.min(data)
    max = np.max(data)
    if not min == max:
        return min,max,(data - min)/(max - min)
    else:
        return min,max,data*0

def onehotLabel(labels):
    return (np.arange(labels.max()+1) == labels[...,None]).astype(int)

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

class NiiDataset(object):
    def __init__(self , searchPath , params):
        self.params = params
        self.overlap = 4

        data_paths = glob.glob(str(os.path.join(searchPath,"**/ct.nii.gz")),recursive=True)
        data_paths += glob.glob(str(os.path.join(searchPath,"**/0.nii.gz")),recursive=True)
        
        label_paths = glob.glob(str(os.path.join(searchPath,"**/label_ctv.nii.gz")),recursive=True)
        label_paths += glob.glob(str(os.path.join(searchPath,"**/1.nii.gz")),recursive=True)


        data_chunk = chunks(data_paths,4)
        label_chunk = chunks(label_paths,4)

        train_batch = self.params['train_batch']
        test_batch = self.params['test_batch']

        self.train_paths = [[],[]]
        self.test_paths = [[],[]]
        for batch in train_batch:
            self.train_paths[0] += data_chunk[batch]
            self.train_paths[1] += label_chunk[batch]

        for batch in test_batch:
            self.test_paths[0] += data_chunk[batch]
            self.test_paths[1] += label_chunk[batch]

        print(train_batch,test_batch)

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

        self.train_volume = []
        for i,nii in enumerate(self.train_niis):
            data = nii[0].get_data()
            label = nii[1].get_data()

            depth = np.shape(data)[-1]

            data = np.moveaxis(data,-1,0)
            min , max, data = normalize(data)
            data = skimage.transform.resize(
                image=data,
                output_shape=(depth,w, w )
            )

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

            self.train_volume.append([])
            de = 0
            while de < depth:
                begin = de - self.overlap
                begin = begin if begin>=0 else 0
                end = begin  + max_s
                end = end if end < depth else depth

                self.train_volume[i].append([begin,end])
                data_volume = data[begin:end,:,:]
                label_volume = label[begin:end,:,:]

                d = end - begin
                if d < max_s:
                    zerod = np.zeros((max_s-d,w,w))
                    zerol = np.zeros((max_s-d,w,w,n_classes))
                    data_volume = np.concatenate((data_volume,zerod))
                    label_volume = np.concatenate((label_volume,zerol))

                datas.append(data_volume)
                labels.append(label_volume)

                de = end
                
            print("preprocess train data %s"%i)
                    
        datas = np.asarray(datas,dtype=np.float32)
        labels = np.asarray(labels , dtype= np.int32)

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

        self.test_volume = []

        for i,nii in enumerate(self.test_niis):
            data = nii[0].get_data()
            label = nii[1].get_data()

            depth = np.shape(data)[-1]

            data = np.moveaxis(data,-1,0)
            min,max,data = normalize(data)
            data = skimage.transform.resize(
                image=data,
                output_shape=(depth,w, w )
            )
            
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

            self.test_volume.append([])
            de = 0
            while de < depth:
                begin = de - self.overlap
                begin = begin if begin>=0 else 0
                end = begin + max_s
                end = end if end < depth else depth

                self.test_volume[i].append([begin,end])
                data_volume = data[begin:end,:,:]
                label_volume = label[begin:end,:,:]

                d = end - begin
                if d < max_s:
                    zerod = np.zeros((max_s-d,w,w))
                    zerol = np.zeros((max_s-d,w,w,n_classes))
                    data_volume = np.concatenate((data_volume,zerod))
                    label_volume = np.concatenate((label_volume,zerol))

                datas.append(data_volume)
                labels.append(label_volume)

                de = end
                
            print("preprocess test data %s"%i,self.test_volume[i])
        
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

        prediction = list(prediction)
        pre_i = 0
        max_s = self.params['max_scans']
        for i,volume in enumerate(self.test_volume):
            nii_data = self.test_niis[i][0]
            nii_label = self.test_niis[i][1]

            depth = np.shape(nii_data.get_data())[-1]
            for r in volume:
                pred = prediction[pre_i]['classes']
                pred = pred.astype(bool)
                pred = skimage.img_as_bool(
                    skimage.transform.resize(
                        image=pred,
                        output_shape=(max_s,512, 512 )
                    )
                )
                pred = pred.astype('int16')

                if r[0]==0:
                    template = pred[r[0]:r[1],:,:]
                else:
                    template = np.concatenate((template,pred[self.overlap:r[1]-r[0],:,:]))
                print("predictions: %d/%d"%(pre_i,len(prediction)))
                pre_i += 1
                
            template = np.moveaxis(template,0,-1)
            template = np.asarray(template,dtype = 'int16')
            assert depth == np.shape(template)[-1]

            gt = nii_label.get_data()
            dsc_label_1g = me.dice_with_class(template,gt,1)
            avd = me.avd_with_class(template,gt,1)
            hd = me.hd_with_class(template,gt,1)

            patient = re.match(r".*/(\w+)/.*?.nii.gz",self.test_paths[0][i]).group(1)
            print("%s\t\tdsc:%.4f\tavd:%.4f\thd:%.4f"%(patient,dsc_label_1g,avd,hd))

            save_nii = nb.Nifti1Image(template, nii_data.affine)
            nb.save(save_nii,os.path.join(savepath,"%s_pred.nii.gz"%patient))

            dsc_ss += dsc_label_1g
            avd_s += avd
            hd_s += hd
        l = len(self.test_niis)
        print(self.params['train_batch'],"%.4f,%.4f,%.4f" % (dsc_ss/l,avd_s/l,hd_s/l))

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



