from glob import glob
import numpy as np
import os
import tensorflow.compat.v1 as tf
import SimpleITK as sitk
import nibabel
import nibabel.processing
import argparse
from scipy.ndimage import zoom
tf.disable_v2_behavior()


class auto_encoder(object):
    def __init__(self, sess):
        self.sess           = sess
        self.phase          = 'train'
        self.batch_size     = 1
        self.inputI_size    = 128
        self.inputI_chn     = 1
        self.output_chn     = 12
        self.lr             = 0.0001
        self.beta1          = 0.3
        self.epoch          = 500
        self.model_name     = 'n1.model'
        self.save_intval    = 20
        self.build_model()
        self.chkpoint_dir   = "./ckpt"
        self.train_data_dir = "./real_train/incomplete/"
        self.train_label_dir = "./real_train/complete"
        self.test_data_dir = "./real_test/incomplete"

        self.test_label_dir="./Dtest4/dataset/0_ground_truth/lung/"
        self.save_output_dir = "./output_multiclass/"
        self.save_residual_dir = "./output_multiclass/residual/"


    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, 12)
        dice = 0
        for i in range(12):
            inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2*inse/(l+r)
        return -dice



    def conv3d(self,input, output_chn, kernel_size, stride, use_bias=False, name='conv'):
        return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                                padding='same', data_format='channels_last',
                                kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),use_bias=use_bias, name=name)


    def conv_bn_relu(self,input, output_chn, kernel_size, stride, use_bias, is_training, name):
        with tf.variable_scope(name):
            conv = self.conv3d(input, output_chn, kernel_size, stride, use_bias, name='conv')
            relu = tf.nn.relu(conv, name='relu')
        return relu



    def Deconv3d(self,input, output_chn, name):
        batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
        filter = tf.get_variable(name+"/filter", shape=[4, 4, 4, output_chn, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.01))
        conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, in_depth * 2, in_height * 2, in_width * 2, output_chn],
                                      strides=[1, 2, 2, 2, 1], padding="SAME", name=name)
        return conv



    def deconv_bn_relu(self,input, output_chn, is_training, name):
        with tf.variable_scope(name):
            conv = self.Deconv3d(input, output_chn, name='deconv')
            relu = tf.nn.relu(conv, name='relu')
        return relu




    def build_model(self):
        print('building patch based model...')       
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,self.inputI_size,self.inputI_size,128, self.inputI_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int64, shape=[self.batch_size,self.inputI_size,self.inputI_size,128,1], name='target')
        self.soft_prob , self.task0_label = self.encoder_decoder(self.input_I)
        self.main_dice_loss = self.dice_loss_fun(self.soft_prob, self.input_gt[:,:,:,:,0])
        self.dice_loss=200000000*self.main_dice_loss
        self.Loss = self.dice_loss
        self.saver = tf.train.Saver()


    def encoder_decoder(self, inputI):
        phase_flag = (self.phase=='train')
        conv1_1 = self.conv3d(input=inputI, output_chn=64, kernel_size=3, stride=2, use_bias=True, name='conv1')
        conv1_relu = tf.nn.relu(conv1_1, name='conv1_relu')
        conv2_1 = self.conv3d(input=conv1_relu, output_chn=128, kernel_size=3, stride=2, use_bias=True, name='conv2')
        conv2_relu = tf.nn.relu(conv2_1, name='conv2_relu')
        conv3_1 = self.conv3d(input=conv2_relu, output_chn= 256, kernel_size=3, stride=2, use_bias=True, name='conv3a')
        conv3_relu = tf.nn.relu(conv3_1, name='conv3_1_relu')
        conv4_1 = self.conv3d(input=conv3_relu, output_chn=512, kernel_size=3, stride=2, use_bias=True, name='conv4a')
        conv4_relu = tf.nn.relu(conv4_1, name='conv4_1_relu')
        conv5_1 = self.conv3d(input=conv4_relu, output_chn=512, kernel_size=3, stride=1, use_bias=True, name='conv5a')
        conv5_relu = tf.nn.relu(conv5_1, name='conv5_1_relu')
        feature= self.conv_bn_relu(input=conv5_relu, output_chn=256, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='conv6_1')
        deconv1_1 = self.deconv_bn_relu(input=feature, output_chn=256, is_training=phase_flag, name='deconv1_1')
        deconv1_2 = self.conv_bn_relu(input=deconv1_1, output_chn=128, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv1_2')
        deconv2_1 = self.deconv_bn_relu(input=deconv1_2, output_chn=128, is_training=phase_flag, name='deconv2_1')
        deconv2_2 = self.conv_bn_relu(input=deconv2_1, output_chn=64, kernel_size=3,stride=1, use_bias=True, is_training=phase_flag, name='deconv2_2')
        deconv3_1 = self.deconv_bn_relu(input=deconv2_2, output_chn=64, is_training=phase_flag, name='deconv3_1')
        deconv3_2 = self.conv_bn_relu(input=deconv3_1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv3_2')
        deconv4_1 = self.deconv_bn_relu(input=deconv3_2, output_chn=32, is_training=phase_flag, name='deconv4_1')
        deconv4_2 = self.conv_bn_relu(input=deconv4_1, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='deconv4_2')
        pred_prob1 = self.conv_bn_relu(input=deconv4_2, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='pred_prob1')
        pred_prob = self.conv3d(input=pred_prob1, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob')
        pred_prob2 = self.conv3d(input=pred_prob, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob2')
        pred_prob3 = self.conv3d(input=pred_prob2, output_chn=self.output_chn, kernel_size=3, stride=1, use_bias=True, name='pred_prob3')
        soft_prob=tf.nn.softmax(pred_prob3,name='task_0')
        task0_label=tf.argmax(soft_prob,axis=4,name='argmax0')
        return  soft_prob,task0_label


    def train(self):
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.Loss)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        counter=1

        train_label_list=glob('{}/*.nii.gz'.format(self.train_label_dir))

        i=0
        for epoch in np.arange(self.epoch):
            i=i+1
            print('epoch:',i )
            for j in range(len(train_label_list)):
                labelImg=sitk.ReadImage(train_label_list[j])
                labelNpy=sitk.GetArrayFromImage(labelImg)
                labelNpy_resized=zoom(labelNpy,(128/labelNpy.shape[0],128/labelNpy.shape[1],128/labelNpy.shape[2]),order=0, mode='constant')
                labelNpy_resized=np.expand_dims(np.expand_dims(labelNpy_resized,axis=0),axis=4) 
                name=train_label_list[j][-len('_full.nii.gz')-len('s0556'):-len('_full.nii.gz')]
                for k in range(10):
                    data_dir=self.train_data_dir+str(name)+'./'+str(name)+'_%d'%k+'.nii.gz'
                    trainImg=sitk.ReadImage(data_dir)
                    trainNpy=sitk.GetArrayFromImage(trainImg)
                    trainNpy_resized=zoom(trainNpy,(128/trainNpy.shape[0],128/trainNpy.shape[1],128/trainNpy.shape[2]),order=0, mode='constant')
                    trainNpy_resized=np.expand_dims(np.expand_dims(trainNpy_resized,axis=0),axis=4) 
                    _, cur_train_loss = self.sess.run([u_optimizer, self.Loss], feed_dict={self.input_I: trainNpy_resized, self.input_gt: labelNpy_resized})
                    train_output0 = self.sess.run(self.task0_label, feed_dict={self.input_I: trainNpy_resized})
                    print('sum for current training whole: %.8f, pred whole:  %.8f'%(np.sum(labelNpy_resized),np.sum(train_output0)))        
                    print('current training loss:',cur_train_loss)
            counter+=1
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)

        self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)


    def test(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load_chkpoint(self.chkpoint_dir):
            print(" *****Successfully load the checkpoint**********")
        else:
            print("*******Fail to load the checkpoint***************")

        test_list=glob('{}/*.nii.gz'.format(self.test_data_dir))

        k=1
        for i in range(len(test_list)):

            ### input 
            print(test_list[i])                    
            test_img=sitk.ReadImage(test_list[i])
            test_input = sitk.GetArrayFromImage(test_img)
            test_input_resized_ = zoom(test_input,(256/test_input.shape[0],256/test_input.shape[1],128/test_input.shape[2]),order=0, mode='constant')
            test_input_resized_[test_input_resized_>12]=0
            test_input_resized_[test_input_resized_<0]=0
            print('test_input_resized_',np.unique(test_input_resized_))
            test_input_resized=np.expand_dims(np.expand_dims(test_input_resized_,axis=0),axis=4)


            ## prediction
            test_output = self.sess.run(self.task0_label, feed_dict={self.input_I: test_input_resized})
            print(test_output.shape)
            print(np.unique(test_output))


            ## output
            filename=self.save_output_dir+test_list[i][-7-len('s0332_0'):-7]+'.nii.gz'
            resize2original=1

            if resize2original:
                print('resizing predictions...')

                test_output=zoom(test_output[0],(test_input.shape[0]/256,test_input.shape[1]/256,test_input.shape[2]/128),order=0, mode='constant')
                print(test_output.shape)

                test_output[test_output>12]=0
                test_output[test_output<0]=0
                test_pred=sitk.GetImageFromArray(test_output.astype('int32'))
                test_pred.CopyInformation(test_img)
                sitk.WriteImage(test_pred,filename)

            else:
                print('resizing input...')
                #test_img_downsampled=self.downsamplePatient(test_img,test_input.shape[0]/256,test_input.shape[1]/256,test_input.shape[2]/128)
                print('resizing done...')

                input_img = nibabel.load(test_list[i])

                voxel_size=input_img.header.get_zooms()
                voxel_size_new=[voxel_size[0]*(test_input.shape[0]/256),voxel_size[1]*(test_input.shape[1]/256),voxel_size[2]*(test_input.shape[2]/128)]
                resampled_img = nibabel.processing.resample_to_output(input_img, voxel_size_new)
                filename_img=self.save_output_dir+test_list[i][-7-len('s0332_0'):-7]+'_org'+'.nii.gz'
                nibabel.save(resampled_img, filename_img)


                test_pred=sitk.GetImageFromArray(test_output[0].astype('int32'))
                sitk.WriteImage(test_pred,filename)


            k+=1
            #filename_res=self.save_residual_dir+test_list[i][-7-len('s0332_0'):-7]+'.nii.gz'
            #res_output=test_output-test_input
            #res_output_img=sitk.GetImageFromArray(res_output.astype('int32'))
            #res_output_img.CopyInformation(test_img)
            #sitk.WriteImage(res_output_img,filename_res)



    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s" % ('ckpt')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)



    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s" % ('ckpt')
        print('########################################################')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase")
    args = parser.parse_args()

    sess1 = tf.compat.v1.Session()
    with sess1.as_default():
        with sess1.graph.as_default():
            model = auto_encoder(sess1)
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('trainable params:',total_parameters)

    if args.phase == "train":
        print('training model...')
        model.train()
    if args.phase == "test":
        print('testing model...')
        model.test()
