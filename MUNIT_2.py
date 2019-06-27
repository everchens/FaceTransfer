from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import batch_and_drop_remainder
import numpy as np
import cv2

class MUNIT(object) :
    def __init__(self, sess, args):
        self.model_name = 'MUNIT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.num_style = args.num_style # for test
        self.guide_img = args.guide_img
        self.direction = args.direction

        self.img_h = args.img_h
        self.img_w = args.img_w
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.gan_w = args.gan_w
        self.recon_x_w = args.recon_x_w
        self.recon_s_w = args.recon_s_w
        self.recon_c_w = args.recon_c_w
        self.recon_x_cyc_w = args.recon_x_cyc_w

        """ Generator """
        self.n_res = args.n_res
        self.mlp_dim = pow(2, args.n_sample) * args.ch # default : 256

        self.n_downsample = args.n_sample
        self.n_upsample = args.n_sample
        self.style_dim = args.style_dim

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.trainC_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainC'))
        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset), len(self.trainC_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# style in test phase : ", self.num_style)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# Style dimension : ", self.style_dim)
        print("# MLP dimension : ", self.mlp_dim)
        print("# Down sample : ", self.n_downsample)
        print("# Up sample : ", self.n_upsample)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)
        print("# Multi-scale Dis : ", self.n_scale)

    ##################################################################################
    # Encoder and Decoders
    ##################################################################################

    def Content_Encoder(self, x, reuse=False, scope='content_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            x = relu(x)

            for i in range(self.n_downsample) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i+1))
                x = instance_norm(x, scope='ins_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(self.n_res) :
                x = resblock(x, channel, scope='resblock_'+str(i))

            return x

    def generator(self, contents, reuse=False, scope="decoder"):
        channel = self.mlp_dim
        with tf.variable_scope(scope, reuse=reuse) :
            x = contents
            for i in range(self.n_res) :
                x = resblock(x, channel, scope='resblock'+str(i))

            for i in range(self.n_upsample) :
                # # IN removes the original feature mean and variance that represent important style information
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect', scope='conv_'+str(i))
                x = layer_norm(x, scope='layer_norm_'+str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        D_cls = []
        with tf.variable_scope(scope, reuse=reuse) :
            for scale in range(self.n_scale) :
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.2)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) +'conv_' + str(i))
                    x = lrelu(x, 0.2)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, scope='ms_' + str(scale) + 'D_logit')
                D_logit.append(x)
                
                x_init = down_sample(x_init)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def Encoder_A(self, x_A, reuse=False):
        content_A = self.Content_Encoder(x_A, reuse=reuse, scope='content_encoder')

        return content_A#, style_A

    def Encoder_B(self, x_B, reuse=True):
        content_B = self.Content_Encoder(x_B, reuse=reuse, scope='content_encoder')

        return content_B#, style_B

    def Decoder_A(self, content_A, content_B, reuse=False):
        x_aa = self.generator(contents=content_A, reuse=reuse, scope='decoder_A')
        x_ba = self.generator(contents=content_B, reuse=True, scope='decoder_A')

        return x_aa, x_ba
    
    def Decoder_B(self, content_A, content_B, reuse=False):
        x_ab = self.generator(contents=content_A, reuse=reuse, scope='decoder_B')
        x_bb = self.generator(contents=content_B, reuse=True, scope='decoder_B')

        return x_ab, x_bb

    def discriminate_real(self, x_A, x_B):#, x_C):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")
        
        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):#, x_cb, x_ac, x_bc):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def gradient_panalty(self, real, fake, scope="discriminator"):
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit = self.discriminator(interpolated, reuse=True, scope=scope)

        GP = 0

        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp' :
            GP = 10 * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = 10 * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_h, self.img_w, self.img_ch, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

        trainA = trainA.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()
        trainB = trainB.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()

        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()

        self.domain_A = trainA_iterator.get_next()
        self.domain_B = trainB_iterator.get_next()

        """ Define Encoder, Generator, Discriminator """
        # encode
        content_a = self.Encoder_A(self.domain_A)
        content_b = self.Encoder_B(self.domain_B, reuse=True)

        # decode (within domain and cross domain)
        x_aa, x_ba = self.Decoder_A(content_A=content_a, content_B=content_b)
        x_ab, x_bb = self.Decoder_B(content_A=content_a, content_B=content_b)

        # encode again
        content_aa_ = self.Encoder_A(x_aa, reuse=True)
        content_ab_ = self.Encoder_B(x_ab, reuse=True)
        content_ba_ = self.Encoder_A(x_ba, reuse=True)
        content_bb_ = self.Encoder_B(x_bb, reuse=True)
 
        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

        """ Define Loss """
        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
            GP_ba = self.gradient_panalty(real=self.domain_A, fake=x_ba, scope="discriminator_A")
            GP_ab = self.gradient_panalty(real=self.domain_B, fake=x_ab, scope="discriminator_B")
        else :
            GP_ba = GP_ab = 0


        G_ad_loss_a = generator_loss(self.gan_type, fake_A_logit)
        G_ad_loss_b = generator_loss(self.gan_type, fake_B_logit)

        D_ad_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit) + GP_ba
        D_ad_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit) + GP_ab
        
        recon_A = L1_loss(x_aa, self.domain_A)
        recon_B = L1_loss(x_bb, self.domain_B)

        # The style reconstruction loss encourages
        # diverse outputs given different style codes
        #recon_style_A = L1_loss(style_a_, self.style_a)
        #recon_style_B = L1_loss(style_b_, self.style_b)

        # The content reconstruction loss encourages
        # the translated image to preserve semantic content of the input image
        recon_content_A = L1_loss(content_aa_, content_a) + L1_loss(content_ab_, content_a)
        recon_content_B = L1_loss(content_ba_, content_b) + L1_loss(content_bb_, content_b)


        Generator_A_loss = self.gan_w * G_ad_loss_a + \
                           self.recon_x_w * recon_A + \
                           self.recon_c_w * recon_content_A
                                                      

        Generator_B_loss = self.gan_w * G_ad_loss_b + \
                           self.recon_x_w * recon_B + \
                           self.recon_c_w * recon_content_B
                           
                           
        
        Discriminator_A_loss = self.gan_w * D_ad_loss_a
        Discriminator_B_loss = self.gan_w * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.9).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        """ Image """
        self.fake_BA = x_ba
        self.fake_AB = x_ab

        self.real_A = self.domain_A
        self.real_B = self.domain_B

        """ Test """
        self.test_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='test_image')

        test_content_a = self.Encoder_A(self.test_image, reuse=True)
        test_content_b = self.Encoder_B(self.test_image, reuse=True)

        self.test_fake_AA, self.test_fake_BA = self.Decoder_A(content_A=test_content_a, content_B=test_content_b,  reuse=True)
        self.test_fake_AB, self.test_fake_BB = self.Decoder_B(content_A=test_content_a, content_B=test_content_b,  reuse=True)


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()

        if type == 'wgan-gp' :
            n_critic = 5    #5
        else :
            n_critic = 1

        for epoch in range(start_epoch, self.epoch):

            lr = self.init_lr * pow(0.5, epoch)

            for idx in range(start_batch_id, self.iteration):
                
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                for critic_itr in range(n_critic):
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                # Update G
                
                batch_A_images, batch_B_images, fake_BA, fake_AB, _, g_loss, summary_str = self.sess.run([self.real_A, self.real_B, self.fake_BA, self.fake_AB, self.G_optim, self.Generator_loss, self.G_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    save_images(batch_B_images, [self.batch_size, 1],
                                './{}/real_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

                    save_images(fake_BA, [self.batch_size, 1],
                                './{}/fake_BA_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_AB, [self.batch_size, 1],
                                './{}/fake_AB_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx+1, self.save_freq) == 0 :
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        fps = 5
        img_size = (512,256)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0

        video_path = os.path.join(self.result_dir, 'A2B.avi')
        videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
        
        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style) :
                test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
                image_path = os.path.join(self.result_dir, 'A_{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_AB, feed_dict = {self.test_image : sample_image})
                stack_img = np.concatenate([sample_image,fake_img], axis=2)
                save_images(stack_img, [1, 1], image_path)

                #videoWriter.write(np.uint8(stack_img))
                frame = cv2.imread(image_path)
                videoWriter.write(frame)    
        videoWriter.release()
        
        
        video_path = os.path.join(self.result_dir, 'B2A.avi')
        videoWriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)
        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]
            

            for i in range(self.num_style):
                image_path = os.path.join(self.result_dir, 'B_{}_style{}.{}'.format(file_name, i, file_extension))
                

                fake_img = self.sess.run(self.test_fake_BA, feed_dict={self.test_image: sample_image})
                stack_img = np.concatenate([sample_image,fake_img], axis=2)
                save_images(stack_img, [1, 1], image_path)

                #videoWriter.write(np.uint8(stack_img))
                frame = cv2.imread(image_path)
                videoWriter.write(frame)    
        videoWriter.release()
