#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Get the FCN graph
    graph = tf.get_default_graph()
    # Extract the encoder layers
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob_tensor, l3_tensor, l4_tensor, l7_tensor


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # DECODER

    # 1x1 convolution on the 7th encoder layer
    l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                   filters=num_classes,
                                   kernel_size=1,
                                   strides=(1, 1),
                                   padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                   )
    # Upsample the 1x1 convolution layer
    decoder_l1_out_deconv = tf.layers.conv2d_transpose(l7_conv_1x1,
                                                       filters=num_classes,
                                                       strides=(2, 2),
                                                       kernel_size=4,
                                                       padding='same',
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                                       )
    # do a 1x1 convolution on the 4th layer for a skip connection
    vgg_layer4_1x1 = tf.layers.conv2d(vgg_layer4_out,
                                      filters=num_classes,
                                      kernel_size=1,
                                      strides=(1, 1),
                                      padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                      )
    decoder_l1_out = tf.add(decoder_l1_out_deconv, vgg_layer4_1x1)

    # Upsample
    decoder_l2_out_deconv = tf.layers.conv2d_transpose(decoder_l1_out,
                                                       filters=num_classes,
                                                       strides=(2, 2),
                                                       kernel_size=4,
                                                       padding='same',
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                                       )
    # do a 1x1 convolution on the 3rd layer for a skip connection
    vgg_layer3_1x1 = tf.layers.conv2d(vgg_layer3_out,
                                      filters=num_classes,
                                      kernel_size=1,
                                      strides=(1, 1),
                                      padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                      )
    decoder_l2_out = tf.add(decoder_l2_out_deconv, vgg_layer3_1x1)

    # Add final layer
    # Upsample
    output_layer = tf.layers.conv2d_transpose(decoder_l2_out,
                                              filters=num_classes,
                                              kernel_size=16,
                                              strides=(8, 8),
                                              padding='same',
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
                                              )
    return output_layer


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    true_label = tf.reshape(correct_label, (-1, num_classes))

    # define cross-entropy loss
    x_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=true_label,
                                                                       )
                               )
    # Instantiate the Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_train = optimizer.minimize(x_entropy_loss)

    return logits, optimizer_train, x_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Tune parameters!
    # Init global variables
    sess.run(tf.global_variables_initializer())

    # Training
    for epoch in range(epochs):
        print('Epoch {epoch}:'.format(epoch=epoch))
        for images, label in get_batches_fn(batch_size):
            # train the network! (stuff kept in 'meh' are unnecessary)
            meh, loss = sess.run([train_op, cross_entropy_loss],
                                 feed_dict={input_image: images,
                                            correct_label: label,
                                            keep_prob: 0.5,
                                            learning_rate: 0.0012})
            print("Loss: {loss}".format(loss=loss))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    # TODO: Tweak parameters!
    epochs = 20 #6
    batch_size = 25 #100

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Build NN using the functions above: load_vgg, layers, and optimize function
        input_image, keep_probability, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        output_layer = layers(l3_out, l4_out, l7_out, 2)

        logits, optimizer_training, x_entropy_loss = optimize(output_layer,
                                                              correct_label=correct_label,
                                                              learning_rate=learning_rate,
                                                              num_classes=num_classes)

        # Train the fully-convolutional neural network
        train_nn(sess,
                 epochs=epochs,
                 batch_size=batch_size,
                 get_batches_fn=get_batches_fn,
                 train_op=optimizer_training,
                 cross_entropy_loss=x_entropy_loss,
                 input_image=input_image,
                 correct_label=correct_label,
                 keep_prob=keep_probability,
                 learning_rate=learning_rate
                 )

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_probability, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
