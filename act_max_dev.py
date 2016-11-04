#!/usr/bin/env python
'''
Anh Nguyen <anh.ng8@gmail.com>
2016-06-04
Customized by Jiri Roznovjak <jiri.roznovjak@gmail.com>
'''
import os
os.environ['GLOG_minloglevel'] = '2'  # suprress Caffe verbose prints

import settings
import site
site.addsitedir(settings.caffe_root)
import caffe

import numpy as np
import math, random
import sys, subprocess
from IPython.display import clear_output, Image, display
from scipy.misc import imresize
from numpy.linalg import norm
from numpy.testing import assert_array_equal
import scipy.misc, scipy.io
import patchShow
import argparse # parsing arguments

mean = np.float32([104.0, 117.0, 123.0])

fc_layers = ["fc6", "fc7", "fc8", "loss3/classifier", "fc1000", "prob", "fc8_oxford_102"]
conv_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

if settings.gpu:
    caffe.set_mode_gpu() # uncomment this if gpu processing is available
    caffe.set_device(3)

def main():

    args = parseArguments()

    # Default to constant learning rate
    if args.end_lr < 0:
        args.end_lr = args.start_lr

    # which neuron to visualize
    print "-------------"
    print " objective: %s" % (args.obj)
    print " n_iters: %s" % args.n_iters
    print " L2: %s" % args.lambd
    print " start learning rate: %s" % args.start_lr
    print " end learning rate: %s" % args.end_lr
    print " seed: %s" % args.seed
    print " opt_layer: %s" % args.opt_layer
    print " act_layer: %s" % args.act_layer
    print " init_file: %s" % args.init_file
    print " clip: %s" % args.clip
    print " bound: %s" % args.bound
    print "-------------"
    print " output dir: %s" % args.output_dir
    print " net weights: %s" % args.net_weights
    print " net definition: %s" % args.net_definition
    print "-------------"

    params = {
    'layer': args.act_layer,
    'iter_n': args.n_iters,
    'L2': args.lambd,
    'gamma':args.gamma,
    'start_step_size': args.start_lr,
    'end_step_size': args.end_lr
    }

    # networks
    generator = caffe.Net(settings.generator_definition, settings.generator_weights, caffe.TEST)
    net = caffe.Classifier(args.net_definition, args.net_weights,
                             mean = mean, # ImageNet mean
                             channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    batch_size = 1
    size = net.blobs["data"].data.shape[-1]
    image_size = (3, size, size)
    input_size = (batch_size, ) + image_size
    net.blobs["data"].reshape(*input_size)

    # input / output layers in generator
    #gen_in_layer = "defc7"
    gen_in_layer = "feat"
    gen_out_layer = "deconv0"

    # shape of the code being optimized
    shape = generator.blobs[gen_in_layer].data.shape

    # Fix the seed
    np.random.seed(args.seed)

    if args.init_file != "None":
        start_code, start_image = get_code(args.init_file, args.opt_layer)
    else:
        start_code = np.random.normal(0, 1, shape)

    # Load the activation range
    upper_bound = lower_bound = None

    # Set up clipping bounds
    if args.bound != "":
        n_units = shape[1]
        upper_bound = np.loadtxt(args.bound, delimiter=' ', usecols=np.arange(0, n_units), unpack=True)
        upper_bound = upper_bound.reshape(start_code.shape)

        # Lower bound of 0 due to ReLU
        lower_bound = np.zeros(start_code.shape)

    #compute objective
    if args.mode == "unit":
        objective = int(args.obj)
    elif args.mode == "custom":
        objective = computeObjective(args.obj, args.act_layer, net=net)
    elif args.mode == "multiple":
        objective = [int(o) for o in args.obj.split(",")]

    feat_layer = "conv3"
    img = "data/tabby.jpg"
    feat_objective, _ = get_code(img, feat_layer, net=net)
    feat_objective = {feat_layer: feat_objective}

    # Optimize a code via gradient ascent
    output_image = activation_maximization(net, generator, gen_in_layer, gen_out_layer, start_code, params, 
        clip=args.clip, xy=args.xy, 
        upper_bound=upper_bound, lower_bound=lower_bound, objective=objective
        feat_objective=feat_objective)

    filename = "example.jpg"
    # Save image
    save_image(output_image, filename)
    print "Saved to %s" % filename


def parseArguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--obj', metavar='obj', type=str, default="", help='an objective (a filename or a class)')
    parser.add_argument('--mode', metavar='mode', type=str, default='class', help='Mode (unit, multiple, interpolate, custom)')
    parser.add_argument('--n_iters', metavar='iter', type=int, default=10, help='Number of iterations')
    parser.add_argument('--lambd', metavar='lambda', type=float, default=1.0, nargs='?', help='Regularization parameter')
    parser.add_argument('--gamma', metavar='gamma', type=float, default=1.0, nargs='?', help='Distance between neurons regularization')
    parser.add_argument('--start_lr', metavar='lr', type=float, default=2.0, nargs='?', help='Learning rate')
    parser.add_argument('--end_lr', metavar='lr', type=float, default=-1.0, nargs='?', help='Ending Learning rate')
    parser.add_argument('--seed', metavar='n', type=int, default=0, nargs='?', help='Learning rate')
    parser.add_argument('--xy', metavar='n', type=int, default=0, nargs='?', help='Spatial position for conv units')
    parser.add_argument('--opt_layer', metavar='s', type=str, help='Layer at which we optimize a code')
    parser.add_argument('--act_layer', metavar='s', type=str, default="fc8", help='Layer at which we activate a neuron')
    parser.add_argument('--init_file', metavar='s', type=str, default="None", help='Init image')
    parser.add_argument('--clip', metavar='b', type=int, default=0, help='Clip out within a code range')
    parser.add_argument('--bound', metavar='b', type=str, default="", help='The file to an array that is the upper bound for activation range')
    parser.add_argument('--output_dir', metavar='b', type=str, default=".", help='Output directory for saving results')
    parser.add_argument('--net_weights', metavar='b', type=str, default=settings.net_weights, help='Weights of the net being visualized')
    parser.add_argument('--net_definition', metavar='b', type=str, default=settings.net_definition, help='Definition of the net being visualized')

    args = parser.parse_args()
    return args

        

def computeObjective(obj, act_layer, net=None):

    objective, _ = get_code(obj, act_layer, net=net)
    return objective

    #custom objective here

    #objective = get_average(["cat.jpg"], act_layer, net)

    #Image arithmetic
    #base_img, _ = get_code("data/beer_glass.jpg", act_layer, net=net)
    #img_subtract, _ = get_code("data/no_beer_glass.jpg", act_layer, net=net)
    #img_onto, _ = get_code("data/no_beer_glass2.jpg", act_layer, net=net)
    #diff = base_img - img_subtract
    #objective = diff

    #Image intersection
    img1, _ = get_code("data/black_cat.jpg", act_layer, net=net)
    img2, _ = get_code("data/yellow_cat.jpg", act_layer, net=net)
    sub = np.abs(img1 - img2)
    meandiff = np.mean(sub)
    mean = (img1 + img2) / 2
    #Threshold everything above mean difference
    mask = (sub < meandiff * 1.5).astype(int)
    objective = mask * mean
    return objective

def get_average(imgs, act, net):
    reprs = [get_code("data/" + im, act, net=net)[0] for im in imgs]
    return np.mean(reprs, axis=0)


def activation_maximization(net, generator, gen_in_layer, gen_out_layer, start_code, params, 
        clip=False, xy=0, upper_bound=None, lower_bound=None, objective=None, output="image", feat_objective=None):

    # Get the input and output sizes
    data_shape = net.blobs['data'].data.shape
    generator_output_shape = generator.blobs[gen_out_layer].data.shape

    # Calculate the difference between the input image to the net being visualized
    # and the output image from the generator
    image_size = get_shape(data_shape)
    output_size = get_shape(generator_output_shape)

    # The top left offset that we start cropping the output image to get the 227x227 image
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)

    print "Starting optimizing"

    x = None
    src = generator.blobs[gen_in_layer]

    # Make sure the layer size and initial vector size match
    assert_array_equal(src.data.shape, start_code.shape)

    # Take the starting code as the input to the generator
    src.data[:] = start_code.copy()[:]

    # Initialize an empty result
    best_xx = np.zeros(image_size)[np.newaxis]
    best_act = -sys.maxint

    # Save the activation of each image generated
    list_acts = []

    # select layer
    layer = params['layer']

    for i in xrange(params['iter_n']):

        step_size = params['start_step_size'] + ((params['end_step_size'] - params['start_step_size']) * i) / params['iter_n']
        
        # 1. pass the code to generator to get an image x0
        generated = generator.forward(feat=src.data[:])
        x0 = generated[gen_out_layer]   # 256x256

        # Crop from 256x256 to 227x227
        cropped_x0 = x0.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

        # 2. forward pass the image x0 to net to maximize an unit k
        # 3. backprop the gradient from net to the image to get an updated image x
        grad_norm_net, x, act, reg_penalty = make_step_net(net=net, end=layer, objective=objective, image=cropped_x0,
                                                            xy=xy, step_size=step_size, output=i % 5 == 0, gamma=params['gamma'],
                                                            feat_objective=feat_objective)
        
        # Save the solution
        # Note that we're not saving the solutions with the highest activations
        # Because there is no correlation between activation and recognizability
        best_xx = cropped_x0.copy()
        best_act = act

        # 4. Place the changes in x (227x227) back to x0 (256x256)
        updated_x0 = x0.copy()        
        updated_x0[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]] = x.copy()

        # 5. backprop the image to generator to get an updated code
        grad_norm_generator, updated_code = make_step_generator(net=generator, x=updated_x0, x0=x0, 
                start=gen_in_layer, end=gen_out_layer, step_size=step_size)

        # Clipping code
        if clip:
            updated_code = np.clip(updated_code, a_min=-1, a_max=1) # VAE prior is within N(0,1)

            # Clipping each neuron independently
        elif upper_bound is not None:
            updated_code = np.maximum(updated_code, lower_bound) 
            updated_code = np.minimum(updated_code, upper_bound) 

        # L2 on code to make the feature vector smaller every iteration
        if params['L2'] > 0 and params['L2'] < 1:
            updated_code[:] *= params['L2']
        if reg_penalty != None:
            updated_code[:] -= reg_penalty

        # Update code
        src.data[:] = updated_code

        #if i % 5 == 0:
        #    name = "debug/%s.jpg" % str(i).zfill(3)
        #    save_image(x.copy(), name)

        # Stop if grad is 0
        if grad_norm_generator == 0:
            print " grad_norm_generator is 0"
            break
        elif grad_norm_net == 0:
            print " grad_norm_net is 0"
            break

    # returning the resulting image
    print " -------------------------"
    print " Result: obj act [%s] " % best_act

    if output == "image":
        return best_xx
    elif output == "code":
        return src.data[:].copy()


'''
Push the given image through an encoder to get a code.
'''
def get_code(path, layer, net=None):

    # initialize the encoder
    if net == None:
        encoder = caffe.Net(settings.encoder_definition, settings.encoder_weights, caffe.TEST)
    else:
        encoder = net

    # set up the inputs for the net: 
    batch_size = 1
    size = encoder.blobs["data"].data.shape[-1]
    image_size = (3, size, size)

    input_size = (batch_size, ) + image_size
    encoder.blobs["data"].reshape(*input_size)

    images = np.zeros((batch_size,) + image_size, dtype='float32')

    in_image = scipy.misc.imread(path)
    in_image = scipy.misc.imresize(in_image, (image_size[1], image_size[2]))

    for ni in range(images.shape[0]):
        images[ni] = np.transpose(in_image, (2, 0, 1))

    # Convert from RGB to BGR
    data = images[:,::-1] 

    # subtract the ImageNet mean
    matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
    image_mean = matfile['image_mean']
    topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
    image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
    del matfile
    data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

    # run encoder and extract the features
    encoder.forward(data=data)
    feat = np.copy(encoder.blobs[layer].data)
    del encoder

    zero_feat = feat[0].copy()[np.newaxis]

    return zero_feat, data

'''
Forward and backward passes through the DNN being visualized.
'''
def make_step_net(net, end, objective, image, xy=0, step_size=1, output=True, gamma=1.0, feat_objective=None, featContr=1.0):

    src = net.blobs['data'] # input image
    dst = net.blobs[end]

    acts = net.forward(data=image, end=end)

    updated_diff = np.zeros_like(dst.data)
    
    if type(objective) is np.ndarray:
        #Move in the direction of the objective
        updated_diff = objective - dst.diff
    else:
        # Move in the direction of increasing activation of the given neuron
        if end in fc_layers:
            if type(objective) is list:
                activations = [acts[end][0][obj] for obj in objective]
                mean_acts = np.mean(activations)
                reg_penalty = gamma * np.sum(np.square(activations - mean_acts))
                for obj in objective:
                    updated_diff.flat[obj] = 1.
                for i in range(len(objective)):
                    di = -2 * gamma * (activations[i] - mean_acts) * (1 - 1.0/len(activations))
                    updated_diff.flat[objective[i]] += di
            else:
                updated_diff.flat[objective] = 1.
        elif end in conv_layers:
            updated_diff[:, objective, xy, xy] = 1.
        else:
            raise Exception("Invalid layer type!")
                
    dst.diff[:] = updated_diff

    loss = 0.0

    if feat_objective != None:
        feat_layers = feat_objective.keys()
        #Only one layer for now
        l = feat_layers[0]
        F = net.blobs[l].data[0]

        featLoss, featGrad = compFeatureGrad(F, feat_objective[l])
        loss += featLoss * featContr

        net.backward(start=end, end=l)

        diff = net.blobs[l].diff[0]
        diff += featGrad.reshape(diff.shape) * featContr

    new_end = end
    if feat_objective != None:
        new_end = feat_objective.keys()[0]

    # Get back the gradient at the optimization layer
    diffs = net.backward(start=new_end, diffs=['data'])
    g = diffs['data'][0]

    grad_norm = norm(g)
    obj_act = 0
    # reset objective after each step
    dst.diff.fill(0.)

    reg_penalty = None
    # If grad norm is Nan, skip updating
    if math.isnan(grad_norm):
        return 1e-12, src.data[:].copy(), obj_act, reg_penalty
    elif grad_norm == 0:
        return 0, src.data[:].copy(), obj_act, reg_penalty

    # Check the activations
    if type(objective) is np.ndarray:
        best_unit = 0
        obj_act = 0
    elif type(objective) is list:
        fc = acts[end][0]
        best_unit = fc.argmax()
    elif end in fc_layers:
        fc = acts[end][0]
        best_unit = fc.argmax()
        obj_act = fc[objective]
    elif end in conv_layers:
        fc = acts[end][0, :, xy, xy]
        best_unit = fc.argmax()
        obj_act = fc[objective]

    if output:
        if type(objective) is np.ndarray:
            diff_norm = norm(updated_diff)
            print "Diff norm: %.2f, opt norm: %.2f" % (diff_norm, grad_norm)
        elif type(objective) is list:
            out = "max: %4s [%.2f] " % (best_unit, fc[best_unit])
            for i in range(len(objective)):
                unit = objective[i]
                act = activations[i]
                out = out + " %d: %.2f " % (unit, act)
            print out
        else:
            print "max: %4s [%.2f]\t obj: %4s [%.2f]\t norm: [%.2f]" % (best_unit, fc[best_unit], objective, obj_act, grad_norm)

    # Make an update
    src.data[:] += step_size/np.abs(g).mean() * g

    return (grad_norm, src.data[:].copy(), obj_act, reg_penalty)


'''
Forward and backward passes through the generator DNN.
'''
def make_step_generator(net, x, x0, start, end, step_size=1):

    src = net.blobs[start] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    # L2 distance between init and target vector
    net.blobs[end].diff[...] = (x-x0)
    net.backward(start=end)
    g = net.blobs[start].diff.copy()

    grad_norm = norm(g)

    # reset objective after each step
    dst.diff.fill(0.)

    # If norm is Nan, skip updating the image
    if math.isnan(grad_norm):
        return 1e-12, src.data[:].copy()  
    elif grad_norm == 0:
        return 0, src.data[:].copy()

    # Make an update
    src.data[:] += step_size/np.abs(g).mean() * g

    return grad_norm, src.data[:].copy()

def compFeatureGrad(F, F_guide):
    E = F - F_guide
    loss = np.sum(np.square(E)) / 2
    grad = E * (F > 0)
    return loss, grad

def get_shape(data_shape):

    # Return (227, 227) from (1, 3, 227, 227) tensor
    if len(data_shape) == 4:
        return (data_shape[2], data_shape[3])
    else:
        raise Exception("Data shape invalid.")


'''
Normalize and save the image.
'''
def save_image(img, name):

    img = img[:,::-1, :, :] # Convert from BGR to RGB
    normalized_img = patchShow.patchShow_single(img, in_range=(-120,120))        
    scipy.misc.imsave(name, normalized_img)


def write_label(filename, act):
    # Add activation below each image via ImageMagick
    subprocess.call(["convert %s -gravity south -splice 0x10 %s" % (filename, filename)], shell=True)
    subprocess.call(["convert %s -append -gravity Center -pointsize %s label:\"%.2f\" -bordercolor white -border 0x0 -append %s" %
         (filename, 30, act, filename)], shell=True)


if __name__ == '__main__':
    main()
