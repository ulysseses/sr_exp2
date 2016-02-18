import sys
import time
import yaml
import run_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tower_name', 'tower',
    """If a model is trained with multiple GPU's prefix all Op names with """
    """tower_name to differentiate the operations. Note that this prefix """
    """is removed from the names of all summaries when visualizing a model.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
    "Whether to log device placement.")
tf.app.flags.DEFINE_integer('num_gpus', 4, "How many GPUs to use.")
tf.app.flags.DEFINE_boolean('dev_assign', True, "Do assign tf.devices.")


def lcod():
    template_conf_path = "template.yaml"
    with open(template_conf_path, 'r') as f:
        conf = yaml.load(f)
    
    Ts = [4]
    Cs = [128]
    Ks = [28]
    budget = 128 * 4 * 16
    grid = [(T, C, K) for T in Ts for C in Cs for K in Ks if K < C]
    flag = False
    for T, C, K in grid:
        print("T: %d C: %03d K: %02d" % (T, C, K))
        time.sleep(2)
        conf['T'] = T
        conf['n_c'] = C
        conf['e_rank'] = K
        conf['mb_size'] = budget / (T * C)
        conf['path_tmp'] = 'tmp/%03d_%03d_%03d' % (T, C, K)
        run_model.train(conf, False)
        psnr, bl_psnr = run_model.eval_te(conf)
        with open('notes/log.txt', 'a') as f:
            f.write('T: %03d C: %03d K: %03d PSNR: %.2f (%.2f)\n' % \
                    (T, C, K, psnr, bl_psnr))


def lista():
    template_conf_path = "template.yaml"
    with open(template_conf_path, 'r') as f:
        conf = yaml.load(f)
    conf['subnet_name'] = 'lista'
    
    Ts = [1]
    Cs = [128]
    Ks = [0]
    budget = 128 * 4 * 48
    grid = [(T, C, K) for T in Ts for C in Cs for K in Ks if K < C]
    flag = False
    for T, C, K in grid:
        conf['T'] = T
        conf['n_c'] = C
        conf['e_rank'] = K
        conf['s_rank'] = C // 4
        conf['mb_size'] = budget / (T * C)
        conf['path_tmp'] = 'tmp/lista'
        run_model.train(conf)
        psnr, bl_psnr = run_model.eval_te(conf)
        with open('notes/log_lista.txt', 'a') as f:
            f.write('T: %03d C: %03d K: %03d PSNR: %.2f (%.2f)\n' % \
                    (T, C, K, psnr, bl_psnr))


if __name__ == '__main__':
    lcod()
    #lista()
