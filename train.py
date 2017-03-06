#! ~/usr/bin/env python
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
from data import yelp, generate
#reproduction of HAN 

tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir",'data/yelp-2013', 'directory containing train, val and test h5')
flags.DEFINE_float('checkpoint_dir','checkpoint','directory to save the best model saved as checkpoint_dir/model.chkpt')

flags.DEFINE_integer('epoch',2,'epoch : default 2s')
flags.DEFINE_integer('batchsize',100,'batchsize: default 100')

#hyper params
flags.DEFINE_float('lr',1e-3,'Learning rate : default 1e-3')


class Attention():
	def __init__(self, input, mask, scope ='A0'):
		assert input.get_shape().as_list()[:-1] == mask.get_shape().as_list() and len(mask.get_shape().as_list()) == 2
		_, steps, embed_dim = input.get_shape().as_list()
		#trainable variales
		self.u_w = tf.Variable(tf.truncated_normal([1, embed_dim], stddev=0.1),  name='%s_query' %scope, dtype=tf.float32)
        weights = tf.Variable(tf.truncated_normal([embed_dim, embed_dim], stddev=0.1),  name='%s_Weight' %scope, dtype=tf.float32)
        bias = tf.Variable(tf.truncated_normal([1, embed_dim], stddev=0.1),  name='%s_bias' %scope, dtype=tf.float32)

        #equations
        u_i = tf.tanh(tf.matmul(tf.reshape(input,[-1,embed_dim]), weights) + bias)
        u_i = tf.reshape(u_i, [-1,steps, embed_dim])
        distances = tf.reduce_sum(tf.mul(u_i, self.u_w), reduction_indices=-1)
        self.debug = distances
        self.distances = distances -tf.expand_dims(tf.reduce_max(distances),-1) #avoid exp overflow
        expdistance = tf.mul(tf.exp(self.distances), mask) #
        Denom = tf.expand_dims(tf.reduce_sum(expdistance, reduction_indices=1), 1) + kwargs.get('eps', 1e-13) #avoid 0/0 error
        self.Attn = expdistance/Denom
        
        return


class HAN():
	def __init__(self, x, mask, **kwargs):
		_, doclen, sentlen, embed_dim = x.get_shape().as_list()

		xnew = tf.reshape(x,[-1, sentlen, embed_dim]) #example_sentences, steps, embedding
		masknew = tf.reshape(mask, [-1,sentlen]) #wordmask

		xnew = tf.unpack(xnew, axis=1)
		cell_fw = tf.nn.rnn_cell.GRUCell(kwargs.get('hidden_dim',100), scope='Word_Layer_fw')
		cell_bw = tf.nn.rnn_cell.GRUCell(kwargs.get('hidden_dim',100), scope='Word_layer_bw')
		output = tf.nn.bidirectional_rnn(cell_fw, cell_bw, xnew, scope='L0')
		output = tf.pack(output, axis=1)
		out_dim = output.get_shape().as_list[-1]

		self.A0 = Attention(output, masknew, scope='A0')
		output = tf.reduce_sum(input*tf.expand_dims(self.A0.Attn,-1) , reduction_indices=1) #sum_j Attn[i][j]*Word_embed[i][j][:]

		masknew = tf.cast(tf.reduction_indices(mask, reduction_indices= -1)>0,tf.int32) #sentence mask

		output = tf.unpack(self.A0.output, axis=1)
		cell_fw = tf.nn.rnn_cell.GRUCell(kwargs.get('hidden_dim',100), scope='Sentece_Layer_fw')
		cell_bw = tf.nn.rnn_cell.GRUCell(kwargs.get('hidden_dim',100), scope='Sentece_Layer_bw')
		output = tf.nn.bidirectional_rnn(cell_fw, cell_bw, output, scope='Sentece_Layer_output')
		output = tf.pack(output, axis=1)
		out_dim = output.get_shape().as_list[-1]

		self.A1 = Attention(output, masknew, scope='A1')		
		self.output = tf.reduce_sum(input*tf.expand_dims(self.A1.Attn,-1) , reduction_indices=1) #sum_j Attn[i][j]*Senten_embed[i][j][:]
	
		return 
		

################## STANDARD
D = h5.File('%s/train.h5' %FLAGS.data_dir)
pretrained_embedding_matrix = np.load('%s/embed.npy' %FLAGS.data_dir)
#############################

INPUT_SHAPE = [None,]+ D['x'].shape[1:]
OUTPUT_SHAPE = [None, D['y'].shape[1]]
WE_SHAPE = pretrained_embedding_matrix.shape


x  = tf.placeholder(INPUT_SHAPE, dtype=tf.int32)
y  = tf.placeholder(OUTPUT_SHAPE, dtype=tf.int32)
mask  = tf.placeholder(INPUT_SHAPE, dtype=tf.int32)
pretrained_we = tf.placeholder(WE_SHAPE, dtype=tf.int32)

WELayer = tf.Variable(WE_SHAPE, dtype=tf.int32)
embedding_init = WELayer.assign(pretrained_we)

_, doclen,sentlen =  x.get_shape().as_list()
xnew = tf.reshape(x, [-1,sentlen])
WE = tf.nn.embedding_lookup(WELayer, xnew)

############Document Model#################
H = HAN(tf.reshape(WE, [-1, doclen, sentlen, DATA.embed_dim]))

#######Classifier################
output = tf.contrib.layers.fully_connected(H.output, FLAGS.pre_output_size, activation_fn = tf.tanh, scope='fc0')
output = tf.contrib.layers.fully_connected(output ,DATA.NUM_CLASSES, scope='fc1', activation_fn=None)


assert y.get_shape().as_list() == output.get_shape().as_list()

########Loss##################
log_softmax_output = tf.log(tf.nn.softmax(output)+1e-13) #log softmax 1e-13 for stability
loss = -DATA.NUM_CLASSES*tf.reduce_mean(tf.mul(log_softmax_output, t.cast(y,tf.float32))) #log aka cross entropy (close) aka logistic loss
global_step = tf.Variable(0, trainable=False)
train_op = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate=FLAGS.lr, optimizer='Adam')

#Metrics
accuracy  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1) ,tf.argmax(y, 1)), tf.float32))

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)
sess.run(embedding_init, feed_dict = {pretrained_we:pretrained_embedding_matrix})

saver = tf.train.Saver()

start = time.time()
for batch_x, batch_y, batch_mask in generate('%s/train.h5' %FLAGS.data_dir, FLAGS.epoch, FLAGS.batchsize, small=True):
	l,a,g,_ = sess.run([loss, acc, global_step, train_op], feed_dict = {x: batch_x, y: batch_y, mask: batch_mask})
	print('Train Iterations: %d , loss %.3f accuracy %.3f' %(global_step, l, a))
	if g%100 == 0:
		print ('Time taken for 100 iterations %.3f' %(time.time()-start))
		avg_loss, avg_acc, examples = 0.0, 0.0, 0.0
		for val_x, val_y, val_mask in generate('%s/val.h5' %FLAGS.data_dir,1, 128):
			l, a = sess.run([loss, accuracy], feed_dict = {x:val_x, y:val_y, mask: val_mask})
			avg_loss +=l*val_y.shape[0]
			avg_acc +=l*val_y.shape[0]
			examples += val_y.shape[0]
		print('Val loss %.3f accuracy %.3f' %(avg_loss//examples, avg_acc//examples))
		val = avg_acc//examples
		if best_val < val: 			
			best_val = val
			print('Got a best validation score, Saving Model...')			
			save_path = saver.save(sess, "%s/model.ckpt" %FLAGS.checkpoint_dir)
			print('Model Saved @ %s' %save_path)

		print('Best val accuracy %.3f' %best_val)
		start = time.time()

sess.close()
