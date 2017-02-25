import os
import cPickle as pickle
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2


from keras.layers.core import Lambda
import keras.backend as K

FN0 = 'vocabulary-embedding-baby'
FN1 = 'train'

maxlend = 25 # 0 - if we dont want to use description at all
maxlenh = 25
maxlen = maxlend + maxlenh
rnn_size = 256 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm=False
activation_rnn_size = 40 if maxlend else 0

empty = 0
eos = 1
batch_size=64
nflips=10

nb_train_samples = 4000
nb_val_samples = 1010

#import data
with open('data/%s.pkl'%FN0, 'rb') as fp:
    embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape


nb_unknown_words = 10

for i in range(nb_unknown_words):
	idx2word[vocab_size-1-i] = '<%d>'%i

oov0 = vocab_size-nb_unknown_words #outside vocab
#marking outside vocab
for i in range(oov0, len(idx2word)):
    idx2word[i] = idx2word[i]+'^'


seed = 34

def str_shape(x):
    return 'x'.join(map(str,x.shape))
    
def inspect_model(model):
    for i,l in enumerate(model.layers):
        print i, 'cls=%s name=%s'%(type(l).__name__, l.name)
        weights = l.get_weights()
        for weight in weights:
            print str_shape(weight),
        print


def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    
    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]
    
    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)




def main():
	FN0 = 'vocabulary-embedding-baby'
	FN1 = 'train'

	maxlend = 25 # 0 - if we dont want to use description at all
	maxlenh = 25
	maxlen = maxlend + maxlenh
	rnn_size = 256 # must be same as 160330-word-gen
	rnn_layers = 3  # match FN1
	batch_norm=False

	activation_rnn_size = 40 if maxlend else 0

	seed=42
	p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
	optimizer = 'adam'
	LR = 1e-4
	batch_size=64
	nflips=10

	nb_train_samples = 4000
	nb_val_samples = 1010	

	with open('data/%s.data.pkl'%FN0, 'rb') as fp:
		X, Y = pickle.load(fp)

	#splitting data
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
	
	del X
	del Y

	empty = 0
	eos = 1
	idx2word[empty] = '_'
	idx2word[eos] = '~'

	#model building
	random.seed(seed)
	np.random.seed(seed)
	regularizer = l2(weight_decay) if weight_decay else None

	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size,
	                    input_length=maxlen,
	                    W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,
	                    name='embedding_1'))
	for i in range(rnn_layers):
	    lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
	                W_regularizer=regularizer, U_regularizer=regularizer,
	                b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
	                name='lstm_%d'%(i+1)
	                  )
	    model.add(lstm)
	    model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))


	if activation_rnn_size:
	    model.add(SimpleContext())
	model.add(TimeDistributed(Dense(vocab_size,
	                                W_regularizer=regularizer, b_regularizer=regularizer,
	                                name = 'timedistributed_1')))
	model.add(Activation('softmax', name='activation_1'))

	from keras.optimizers import Adam, RMSprop # usually I prefer Adam but article used rmsprop
	# opt = Adam(lr=LR)  # keep calm and reduce learning rate
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)

	inspect_model(model)

	if FN1:
		model.load_weights('data/%s.hdf5'%FN1)

   	#training part

   	history = {}

   	traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
	valgen = gen(X_test, Y_test, nb_batches=nb_val_samples//batch_size, batch_size=batch_size)

	for iteration in range(1):
	    print 'Iteration', iteration
	    h = model.fit_generator(traingen, samples_per_epoch=nb_train_samples, nb_epoch=1)
	    for k,v in h.history.iteritems():
	        history[k] = history.get(k,[]) + v
	    with open('data/%s.history.pkl'%FN,'wb') as fp:
	        pickle.dump(history,fp,-1)
	    model.save_weights('data/%s.hdf5'%FN, overwrite=True)


	print "Bas ho gaya"


def lpadd(x, maxlend=maxlend, eos=eos):
    """left (pre) pad a description to maxlend and then add eos.
    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty]*(maxlend-n) + x + [eos]

def vocab_fold(xs):
    """convert list of word indexes that may contain words outside vocab_size to words inside.
    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < oov0 else glove_idx2idx.get(x,x) for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= oov0])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
    xs = [outside.get(x,x) for x in xs]
    return xs

def flip_headline(x, nflips=None, model=None, debug=False):
    """given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
    with words predicted by the model
    """
    if nflips is None or model is None or nflips <= 0:
        return x
    
    batch_size = len(x)
    assert np.all(x[:,maxlend] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(xrange(maxlend+1,maxlen), nflips))
        if debug and b < debug:
            print b,
        for input_idx in flips:
            if x[b,input_idx] == empty or x[b,input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend+1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            if debug and b < debug:
                print '%s => %s'%(idx2word[x_out[b,input_idx]],idx2word[w]),
            x_out[b,input_idx] = w
        if debug and b < debug:
            print
    return x_out


def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)
    
    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh) + [eos] + [empty]*maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i,:,:] = np_utils.to_categorical(xh, vocab_size)
        
    return x, y


def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    """yield batches. for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches
    
    while training it is good idea to flip once in a while the values of the headlines from the
    value taken from Xh to value generated by the model.
    """
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, sys.maxint)
        random.seed(c+123456789+seed)
        for b in range(batch_size):
            t = random.randint(0,len(Xd)-1)

            xd = Xd[t]
            s = random.randint(min(maxlend,len(xd)), max(maxlend,len(xd)))
            xds.append(xd[:s])
            
            xh = Xh[t]
            s = random.randint(min(maxlenh,len(xh)), max(maxlenh,len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c+= 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)


def test_gen(gen, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        assert Xtr[i,maxlend] == eos
        x = Xtr[i,:maxlend]
        y = Xtr[i,maxlend:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',yy)
        prt('H',y)
        if maxlend:
            prt('D',x)

def str_shape(x):
    return 'x'.join(map(str,x.shape))
    
def inspect_model(model):
    for i,l in enumerate(model.layers):
        print i, 'cls=%s name=%s'%(type(l).__name__, l.name)
        weights = l.get_weights()
        for weight in weights:
            print str_shape(weight),
        print


if __name__ == "__main__":
	main()