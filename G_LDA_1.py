import scipy.stats
import numpy
import re
from optparse import OptionParser

def test():
	#parser = OptionParse()
	#parse.add_option()
	T = 
	alpha = 
	K = 
	MU = 
	NU = 
	SIGMA = 
	D = 
	wordvec_file = ''
	corpus_file = ''
	iteration = 	

	glda = GLDA(T, alpha, K, NU, MU, SIGMA, D)
	glad.load_wordvec(wordvec_file)
	glad.load_corpus(corpus_file)
	glad.set_corpus()
	
	for i in range(iteration):
		glad.inference()
	
	phi = glad.doc_topic()
	g_mu, g_sigma = glad.gaussian_topic()




class GLDA:
	def __init__(self, T, alpha, K, NU, MU, SIGMA, D):
		#D : dimension of word vectors
		#T : number of topics, a scale
		#alpha: hyper-parameters of doc-topic prior drichlet dist.
		#student-t dist. parameters:
		#K, NU, MU, SIGMA
		self.D = D
		self.T = T
		self.alpha = alpha
		self.K = K
		self.NU = NU
		self.MU = MU
		self.SIGMA = SIGMA

	def load_corpus(self, filename):
		self.corpus = []
		input = open(filename, 'r')
		for line in input:
			doc = re.findall( ''' some regulation for tweets or text ''', line.lower())	
			if len(doc) > 0:
				self.corpus.append(doc)
		input.close()		


	def load_wordvec(self, filename):
		self.wordvec = {}
		#let assume the file looks like: word : 1,2,3,4,5,...,300
		input = open(filename, 'r')
		line = input.readline()
		while line != '':
			temp = line.split(':')
			word = temp[0]
			lst = temp[1].split(',')
			#convert string array to float array, store them in map
			self.wordvec[word] = numpy.array(map(float, lst))
			line = input.readline()
		input.close()

		


	def term_to_id(self, term):
		if term not in self.wordvec:
			return None
		if term not in self.vocas_id:
			voca_id = len(self.vocas_id)
			self.vocas_id[term] = voca_id
			self.re_vocas_id[voca_id] = term
			##self.vocas.append(term)# don't see the function of this sentence here
		else:
			voca_id = self.vocas_id[term]
		return voca_id
	
	def set_corpus(self): #more like initialize
		##self.vocas = []
		self.vocas_id = {}
		self.re_vocas_id = {}
		#self.docs is a 2-D array;
		#each row is a document;
		#a document is represented by a vecotor;
		#element in this vector is word id.
		#also get rid of the word not included in wordvec 
		temp = [[self.term_to_id(term) in doc] for doc in corpus]
		self.docs = [[word_id in doc if word_id is not None] for doc in temp]
		M = len(corpus)#number of documents
		##V = len(self.vocas)#number of terms, not words
		V = len(self.vocas_id)#number of terms, not words
		self.z_m_n = []#topic assignment of each word in docs
		#in a document, how many words are assigned to a topic
		self.n_m_z = numpy.zeros((M, self.T), dtype = int) #document topic dist.
		#in a topic, how many time a term is assgined to a topic
		self.n_z_t = numpy.zeros((self.T, V), dtype = int) #topic word dist.
		self.n_z = numpy.zeros(self.T, dtype = int) #words acount in each topic
		#m is m_th document, doc is the vectorized words of this document
		for m, doc in zip(range(M), self.docs):
			N_m = len(doc) #length of m_th document. doc is a N_m array, each element is a word id 
			z_n = [numpy.random.multinomial(1, ([1.0] * T) / T).argmax() for x in range(N_m)]
			self.z_m_n.append(z_n)
			#t is a word (word id is t), z is the topic this word is assigned to 
			for t, z in zip(doc, z_n):
				#number of words of topic z in m_th document plus one
				self.n_m_z[m, z] += 1
				#number of t_th term assigned to topic z plus one
				self.n_z_t[z, t] += 1
				#number of words assinged to topic z plus one 
				self.n_z[z] += 1


	#deal with self.n_z_t, take the average of each row (each topic)
	#for word_id, its word vector is word_vector[word_id]
	def gassian_likelihood(self, word_id):
		#step1 : compute average and variance
		#average : a self.T by self.D matrix, each element is the average value of a dimenison of word vector on a topic
		average = numpy.zeros((self.T, self.D), dtype = float)
		#variance: a self.T by self.D matrix, each element is the variance of a dimension of word vector on a topic
		variance = numpy.zeros((self.T, self.D), dtype = float)
		for t in range(self.T):
			#compute the average of each topic
			for v in range(len(self.vocas)):
				times = self.n_z_t[t][v]
				average[t] += self.wordvec[self.re_vocas_id[v]] * times #this word appears 'times' on this topic
			average[t] = average[t] / self.n_z[t]
			#compute the variance of each topic			
			for v in range(len(self.vocas)):
				times = self.n_z_t[t][v]
				variance[t] += (self.wordvec[self.re_vocas_id[v]] - average[t]) ** 2 * times
		#step2 : compute mu and sigma
		#compute mu
		#self.K is a D-dimensional vector
		#self.MU is a D-dimensional vector
		#average is a self.T by self.D vector
		mu = numpy.zeros((self.T, self.D), dtype = float)
		for t in range(self.T):
			mu[t] = (self.K * self.MU +  self.n_z[t] * average[t]) / (self.K + self.n_z[t])
		#compute sigma
		#self.NU is a D-dimensional vector
		#self.SIGMA is a D-dimensional vector
		sigma = numpy.zeros((self.T, self.D), dtype = float)	
		for t in range(self.T):
			sigma[t] = (self.NU * self.SIGMA + variance[t] + self.n_z[t] * self.K / (self.K + self.n_z[t]) * (self.MU - average[t]) ** 2) / (self.NU + self.n_z[t])
		#step3 : compute student-t distribution
		#we have mu, K by D; sigma, K by D
		#call scipy.stats.t.pdf to calculate the likelihood of each dimension of the word on each topic
		likelihood = numpy.zeros(self.T)
		for t in range(self.T):
			d_likelihood = []
			shape_p = self.NU + self.n_z[t]
			loc_p = mu[t]
			scale_p = sigma[t] ** 0.5
			#for each topic, there are D likelihoods
			for d in range(self.D):
				d_likelihood.append(scipy.stats.t.pdf(wordvec[self.re_vocas_id[word_id]][d], df = shape_p[d], loc = loc_p[d], scale = scale_p[d]))
			#step4 : compute the multiplication of all dimensions
			#multiplication indicates the independence
			likelihood[t] = numpy.prod(d_likelihood)
		return likelihood



	def inference(self):
		##V = len(self.vocas)
		V = len(self.vocas_id)
		#m is id of document, doc is the m_th document's words
		for m, doc in zip(range(len(self.docs)), self.docs):
			#iterate all the word in this document
			for n in range(len(doc)):
				#t is the id of current word
				t = doc[n]
				z = self.z_m_n[m][n]
				self.n_m_z[m, z] -= 1
				self.n_z_t[z, t] -= 1
				self.n_z[z] -= 1
				#doesn't really need a denom
				#gassian_likelihood is the only difference from original
				#we are considering make it two gassian later.
				p_z = gassian_likelihood(t) * (self.n_m_z[m] + self.alpha)
				new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
				self.z_m_n[m][n] = new_z
				self.n_m_z[m, new_z] += 1
				self.n_z_t[new_z, t] += 1
				self.n_z[new_z] += 1

	
	#topic distribution in each doc
	#return a M by T matrix
	#each row is the distribution each T topic is m_th document.
	def doc_topic(self):
		return (self.n_m_z + self.alpha) / (self.n_m_z.sum(axis = 1)[:, numpy.newaxis] + self.alpha)


	
	#T D-dimensional Gaussian distribution for T topics
	#calculation is the same to that of gaussian-likelihood
	def gaussian_topic(self):
		average = numpy.zeros((self.T, self.D), dtype = float)
		variance = numpy.zeros((self.T, self.D), dtype = float)
		for t in range(self.T):
			for v in range(len(self.vocas)):
				times = self.n_z_t[t][v]
				average[t] += self.wordvec[self.re_vocas_id[v]] * times #this word appears 'times' on this topic
			average[t] = average[t] / self.n_z[t]
			for v in range(len(self.vocas)):
				times = self.n_z_t[t][v]
				variance[t] += (self.wordvec[self.re_vocas_id[v]] - average[t]) ** 2 * times
		mu = numpy.zeros((self.T, self.D), dtype = float)
		for t in range(self.T):
			mu[t] = (self.K * self.MU +  self.n_z[t] * average[t]) / (self.K + self.n_z[t])
		sigma = numpy.zeros((self.T, self.D), dtype = float)	
		for t in range(self.T):
			#slight different in the denominator
			sigma[t] = (self.NU * self.SIGMA + variance[t] + self.n_z[t] * self.K / (self.K + self.n_z[t]) * (self.MU - average[t]) ** 2) / (self.NU + self.n_z[t] - 2) 
		#mu and sigma are both T by D matrix
		#preprent T topic
		#each topic are consist of D Gaussian
		return mu, sigma


			
