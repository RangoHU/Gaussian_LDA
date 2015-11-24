import scipy.stats
import numpy
import re
from math import *
from optparse import OptionParser
from scipy.special import gammaln as gamln
from scipy.spatial.distance import cosine
from scipy.spatial.distance import *
import warnings
import datetime
'''


from LDA_orig import *
corpus_file = 'summary_doc'
#corpus_file = 'test_corpus_3'
#corpus_file = 'test_corpus'

glad = GLDA(10, 0.1, 0)
glad.load_corpus(corpus_file)
glad.word2id(n = 10)
glad.set_corpus()

for i in range(10):
	print datetime.datetime.now()
	#print '================================'
        #print i
        glad.inference()
	print glad.n_m_z


'''

class GLDA:
        def __init__(self, T, alpha, beta):
                self.T = T
                self.alpha = alpha
		self.beta = beta
		self.prob = numpy.array([1.0] * self.T) / self.T
		self.docs = []
		warnings.filterwarnings('error')

	def load_corpus(self, filename):
		self.corpus = []
		input = open(filename, 'r')
		for line in input:
			line = line.strip().lower()
			doc = line.split()
			if len(doc) > 0:
				self.corpus.append(doc)
		input.close()		


	def term_to_id(self, term):
		if term not in self.vocas_id:
			v_id = len(self.vocas_id)
			self.vocas_id[term] = v_id
			self.re_vocas_id[v_id] = term
		else:
			v_id = self.vocas_id[term]
		return v_id


	def word2id(self, n = 5):
		self.temp = {}
		self.vocas_id = {}
		self.re_vocas_id = {}
		for doc in self.corpus:
			for word in doc:
				if word not in self.temp:
					self.temp[word] = 1
				else:
					self.temp[word] += 1
		for each in self.temp:
			if self.temp[each] > n:
				word = each
				index = len(self.vocas_id)
				self.vocas_id[word] = index
				self.re_vocas_id[index] = word	
	


        def set_corpus(self):
		#self.vocas_id = {}
		#self.re_vocas_id = {}
		#self.docs = [[self.term_to_id(term) for term in doc] for doc in self.corpus]
		temp = [[self.vocas_id[term] for term in doc if term in self.vocas_id] for doc in self.corpus]
		for each in temp:
			if len(each) > 2:
				self.docs.append(each)
	
                M = len(self.docs)
                V = len(self.vocas_id)
                self.z_m_n = []
                self.n_m_z = numpy.zeros((M, self.T), dtype = float)
		self.n_z_t = numpy.zeros((self.T, V), dtype = float)
                self.n_z = numpy.zeros(self.T, dtype = float)
                for m, doc in zip(range(M), self.docs):
                        N_m = len(doc)
			if N_m < 2:
				continue
                        z_n = [numpy.random.multinomial(1, numpy.array([1.0] * self.T) / self.T).argmax() for x in range(N_m)]
                        self.z_m_n.append(z_n)
			temp = [0.0] * self.T
                        for t, z in zip(doc, z_n):
                                self.n_m_z[m, z] += 1
                                self.n_z_t[z, t] += 1
                                self.n_z[z] += 1
		self.n_z_beta = self.n_z + V * self.beta


        def inference(self):
                V = len(self.vocas_id)
		M = len(self.n_m_z)
                for m, doc in zip(range(M), self.docs):
                        for n in range(len(doc)):
                                t = doc[n]
                                z = self.z_m_n[m][n]
                                self.n_m_z[m, z] -= 1
                                self.n_z_t[z, t] -= 1
                                #self.n_z[z] -= 1
				self.n_z_beta[z] -= 1

				##########################################
				p_z = (self.n_z_t[:, t]) /(self.n_z_beta)* (self.n_m_z[m])
				try:
					prob = p_z / p_z.sum()
				except Warning:
					print p_z
					print self.n_z_t[:, t]
					print self.n_z_beta
					print self.n_m_z[m]
					print 't = ', t
					print 'm = ', m
					#continue
					prob = self.prob
					return self.n_z_t[:, t], self.n_z_beta, self.n_m_z[m]
				#print '##########################################'
				#print p_z
				#print '##########################################'
				##########################################
		
				new_z = numpy.random.multinomial(1, prob).argmax()
                                self.z_m_n[m][n] = new_z
                                self.n_m_z[m, new_z] += 1
                                self.n_z_t[new_z, t] += 1
                                #self.n_z[new_z] += 1
				self.n_z_beta[new_z] += 1


	def phi(self):
		V = len(self.vocas_id)
		return (self.n_z_t + self.beta) / self.n_z_beta[:, numpy.newaxis] #(self.n_z[:, numpy.newaxis] + V * self.beta)

	def theta(self):
		return (self.n_m_z + self.alpha) / (self.n_m_z.sum(axis = 1)[:, numpy.newaxis] + self.T * self.alpha)



	def display_topic(self, n = 10):
		dist = self.phi()
		for i in range(self.T):
			s = dist[i]
			index = sorted(range(len(s)), key=lambda k: s[k])
			string = ''
			for j in range(n):
				string += self.re_vocas_id[index[j]] + '  '
				#string += self.re_vocas_id[index[j]] + '*' + str(s[index[j]]) + '  '
			print string
	














