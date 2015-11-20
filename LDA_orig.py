import scipy.stats
import numpy
import re
from math import *
from optparse import OptionParser
from scipy.special import gammaln as gamln
from scipy.spatial.distance import cosine
from scipy.spatial.distance import *

'''
from LDA_orig import *

corpus_file = 'summary_doc'

glad = GLDA(10, 0, 0)
glad.load_corpus(corpus_file)
glad.set_corpus()

for i in range(30):
	print '================================'
        print i
        glad.inference()
	print glad.n_m_z
'''

class GLDA:
        def __init__(self, T, alpha, beta):
                self.T = T
                self.alpha = alpha
		self.beta = beta



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


        def set_corpus(self):
		self.vocas_id = {}
		self.re_vocas_id = {}
		self.docs = [[self.term_to_id(term) for term in doc] for doc in self.corpus]
                M = len(self.corpus)
                V = len(self.vocas_id)
                self.z_m_n = []
                self.n_m_z = numpy.zeros((M, self.T), dtype = float)
                self.n_z_t = numpy.zeros((self.T, V), dtype = float)
                self.n_z = numpy.zeros(self.T, dtype = float)
                for m, doc in zip(range(M), self.docs):
                        N_m = len(doc)
                        z_n = [numpy.random.multinomial(1, numpy.array([1.0] * self.T) / self.T).argmax() for x in range(N_m)]
                        self.z_m_n.append(z_n)
                        for t, z in zip(doc, z_n):
                                self.n_m_z[m, z] += 1
                                self.n_z_t[z, t] += 1
                                self.n_z[z] += 1


        def inference(self):
                V = len(self.vocas_id)
		M = len(self.corpus)
                for m, doc in zip(range(M), self.docs):
                        for n in range(len(doc)):
                                t = doc[n]
                                z = self.z_m_n[m][n]
                                self.n_m_z[m, z] -= 1
                                self.n_z_t[z, t] -= 1
                                self.n_z[z] -= 1
				denom_a = self.n_m_z[m].sum()
                                denom_b = self.n_z_t.sum(axis = 1)
				p_z = (self.n_z_t[:, t]) / denom_b * (self.n_m_z[m])/denom_a
				prob = p_z / p_z.sum()
				#print '=================='
				#print self.n_z_t[:, t]
				#print self.n_m_z
				#print '=================='
                                
				#print '=================='
				#print prob
				if numpy.isnan(prob[0]):
					#print 'n_z_t: ', self.n_z_t[:, t]
					#print 'n_m_z: ', self.n_m_z[m]
					prob =  numpy.array([1.0] * self.T) / self.T				
				new_z = numpy.random.multinomial(1, prob).argmax()
				#print new_z
                                self.z_m_n[m][n] = new_z
                                self.n_m_z[m, new_z] += 1
                                self.n_z_t[new_z, t] += 1
                                self.n_z[new_z] += 1
