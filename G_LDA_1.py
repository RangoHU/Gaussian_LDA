import scipy.stats
import numpy
import re
import json
from math import *
from optparse import OptionParser
from scipy.special import gammaln as gamln
from scipy.spatial.distance import cosine
from scipy.spatial.distance import *
import datetime


'''
from G_LDA_1 import *
T = 10
D = 300
alpha = 0.001 #numpy.array([0.001] * T) 
K = 1.0
MU = 0.0 #numpy.array([0.0] * D)
NU = 1.0
SIGMA = 0.1 #numpy.array([0.1] * D)

wordvec_file = 'toy_word2vec'
corpus_file = 'summary_doc'
glad = GLDA(T, alpha, K, NU, MU, SIGMA, D)
glad.load_wordvec(wordvec_file)
glad.load_corpus(corpus_file)
glad.set_corpus()



for i in range(20):
	print datetime.datetime.now()
	print i
	glad.inference()


print glad.n_m_z

mu = glad.mu[8]

lst = []
for i in range(len(glad.word_id)):
	print i
	lst.append(cosine(mu, glad.wordvec[i]))



s = sorted(range(len(lst)),key=lambda x:lst[x], reverse = True)

for i in range(10):
	print glad.id_word[s[i]]

'''

class GLDA:
	def __init__(self, T, alpha, K, NU, MU, SIGMA, D):
		self.D = D
		self.T = T
		self.alpha = alpha
		self.Kappa = K
		self.Nu = NU
		self.Mu = MU
		self.Sigma = SIGMA
		self.docs = []

	def load_wordvec(self, filename):
		input = open(filename, 'r')
		line = input.readline()
		temp = json.loads(line)
		index = 0
		word_vec = []
		self.word_id = {}
		self.id_word = {}
		for each in temp:
			self.word_id[each] = index
			self.id_word[index] = each
			word_vec.append(temp[each])
			index += 1
		input.close()
		self.wordvec = numpy.array(word_vec)		



	def load_corpus(self, filename):
		self.corpus = []
		input = open(filename, 'r')
		for line in input:
			line = line.strip().lower()
			doc = line.split()
			if len(doc) > 50:
				self.corpus.append(doc)
		input.close()		





        def set_corpus(self):
		counter = 0
		temp = [[self.word_id[term] for term in doc if term in self.word_id] for doc in self.corpus]
		for each in temp:
			if len(each) > 50:
				self.docs.append(each)
			else:
				counter += 1
		print counter
                M = len(self.docs)
                V = len(self.word_id)
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
		self.mu = []
		self.sigma = []
		self.nu = []
		self.kappa = []
		self.summary = []
		self.variance = []
		self.average = []
		self.first_part = []
		for i in range(self.T):
			mu, sigma, kappa, nu, summary, variance, average, first_part = self.init_topic(i)
			self.mu.append(mu)
			self.sigma.append(sigma)
			self.kappa.append(kappa)
			self.nu.append(nu)
			self.summary.append(summary)
			self.variance.append(variance)
			self.average.append(average)
			self.first_part.append(first_part)
		self.mu = numpy.array(self.mu)
		self.sigma = numpy.array(self.sigma)
		self.nu = numpy.array(self.nu)
		self.kappa = numpy.array(self.kappa)
		self.summary = numpy.array(self.summary)
		self.variance = numpy.array(self.variance)
		self.average = numpy.array(self.average)
		self.first_part = numpy.array(self.first_part)


	def init_topic(self, z):
		times = self.n_z_t[z][self.n_z_t[z] > 0]
		word_vec = self.wordvec[self.n_z_t[z] > 0]
		summary = numpy.dot(times, word_vec)
		total_time = self.n_z[z]
		average = summary / total_time
		variance = numpy.dot(times, (word_vec - average) ** 2)
		kappa = self.Kappa + total_time
		nu = self.Nu + total_time

		mu = (self.Kappa * self.Mu + summary) / kappa

		sigma = (1 + kappa) / (kappa) * \
			(self.Nu * (self.Sigma) + variance + \
			total_time * self.Kappa / (kappa) * (self.Mu - average)**2 )

		first_part = self.D * (gamln((nu + 1) / 2) - gamln(nu / 2) - numpy.log(nu) / 2)

		return mu, sigma, kappa, nu, summary, variance, average, first_part


	'''
	TODO: self.Mu is usually set to be zero!
	'''

	def cal_mu(self, kappa, summary):
		return (self.Kappa * self.Mu + summary) / kappa


	def cal_sigma(self, kappa, variance, total_time, average):
		return 	(1 + kappa) / (kappa) * \
			(self.Nu * (self.Sigma) + variance + \
			total_time * self.Kappa / (kappa) * (self.Mu - average)**2 ) 


	def update_topic(self, z, t, flag):
		vec = self.wordvec[t]
		total_time = self.n_z[z]
		if flag == '-':
			self.summary[z] -= vec
			temp_average = self.average[z]
			self.average[z] = self.summary[z] / total_time
			self.variance[z] -= (vec - temp_average) * (vec - self.average[z])
			self.nu[z] -= 1
			self.kappa[z] -= 1
			self.first_part[z] = self.D * (gamln((self.nu[z] + 1) / 2) - gamln(self.nu[z] / 2) - numpy.log(self.nu[z]) / 2)
		else:
                        self.summary[z] += vec
			temp_average = self.average[z]
                        self.average[z] = self.summary[z] / total_time
                        self.variance[z] += (vec - temp_average) * (vec - self.average[z])
                        self.nu[z] += 1
                        self.kappa[z] += 1
			self.first_part[z] = self.D * (gamln((self.nu[z] + 1) / 2) - gamln(self.nu[z] / 2) - numpy.log(self.nu[z]) / 2)
		self.mu[z] = self.cal_mu(self.kappa[z], self.summary[z])
		self.sigma[z] = self.cal_sigma(self.kappa[z], self.variance[z], total_time, self.average[z])		


	@profile
	def gassian_likelihood(self, x):
		#first_part = self.D * (gamln((self.nu + 1) / 2) - gamln(self.nu / 2) - numpy.log(self.nu) / 2)
		temp = (1 - self.nu[:, numpy.newaxis]) / 2  *  numpy.log(1 + ((x - self.mu) ** 2) / (self.sigma)) 
		second_part = temp.sum(axis = 1)
		temp = self.first_part + second_part
		temp -= temp.max()		
		return numpy.exp(temp)


	@profile
	def inference(self):
		V = len(self.word_id)
		for m, doc in zip(range(len(self.docs)), self.docs):
			#print 'document ', m, ' out of', len(self.docs)
			for n in range(len(doc)):
				t = doc[n]
				z = self.z_m_n[m][n]
				self.n_m_z[m, z] -= 1
				self.n_z_t[z, t] -= 1
				self.n_z[z] -= 1

				self.update_topic(z, t, '-')

				word_topic = self.gassian_likelihood(t)
				topic_document =  self.n_m_z[m] + self.alpha
				p_z = word_topic * topic_document
				new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
				self.z_m_n[m][n] = new_z
                                self.n_m_z[m, new_z] += 1
                                self.n_z_t[new_z, t] += 1
                                self.n_z[new_z] += 1


				self.update_topic(new_z, t, '+')



        def phi(self):
                #V = len(self.word_id)
                #return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] +  V * self.beta)
		return self.n_z_t / self.n_z[:, numpy.newaxis]


        def theta(self):
                return (self.n_m_z + self.alpha) / (self.n_m_z.sum(axis = 1)[:, numpy.newaxis] + self.T * self.alpha)



        def display_topic(self, n = 10):
                dist = self.phi()
                for i in range(self.T):
                        s = dist[i]
                        index = sorted(range(len(s)), key=lambda k: s[k], reverse = True)
                        string = ''
                        for j in range(n):
                                string += self.id_word[index[j]] + '  '
                        print string








T = 10
D = 300
alpha = numpy.array([1.0] * T) / T
K = 1.0
MU = numpy.array([0.0] * D)
NU = 1.0
SIGMA = numpy.array([1.0] * D)

wordvec_file = 'toy_word2vec'
corpus_file = 'summary_doc'
glad = GLDA(T, alpha, K, NU, MU, SIGMA, D)
glad.load_wordvec(wordvec_file)
glad.load_corpus(corpus_file)
glad.set_corpus()

print 'Start running'
print datetime.datetime.now()
glad.inference()
glad.inference()





