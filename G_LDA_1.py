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


for i in range(10):
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


	#
	#TODO: better find a reasonable word set
	#
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
		'''
		calculate mu, sigma, kappa, nu for initialized tpics.
		'''
		self.mu = []
		self.sigma = []
		self.nu = []
		self.kappa = []
		self.summary = []
		self.variance = []
		self.average = []
		for i in range(self.T):
			mu, sigma, kappa, nu, summary, variance, average = self.update_topic(i)
			self.mu.append(mu)
			self.sigma.append(sigma)
			self.kappa.append(kappa)
			self.nu.append(nu)
			self.summary.append(summary)
			self.variance.append(variance)
			self.average.append(average)
		self.mu = numpy.array(self.mu)
		self.sigma = numpy.array(self.sigma)
		self.nu = numpy.array(self.nu)
		self.kappa = numpy.array(self.kappa)
		self.summary = numpy.array(self.summary)
		self.variance = numpy.array(self.variance)
		self.average = numpy.array(self.average)


	def update_topic(self, z):

		times = self.n_z_t[z][self.n_z_t[z] > 0]
		word_vec = self.wordvec[self.n_z_t[z] > 0]
		summary = numpy.dot(times, word_vec)
		#total_time = times.sum()
		total_time = self.n_z[z]
		average = summary / total_time
		variance = numpy.dot(times, (word_vec - average) ** 2)
		kappa = self.Kappa + total_time
		nu = self.Nu + total_time
		mu = (self.Kappa * self.Mu + summary) / kappa

		sigma = (1 + kappa) / (nu * kappa) * \
			(self.Nu * (self.Sigma ** 2) + variance + \
			total_time * self.Kappa / (self.Kappa + total_time) * (self.Mu - average)) ** 2 

		return mu, sigma, kappa, nu, summary, variance, average


	'''
	TODO: self.Mu is usually set to be zero!
	'''
	def cal_mu(self, kappa, summary):
		return (self.Kappa * self.Mu + summary) / kappa


	def cal_sigma(self, kappa, nu, variance, total_time, average):
		return 	(1 + kappa) / (nu * kappa) * \
			(self.Nu * (self.Sigma ** 2) + variance + \
			total_time * self.Kappa / (self.Kappa + total_time) * (self.Mu - average)) ** 2 


	def subtract_update_topic(self, z, t, flag):
		vec = self.wordvec[t]

		summary = self.summary[z]
		variance = self.variance[z]
		average = self.average[z]
		kappa = self.kappa[z]
		nu = self.nu[z]
		mu = self.mu[z]
		sigma = self.sigma[z]
		total_time = self.n_z[z]

		if flag == '-':
			new_summary = summary - vec
			new_average = new_summary / total_time
			new_variance = variance - (vec - average) * (vec - new_average)
			new_nu = nu - 1
			new_kappa = kappa - 1
		else:
			new_summary = summary + vec
			new_average = new_summary / total_time
			new_variance = variance + (vec - average) * (vec - new_average)
			new_nu = nu + 1
			new_kappa = kappa + 1		

		new_mu = self.cal_mu(new_kappa, new_summary)
		new_sigma = self.cal_sigma(new_kappa, new_nu, new_variance, total_time, new_average)
		
		return new_mu, new_sigma, new_kappa, new_nu, new_summary, new_variance, new_average


	#give topic i, word x, calculate the 
	def student_t(self, x, i):
		mu = self.mu[i]
		sigma = self.sigma[i]
		kappa = self.kappa[i]
		nu = self.nu[i]
		first_part = self.D * (gamln((nu + 1) / 2) - gamln(nu / 2) - numpy.log(nu * pi) / 2)
		#temp = numpy.log(1 + ((x - mu) ** 2) / (sigma * nu)) * 2 #incorrect
		temp = numpy.log(1 + ((x - mu) ** 2) / (sigma * nu)) * (1 - nu) / 2
		second_part = temp.sum()
		return first_part + second_part


	'''
	TODO: instead of a loop, can we calculate the whole matrix once?
	'''
	def gassian_likelihood(self, x):
		results = []
		for i in range(self.T):
			results.append(self.student_t(x, i))
		temp = numpy.array(results)
		temp = temp - temp.max()
		likelihood = numpy.exp(temp)
		return likelihood



	def new_gassian_likelihood(self, x):

		first_part = self.D * (gamln((self.nu + 1) / 2) - gamln(self.nu / 2) - numpy.log(self.nu) / 2)
		temp = numpy.log(1 + ((x - self.mu) ** 2) / (self.sigma * self.nu[:, numpy.newaxis])) * (1 - self.nu[:, numpy.newaxis]) / 2
		second_part = temp.sum(axis = 1)
		return first_part + second_part


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

				'''
				TODO: what's the chance of being assigned to the same topic?
				'''
				#print datetime.datetime.now()
				mu, sigma, kappa, nu, summary, variance, average = self.subtract_update_topic(z, t, '-')
				self.mu[z] = mu
				self.sigma[z] = sigma
				self.kappa[z] = kappa
				self.nu[z] = nu
				self.summary[z] = summary
				self.variance[z] = variance
				self.average[z] = average

				#print datetime.datetime.now()

				#word_topic = self.gassian_likelihood(t)
				word_topic = self.new_gassian_likelihood(t)
				topic_document =  self.n_m_z[m] + self.alpha
				p_z = word_topic * topic_document
				new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

				#print datetime.datetime.now()

				self.z_m_n[m][n] = new_z				
				self.n_m_z[m, new_z] += 1
				self.n_z_t[new_z, t] += 1
				self.n_z[new_z] += 1

				#print datetime.datetime.now()

				mu, sigma, kappa, nu, summary, variance, average = self.subtract_update_topic(z, t, '+')
				self.mu[z] = mu
				self.sigma[z] = sigma
				self.kappa[z] = kappa
				self.nu[z] = nu
				self.summary[z] = summary
				self.variance[z] = variance
				self.average[z] = average

				#print datetime.datetime.now()
				#return 












