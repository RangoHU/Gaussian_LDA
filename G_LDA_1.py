import scipy.stats
import numpy
import re
from math import *
from optparse import OptionParser
from scipy.special import gammaln as gamln
from scipy.spatial.distance import cosine

'''
from G_LDA_1 import *
T = 10
D = 300
alpha = numpy.array([1.0] * T) / T
K = 1.0
MU = numpy.array([0.0] * D)
NU = 1.0
SIGMA = numpy.array([1.0] * D)

wordvec_file = 'test_wordvec'
corpus_file = 'test_corpus'
iteration = 300	
glad = GLDA(T, alpha, K, NU, MU, SIGMA, D)
glad.load_wordvec(wordvec_file)
glad.load_corpus(corpus_file)
glad.set_corpus()

glad.inference()

print glad.n_m_z

lst = []
for i in range(len(glad.word_id)):
	print i
	lst.append(cosine(mu, glad.wordvec[i]))



s = sorted(range(len(lst)),key=lambda x:lst[x], reverse = True)
'''


def test():
	#parser = OptionParse()
	#parse.add_option()
	T = 0
	alpha = 0
	K = 0
	MU = 0
	NU = 0
	SIGMA = 0
	D = 0
	wordvec_file = ''
	corpus_file = ''
	iteration = 300	

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
		self.D = D
		self.T = T
		self.alpha = alpha
		self.Kappa = K
		self.Nu = NU
		self.Mu = MU
		self.Sigma = SIGMA

	def load_corpus(self, filename):
		self.corpus = []
		input = open(filename, 'r')
		line = input.readline()
		for line in input:
			doc = line.split() #for test use
			#doc = re.findall(r'\[(.+?)\](.+)', line.lower())	
			if len(doc) > 0:
				self.corpus.append(doc)
		input.close()		


	def load_wordvec(self, filename):
		word_vec = []
		self.word_id = {}
		self.id_word = {}
		input = open(filename, 'r')
		line = input.readline().strip()
		while line != '':
			temp = line.split(' ')
			word = temp[0]
			lst = temp[1 : len(temp)]
			word_vec.append(map(float, lst))
			index = len(self.word_id)
			self.word_id[word] = index
			self.id_word[index] = word
			line = input.readline().strip()
		input.close()
		self.wordvec = numpy.array(word_vec)		
	


	def set_corpus(self):
		self.docs = [[self.word_id[term] for term in doc if term in self.word_id] for doc in self.corpus]
		M = len(self.corpus)
		V = len(self.word_id)
		self.z_m_n = []
		self.n_m_z = numpy.zeros((M, self.T), dtype = int)
		self.n_z_t = numpy.zeros((self.T, V), dtype = int)
		self.n_z = numpy.zeros(self.T, dtype = int)
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
		for i in range(self.T):
			mu, sigma, kappa, nu = self.update_topic(i)
			self.mu.append(mu)
			self.sigma.append(sigma)
			self.kappa.append(kappa)
			self.nu.append(nu)
		self.mu = numpy.array(self.mu)
		self.sigma = numpy.array(self.sigma)
		self.nu = numpy.array(self.nu)
		self.kappa = numpy.array(self.kappa)

	
	def update_topic(self, t):
		times = self.n_z_t[t][self.n_z_t[t] > 0]
		word_vec = self.wordvec[self.n_z_t[t] > 0]
		summary = numpy.dot(times, word_vec)
		total_time = times.sum()
		average = summary / total_time
		variance = numpy.dot(times, (word_vec - average) ** 2)
		kappa = self.Kappa + total_time
		nu = self.Nu + total_time
		mu = (self.Kappa * self.Mu + summary) / kappa
		sigma = (1 + kappa) * (self.Nu * (self.Sigma ** 2) + variance + total_time * self.Kappa / (self.Kappa + total_time) * (self.Mu - average)) ** 2 / (nu * kappa)
		return mu, sigma, kappa, nu



	def student_t(self, x, i):
		mu = self.mu[i]
		sigma = self.sigma[i]
		kappa = self.kappa[i]
		nu = self.nu[i]
		'''
		first_part = (exp(gamln((nu + 1) / 2) - gamln(nu / 2)) / sqrt(nu * pi)) ** self.D
		temp = (1 + ((x - mu) ** 2) / (sigma * nu)) ** 2	
		second_part = numpy.prod(temp)
		return first_part * second_part
		'''
		first_part = self.D * (gamln((nu + 1) / 2) - gamln(nu / 2) - log(nu * pi) / 2)
		temp = numpy.log(1 + ((x - mu) ** 2) / (sigma * nu)) * 2
		second_part = temp.sum()
		
		return first_part + second_part

	def gassian_likelihood(self, x):
		results = []
		for i in range(self.T):
			results.append(self.student_t(x, i))
		temp = numpy.array(results)
		temp = temp - temp.max()
		likelihood = numpy.exp(temp)
		return likelihood


	def inference(self):
		V = len(self.word_id)
		for m, doc in zip(range(len(self.docs)), self.docs):
			print 'now is processing ', m, ' th doc'
			print 'length of this document is ', len(doc) 
			for n in range(len(doc)):
				#print 'now is processing', m, ' th doc', n, ' th word of ', len(doc) 
				t = doc[n]
				z = self.z_m_n[m][n]
				self.n_m_z[m, z] -= 1
				self.n_z_t[z, t] -= 1
				self.n_z[z] -= 1
				mu, sigma, kappa, nu = self.update_topic(z)
				self.mu[z] = mu
				self.sigma[z] = sigma
				self.kappa[z] = kappa
				self.nu[z] = nu

				temp = self.gassian_likelihood(t) 
				#print temp
				p_z = temp * (self.n_m_z[m] + self.alpha)
				new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

				self.z_m_n[m][n] = new_z				
				self.n_m_z[m, new_z] += 1
				self.n_z_t[new_z, t] += 1
				self.n_z[new_z] += 1
				mu, sigma, kappa, nu = self.update_topic(new_z)
				self.mu[new_z] = mu
				self.sigma[new_z] = sigma
				self.kappa[new_z] = kappa
				self.nu[new_z] = nu


	
	def doc_topic(self):
		return (self.n_m_z + self.alpha) / (self.n_m_z.sum(axis = 1)[:, numpy.newaxis] + self.alpha)

			
