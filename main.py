import csv
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StemmerFactory()
factory2 = StopWordRemoverFactory()
stopword = factory2.create_stop_word_remover()
stemmer = factory.create_stemmer()

# import dataset
with open('diagnosa.csv') as csv_file:
	ds = csv.reader(csv_file, delimiter=';')
	dataset = list(ds)

# separating which already diagnosed yet
diagnosed = []
disease = []
no_diagnosed = []
for i in dataset:
	if i[1] == '' :
		no_diagnosed.append(i[0])
	else:
		diagnosed.append(i[0])
		disease.append(i[1])

# preproccessing
size_diag = len(diagnosed)
size_nodiag = len(no_diagnosed)
data = diagnosed + no_diagnosed

terms = []
for i in data:
	terms.append(stemmer.stem(stopword.remove(i)).split())

# print(diagnosed)

# extraction
# menghitung tf
def compute_tf_dict(doc):
	tf_dict = {}
	for term in doc:
		if term in tf_dict:
			tf_dict[term]+=1
		else:
			tf_dict[term]=1

	for term in tf_dict:
		tf_dict[term] = tf_dict[term]/len(doc)
	return tf_dict

tf_dict = []
for i in terms:
	tf_dict.append(compute_tf_dict(i))

# menghitung count
def compute_count_dict(terms):
    count_dict = {}
    for doc in terms:
    	term_dist = set(doc)
    	for term in term_dist:
            if term in count_dict:
                count_dict[term]+=1
            else:
                count_dict[term]=1
    return count_dict

count_dict = compute_count_dict(terms)

# menghitung idf
def compute_idf():
	idf_dict={}
	for term in count_dict:
		idf_dict[term] = math.log(len(terms)/count_dict[term])
	return idf_dict

idf_dict = compute_idf()

# menghitung tf-idf
def compute_tf_idf(tf_dict):
	tf_idf_dict = {}
	for term in tf_dict:
		tf_idf_dict[term] = tf_dict[term] * idf_dict[term]
	return tf_idf_dict

tf_idf_dict = [compute_tf_idf(doc) for doc in tf_dict]

# print(tf_idf_dict[0])

term_dict = sorted(count_dict.keys())

# print(term_dict)

# representation
def compute_tf_idf_vector(doc):
	tf_idf_vector = [0.0] * len(term_dict)

	for i, term in enumerate(term_dict):
		if term in doc:
			tf_idf_vector[i] = doc[term]
	return tf_idf_vector

tf_idf_vector = [compute_tf_idf_vector(doc) for doc in tf_idf_dict]

# similarity with cosine similarity
def dot_product(vector_x, vector_y):
	dot = 0.0
	for e_x, e_y in zip(vector_x,vector_y):
		dot += e_x * e_y
	return dot

def magnitude(vector):
	mag = 0.0
	for index in vector:
		mag += math.pow(index,2)
	return math.sqrt(mag)


diagnosed_vector = tf_idf_vector[:size_diag]
no_diagnosed_vector = tf_idf_vector[size_diag:]

new_desease = []

for i in enumerate(no_diagnosed):
	temp = 0.0
	for j in enumerate(diagnosed):
		num = dot_product(no_diagnosed_vector[i[0]],diagnosed_vector[j[0]])/(magnitude(no_diagnosed_vector[i[0]])*magnitude(diagnosed_vector[j[0]]))
		if num >= temp:
			temp = num
			idx = j[0]
	new_desease.append(disease[idx])


with open('result.csv',mode='w',encoding='utf-8',newline="") as diseases:
	diseases_writer = csv.writer(diseases,delimiter=";",quotechar='"', quoting=csv.QUOTE_MINIMAL)

	diseases_writer.writerow(['data train'])
	for i in range(size_diag):
		res = diagnosed[i] +";" +disease[i]
		diseases_writer.writerow([res])
		# print(res)
	diseases_writer.writerow(['data test + result'])
	for i in range(size_nodiag):
		res = no_diagnosed[i] + ";" +new_desease[i]
		diseases_writer.writerow([res])

