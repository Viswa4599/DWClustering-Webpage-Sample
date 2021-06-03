
import sys
import os
import numpy as np 
from sklearn import manifold
import pandas as pd
from .clustering import Clusterer
import pickle
import os
import re
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import time
import argparse
import random

#Django Imports
from django.shortcuts import render
from django.http import HttpResponse
from .forms import ParameterForm
# Create your views here.

def run_cluster(path,sample,recheck,multcorp,thresh,k):
    parent = path
    ground_truth = pd.DataFrame(columns = ['id','path'])
    if(multcorp):
        domains = os.listdir(parent)
        for domain in domains:
            received_data = pd.read_csv(parent + domain +'/pages.csv',usecols = ['id','path'])
            ground_truth = ground_truth.append(received_data,ignore_index = True)
            
    else:
        domains = ['']
        ground_truth = pd.read_csv(parent+'/pages.csv',usecols=['id','path'])
    
    ground_truth = ground_truth.drop(['path'],axis=1)

    if sample == 0:
        acceptable_size = len(ground_truth)/100
        data_sample = ground_truth
    else:
        acceptable_size = sample/100
        data_sample = ground_truth.sample(n=sample,random_state=1)
  
    if(acceptable_size<1):
        acceptable_size =1
    list_of_labels=[]
    list_of_pages = []
    for page in data_sample['id']:
        list_of_pages.append(str(page)+'.html')
  
    random.shuffle(list_of_pages)

    THRESHOLD = thresh
    K_VALUE = k
    dicts = [{},{},{}]
    filename = 'output.txt'
    outfile = open(filename,'wb')
    pickle.dump(dicts,outfile)
    outfile.close()

    clusterer = Clusterer(threshold=THRESHOLD,k=K_VALUE,dict_file = filename,minority=acceptable_size)
    start_time = time.time()
    i = 0
    for domain in domains:
        for page in list_of_pages:   
            try:
                with open(parent+domain+'/html/'+page,'r') as fin:
                    html = ''.join(fin.readlines())
                print(i)
                i+=1
            except:
                print('HTML Opening Exception or Invalid Path')
                continue
            clusterer.cluster(html,page)
    with open(filename, 'rb') as handle:
            dicts = pickle.load(handle)

    centroids = dicts[0]
    centroids_name = dicts[1]
    cohesion = dicts[2]
    cohesionAVG ={}

    for center in centroids_name:
        clus_size = len(centroids_name[center])
        #cohesionAVG.append(cohesion[center]/clus_size)
        print(center,' : ',clus_size)#,' Cohesion: ',cohesion[center]/clus_size)
        if(len(centroids_name[center])<acceptable_size):
            print(centroids_name[center])
        else:
            print(random.sample(centroids_name[center],int(acceptable_size)))

    #print('STD of cohesion: ' , np.std(cohesion))

    if(recheck == True):
        clusterer.recheck()
        cohesionAVG = []
    else:
        exit(1)
        
    with open(filename, 'rb') as handle:
            dicts = pickle.load(handle)

    centroids = dicts[0]
    centroids_name = dicts[1]
    cohesion = dicts[2]

    for center in centroids_name:
        if('0000' in centroids_name[center]):
            continue 
        clus_size = len(centroids_name[center])
        #cohesionAVG.append(cohesion[center]/clus_size)
        print(center,' : ',clus_size)#,' Cohesion: ',cohesion[center]/clus_size)
        if(len(centroids_name[center])<acceptable_size):
            print(centroids_name[center])
        else:
            print(random.sample(centroids_name[center],int(acceptable_size)))


    #PLOTTING SIMILARITY BASED GRAPH FOR LOCAL TESTING
    #HAVE TO PORT TO DJANGO PAGE
    end_time = time.time()
    clustering_time = end_time-start_time
    return distance_grapher(filename,clustering_time)


def default_or_not(data):
    if data['sample'] is None:
        data['sample'] = 0
    if data['threshold'] is None:
        data['threshold'] = 76  
    if data['kval'] is None:
        data['kval'] = 3
    if data['recheck'] is None:
        data['recheck'] = 1
    
    return data

def distance_grapher(filename,time):

    with open(filename, 'rb') as handle:
        dicts = pickle.load(handle)
    centroids = dicts[0] # cluster centers (which are parsed HTML Page)
    centroids_name = dicts[1]# list of the names of all pages
    cohesion = dicts[2]     
    clusterer = Clusterer(threshold=0.7,k=0.3,dict_file = filename,minority=0)
    print('Inter Cluster Matrix: \n')
    intermat = np.empty((len(centroids),len(centroids)))
    for center1 in centroids:
        for center2 in centroids:         
            intermat[center1][center2] = clusterer.get_similarity(centroids[center1][0][0],centroids[center1][0][1],centroids[center2])    
            
    adist = 1-intermat#np.array(dists)
    print(adist)
    nearest_centers = []


    i = 0
    flag = 0
    for distances in adist:
        if(len(distances)==1):
            nearest_centers.append(0)
            flag = 1
            break
        nc = 1000
        distances = list(distances)
        j = 0
        for distance in distances:
            if j == i:
                j+=1
                continue
            if distance <= nc:
                nc = distance
            j+=1
        nearest_centers.append(distances.index(nc))
        i+=1
  
    amax = np.amax(adist)
    if flag == 0:
        adist /= amax

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    results = mds.fit(adist)

    coords = results.embedding_

    print(len(centroids_name))
    if len(centroids_name) == 1:
        print('hi')
        merge_button_flag = 1
    else:
        print('ho')
        merge_button_flag = 0

    data = {'time':time,'clusters':centroids_name,'x':coords[:,0],'y':coords[:,1],'ncs':nearest_centers,'mgflag':merge_button_flag}
    print(data)
    return data

# ALL PAGES IN ABSORBED WILL BELING TO ABSORBER
def merge_clusters(absorbed,absorber):

    filename = 'output.txt'
    with open(filename, 'rb') as handle:
        dicts = pickle.load(handle)

    centroids = dicts[0] # cluster centers (which are parsed HTML Page)
    centroids_name = dicts[1]# list of the names of all pages
    cohesion = dicts[2]        

    for page in centroids_name[absorbed]:
        centroids_name[absorber].append(page)
    for coh in cohesion[absorbed]:
        cohesion[absorber].append(coh)

    centroids_name[absorbed].append('0000') #To mark nullified clusters.Temporary measure.
    centroids[absorbed] = 0
    cohesion[absorbed] = 0

    new_centroids_name = {}
    new_cohesion = {}
    new_centroids = {}
    i = 0
    for center in centroids_name:
        if '0000' in centroids_name[center]:
            continue
        new_centroids_name[i] = centroids_name[center]
        new_centroids[i] = centroids[center]
        new_cohesion[i]=cohesion[center]
        i+=1
    
    new_dicts = [new_centroids,new_centroids_name,new_cohesion]
    with open(filename, 'wb') as handle:
        pickle.dump(new_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    centroids_name = new_centroids_name
    centroids = new_centroids
    cohesion = new_cohesion

    return distance_grapher(filename,0)




def process_data(data):
    render_data = {}
    size_data = {}
    center_data = {}
    x={}
    y={}
    closest_centers = {}
    render_data.update({'time':data['time']})

    centroids_name = data['clusters']


    for center in centroids_name:
        size_data.update({'size'+str(center):len(centroids_name[center])})
        center_data.update({'center'+str(center):centroids_name[center][0]})
        x.update({center:data['x'][center]})
        y.update({center:data['y'][center]})
        closest_centers.update({center:data['ncs'][center]})

    render_data.update({'sizes':size_data})
    render_data.update({'centers':center_data})
    render_data.update({'flag':1})
    render_data.update({'x':x})
    render_data.update({'y':y})
    render_data.update({'ncs':closest_centers})
    render_data.update({'mgflag':data['mgflag']})

    print(render_data)
    return render_data

    
    
def clusterview(request):
    if(request.method == 'POST'):
        data = list(request.POST.items())
        message = data[1][1]
        if 'merge' in message:
            print("MERGINGNGIGNIGN")
            original =  int(message[-2])
            nearest = int(message[-1])
            data = merge_clusters(original,nearest)
            render_data = process_data(data)
            return render(request,'UI.html',render_data)

        form = ParameterForm(request.POST)
        if form.is_valid():
            args = default_or_not(form.cleaned_data)
            if args['recheck'] == 0:
                recheck = False
            else:
                recheck = True
            path = os.getcwd()+'/templates/Tochka Pages'
            sample = args['sample']
            multcorp = False
            threshold = args['threshold']/100
            k = args['kval']/100
            data = run_cluster(path,sample,recheck,multcorp,threshold,k)
            render_data = process_data(data)
            return render(request,'UI.html',render_data)

        else:
            print('Invalid Form')
    else:
        form = ParameterForm()
    data = {'flag':1}
    return render(request,template_name='UI.html')

def centerview(request):
    if request.method == 'POST':
        data = list(request.POST.items())
        page = data[1][1]
        return render(request,template_name=page)
    return HttpResponse('pari')