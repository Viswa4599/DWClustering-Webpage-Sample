'''
Author: Viswajit Vinod Nair
25th June 2020
'''

import pandas as pd
import numpy as np
from parsel import Selector 
import os
import time
import pickle
import difflib
from io import StringIO
import lxml.html
from bs4 import BeautifulSoup
from html_similarity import similarity,structural_similarity,style_similarity



class Clusterer:

    def __init__(self,threshold,k,dict_file,minority):
        self.dict_file = dict_file
        self.threshold = threshold
        self.k = k
        self.minority = minority

    #Sequence Matcher for structural similarity
    def seqMatcher(self,tags1,tags2):
        diff = difflib.SequenceMatcher()
        diff.set_seq1(tags1)
        diff.set_seq2(tags2)
        return diff.ratio()

    #Jaccard Similarity for Style Similarity
    def jaccard_similarity(self,set1, set2):
        set1 = set(set1)
        set2 = set(set2)
        intersection = len(set1 & set2)

        if len(set1) == 0 and len(set2) == 0:
            return 1.0

        denominator = len(set1) + len(set2) - intersection
        return intersection / max(denominator, 0.000001)

    #Get tags and classes of an html page
    def get_tagsclasses(self,doc):
        tags = list()
        html = doc
        try:
            doc = lxml.html.parse(StringIO(doc))
        except Exception as e:
            print(e)
        for el in doc.getroot().iter():
            if isinstance(el, lxml.html.HtmlElement):
                tags.append(el.tag)
            elif isinstance(el, lxml.html.HtmlComment):
                tags.append('comment')
            else:
                raise ValueError('Don\'t know what to do with element: {}'.format(el))

        doc = Selector(text=html)
        classes = set(doc.xpath('//*[@class]/@class').extract())
        result = set()
        for cls in classes:
            for _cls in cls.split():
                result.add(_cls)
        return tags,result


    def get_similarity(self,tags,classes,cluster_page):
        clus_tags = cluster_page[0][0]
        clus_classes = cluster_page[0][1]
        style_sim1 = self.jaccard_similarity(classes,clus_classes)
        style_sim2 = self.jaccard_similarity(clus_classes,classes)
        struct_sim1 = self.seqMatcher(tags,clus_tags)
        struct_sim2 = self.seqMatcher(clus_tags,tags)
        simil1 = self.k*struct_sim1 + (1-self.k)*style_sim1
        simil2 = self.k*struct_sim2 + (1-self.k)*style_sim2

        return max(simil1,simil2)


    #Main clustering process
    def cluster(self,html,html_name):

        with open(self.dict_file, 'rb') as handle:
            dicts = pickle.load(handle)

        centroids = dicts[0]
        centroids_name = dicts[1]
        cohesion = dicts[2]
        similarities=[]

        if bool(centroids)==False:
            centroids[0]=[self.get_tagsclasses(html)]
            centroids_name[0]=[html_name]
            cohesion[0]=[1]
            new_dicts = [centroids,centroids_name,cohesion]

            with open(self.dict_file, 'wb') as handle:
                pickle.dump(new_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return

        n_cluster = len(centroids)-1
        
        try:
            tags,classes = self.get_tagsclasses(html)
        except:
            print('DocRootException')
            return

        for cluster in centroids: 
           similarities.append(self.get_similarity(tags,classes,centroids[cluster]))


        if max(similarities)>=self.threshold:
            clust = similarities.index(max(similarities))
            cohesion[clust].append((max(similarities)-self.threshold)/(1-self.threshold))
            centroids_name[clust].append(html_name)
        else:
            n_cluster+=1
            centroids[n_cluster] = [self.get_tagsclasses(html)]
            centroids_name[n_cluster] = [html_name]
            cohesion[n_cluster]=[1]
            print(centroids_name.keys())

        new_dicts = [centroids,centroids_name,cohesion]

        with open(self.dict_file, 'wb') as handle:
            pickle.dump(new_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def recheck(self):

        with open(self.dict_file, 'rb') as handle:
            dicts = pickle.load(handle)

        centroids = dicts[0] # cluster centers (which are parsed HTML Page)
        centroids_name = dicts[1]# list of the names of all pages
        cohesion = dicts[2]        

        for center in centroids:
            if(len(centroids_name[center])<self.minority):
                flag = 0
                print('Double Checking center ',center)
                maxm =0 
                for center2 in centroids:
                    if(len(centroids_name[center2])>=self.minority):
                        try:
                            sim = self.get_similarity(centroids[center2][0][0],centroids[center2][0][1],centroids[center])
                        except:
                            print('Ignoring Cluster Due to Type Error')
                            continue

                        if(sim>maxm and sim>(self.threshold-0.05)):
                            maxm = sim
                            new_center = center2
                            flag = 1
                if(flag==1):
                    print(len(centroids_name[center]))
                    print(len(centroids_name[new_center]))
                    print(cohesion[new_center])
                    for page in centroids_name[center]:
                        centroids_name[new_center].append(page)
                    for coh in cohesion[center]:
                        cohesion[new_center].append(coh)
                    print(cohesion[new_center])
                    print(len(centroids_name[new_center]))
                    centroids_name[center].append('0000') #To mark nullified clusters.Temporary measure.
                    centroids[center] = 0
                    cohesion[center] = 0

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
        with open(self.dict_file, 'wb') as handle:
            pickle.dump(new_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__" :
    print("Welcome to the Real Time HTML Template Clusterer")