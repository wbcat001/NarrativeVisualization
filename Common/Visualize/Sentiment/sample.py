# from transformers import pipeline

# sentiment_pipeline = pipeline("sentiment-analysis")
# data = ["I love you", "I hate you"]
# result = sentiment_pipeline(data)
# print(result[0]["label"])
import labMTsimple.storyLab
from tqdm import trange
import pandas as pd
import codecs
import re
import seaborn as sns
from numpy import floor,zeros,array
import numpy as np
import shutil
import subprocess
import datetime
from numpy import floor

lexicon = pd.read_csv('/content/moviearcs/NRC-VAD-Lexicon.txt', sep='\t')
lexicon['Ranking'] = np.arange(1, len(lexicon)+1)

columnsTitles = ["Word","Ranking","Arousal","Valence","Dominance"]
lexicon = lexicon.reindex(columns=columnsTitles)
avd_dict = lexicon.set_index('Word').T.to_dict('list')
nlist = lexicon['Arousal'].tolist()
wlist = lexicon['Word'].tolist()

labMT = avd_dict
labMTvector = nlist
labMTwordList = wlist

def chopper(words,labMT,labMTvector,outfile,minSize=1000):
  # print "now splitting the text into chunks of size 1000"
  # print "and printing those frequency vectors"
  allFvec = []
  
  for i in range(int(floor(len(words)/minSize))):
    chunk = str('')
    if i == int(floor(len(words)/minSize))-1:
      # take the rest
      # print 'last chunk'
      # print 'getting words ' + str(i*minSize) + ' through ' + str(len(words)-1)
      for j in range(i*minSize,len(words)-1):
        chunk += words[j]+str(' ')
    else:
      # print 'getting words ' + str(i*minSize) + ' through ' + str((i+1)*minSize)
      for j in range(i*minSize,(i+1)*minSize):
        chunk += words[j]+str(' ')
        # print chunk[0:10]
    textValence,textFvec = labMTsimple.storyLab.emotion(chunk,labMT,shift=True,happsList=labMTvector)
      # print chunk
    # print 'the valence of {0} part {1} is {2}'.format(rawbook,i,textValence)

    allFvec.append(textFvec)


  f = open(outfile,"w")
  if len(allFvec) > 0:
    print("writing out the file to {0}".format(outfile))
    f.write('{0:.0f}'.format(allFvec[0][0]))
    for k in range(1,len(allFvec)):
      f.write(',{0:.0f}'.format(allFvec[k][0]))
    for i in range(1,len(allFvec[0])):
      f.write("\n")
      f.write('{0:.0f}'.format(allFvec[0][i]))
      for k in range(1,len(allFvec)):
        f.write(',{0:.0f}'.format(allFvec[k][i]))
    f.close()
  else:
    print("\""*40)
    print("could not write to {0}".format(outfile))
    print("\""*40)
  # print "done!"


def precomputeTimeseries(fullVec,labMT,labMTvector,outfile):
  minWindows = 10
  timeseries = [0 for i in range(len(fullVec[0])+1)]
  # print len(timeseries)

  textFvec = [0 for j in range(len(fullVec))]
  for i in range(0,minWindows//2):
    textFvec = [textFvec[j]+fullVec[j][i] for j in range(len(fullVec))]
    # print "adding point {0}".format(i)

  for i in range(minWindows//2,minWindows):
    # print "scoring"
    stoppedVec = labMTsimple.storyLab.stopper(textFvec,labMTvector,labMTwordList,stopVal=2.0)
    timeseries[i-minWindows//2] = labMTsimple.storyLab.emotionV(stoppedVec,labMTvector)
    # print "adding point {0}".format(i)
    textFvec = [textFvec[j]+fullVec[j][i] for j in range(len(fullVec))]

  for i in range(minWindows,len(timeseries)-1):
    # print "scoring"
    stoppedVec = labMTsimple.storyLab.stopper(textFvec,labMTvector,labMTwordList,stopVal=2.0)
    timeseries[i-minWindows//2] = labMTsimple.storyLab.emotionV(stoppedVec,labMTvector)
    # print "adding point {0}".format(i)
    # print "removing point {0}".format(i-minWindows)
    textFvec = [textFvec[j]+fullVec[j][i]-fullVec[j][i-minWindows] for j in range(len(fullVec))]

  for i in range(len(timeseries)-1,len(timeseries)+minWindows//2):
    # print "scoring"
    stoppedVec = labMTsimple.storyLab.stopper(textFvec,labMTvector,labMTwordList,stopVal=2.0)
    timeseries[i-minWindows//2] = labMTsimple.storyLab.emotionV(stoppedVec,labMTvector)
    # print "removing point {0}".format(i-minWindows)
    textFvec = [textFvec[j]-fullVec[j][i-minWindows] for j in range(len(fullVec))]



  g = open(outfile,"w")
  g.write("{0}".format(timeseries[0]))
  for i in range(1,len(timeseries)):
    g.write(",")
    # g.write("{0:.5f}".format(timeseries[i]))
    g.write("{0}".format(timeseries[i]))
  g.write("\n")
  g.close()


def process():
  # windowSizes = [500,1000,2000,5000,10000]
  windowSizes = [2000]
#   movie = 'new4_into_the_wild'

  words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+",raw_text_clean,flags=re.UNICODE)]
  lines = raw_text_clean.split("\n")
  kwords = []
  klines = []
  for i in range(len(lines)):
    if lines[i][0:3] != "<b>":
      tmpwords = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+",lines[i],flags=re.UNICODE)]
      kwords.extend(tmpwords)
      klines.extend([i for j in range(len(tmpwords))])

  # avhapps = emotion(raw_text,labMT)
  print("length of the original parse")
  print(len(words))
  print("length of the new parse")
  print(len(kwords))
  # print len(klines)
  # print klines[0:20]

  for window in windowSizes:
    print(window)

    # print klines[0:(window/10)]
    breaks = [klines[window//10*i] for i in range(int(floor(float(len(klines))//window*10)))]
    breaks[0] = 0
    # print [window/10*i for i in xrange(int(floor(float(len(klines))/window*10)))]
    # print breaks
    # print len(breaks)
    f = open("/content/moviearcs/word-vectors/"+str(window)+"/"+movie+"-breaks.csv","w")
    f.write(",".join(map(str,breaks)))
    f.close()
    chopper(kwords,labMT,labMTvector,"/content/moviearcs/word-vectors/"+str(window)+"/"+movie+".csv",minSize=window//10)

    f = open("/content/moviearcs/word-vectors/"+str(window)+"/"+movie+".csv","r")
    fullVec = [list(map(int,line.split(","))) for line in f]
    f.close()

    # some movies are blank
    if len(list(fullVec)) > 0:
      if len(list(fullVec[0])) > 9:
        precomputeTimeseries(fullVec,labMT,labMTvector,"/content/moviearcs/timeseries/"+str(window)+"/"+movie+".csv")
    else:
      print("this movie is blank:")
      print(movie.title)
      movie.exclude = True
      movie.excludeReason = "movie blank"
    return kwords
  
import matplotlib.pyplot as plt

def process_movie():
    kwords = process()
    df_time_series = pd.read_csv("/content/moviearcs/timeseries/"+str(window)+"/"+movie+".csv")
    df_time_series = df_time_series.T
    df_time_series = df_time_series.reset_index()
    df_time_series = df_time_series.rename(columns={"index": "score"})
    df_time_series['score'] = pd.to_numeric(df_time_series['score'])
    plt.figure()
    plot = sns.lineplot(x=df_time_series.index/max(df_time_series.index), y="score", data=df_time_series)
    plt.close()
    return plot, kwords