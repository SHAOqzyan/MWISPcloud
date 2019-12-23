
import os
import numpy as np
from astropy.table import Table,vstack
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import measure
#fast way to perform DBSCAN
#
from progressbar import *
import math
from myPYTHON import *
from skimage.morphology import watershed
import sys
from skimage.morphology import erosion, dilation
from scipy.ndimage import label, generate_binary_structure,binary_erosion,binary_dilation
from sklearn.cluster import DBSCAN
from madda import  myG210
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import seaborn as sns

from  myGAIA import GAIADIS

gaiaDis=GAIADIS()

doFITS=myFITS()

doG210 = myG210()

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))



class myDBSCAN(object):

	rms = 0.5
	TBModel="minV3minP16_dendroCatTrunk.fit"

	tmpPath="./tmpFiles/"

	def __init__(self):
		pass

	def sumEdgeByCon1(self,extendMask): #7 in total
		raw=extendMask[1:-1,1:-1,1:-1]

		leftShiftZ=extendMask[0:-2, 1:-1, 1:-1]
		rightShiftZ=extendMask[2:, 1:-1 ,1:-1]

		leftShiftY=extendMask[1:-1, 0 : -2, 1:-1]
		rightShiftY=extendMask[1:-1, 2 : ,1:-1]

		leftShiftX=extendMask[1:-1, 1:-1,  0:-2]
		rightShiftX=extendMask[1:-1, 1:-1 , 2: ]


		sumAll=raw+leftShiftZ+rightShiftZ+leftShiftY+rightShiftY+leftShiftX+rightShiftX

		return  sumAll



	def sumEdgeByCon2(self,extendMask): #27 in total
		sumAll=extendMask[1:-1,1:-1,1:-1]*0
		Nz,Ny,Nx= sumAll.shape
		for i in [-1,0,1]:
			for j in [-1,0,1]:
				for k in [-1,0,1]:

					if np.sqrt( abs(i)+abs(j)+abs(k))>1.5:
						continue

					sumAll=sumAll+  extendMask[ 1+i:Nz+1+i , j+1:Ny+1+j , k+1: Nx+1+k  ]

		return  sumAll



	def sumEdgeByCon3(self,extendMask): #27 in total
		raw=extendMask[1:-1,1:-1,1:-1]
		Nz,Ny,Nx= raw.shape
		sumAll=raw*0
		for i in [-1,0,1]:
			for j in [-1,0,1]:
				for k in [-1,0,1]:
					sumAll=sumAll+  extendMask[ 1+i:Nz+1+i , j+1:Ny+1+j , k+1: Nx+1+k  ]


		return  sumAll




	def slowDBSCAN(self,COdata,COHead, min_sigma=2, min_pix=16, connectivity=2 ,region="" ,saveFITS=None ):
		"""
		Use the sklearn DBSCAN to calculate, just for comparison, to test the computeDBSCAN is right
		:param COdata:
		:param COHead:
		:param min_sigma:
		:param min_pix:
		:param connectivity:
		:param region:
		:return:
		"""
		###

		goodIndices=np.where(COdata>= min_sigma*self.rms   )

		coordinates= zip( goodIndices[0] , goodIndices[1] ,goodIndices[2]  )

		#eps=1.5 form connectivity 2,
		if connectivity==2:
			db = DBSCAN(eps=1.5  , min_samples= min_pix      ).fit(coordinates)

		if connectivity==1:
			db = DBSCAN(eps=1.1  , min_samples= min_pix      ).fit(coordinates)
		if connectivity==3:
			db = DBSCAN(eps=1.8  , min_samples= min_pix      ).fit(coordinates)

		labels = db.labels_
		print min(labels),"minimumLabel?"
		#u,c= np.unique(labels,return_counts=True)

		#print len(u)

		mask=np.zeros_like(COdata)-1

		mask[goodIndices]= labels
		if saveFITS==None:
			fits.writeto("dbscanMask1Sigma.fits",mask,header=COHead,overwrite=True)

		else:
			fits.writeto(saveFITS ,mask,header=COHead,overwrite=True)




	def computeDBSCAN(self,COdata,COHead, min_sigma=2, min_pix=16, connectivity=2 ,region="" , getMask=False,savePath="" ,mimicDendro=False):
		"""
		min_pix the the minimum adjacen number for are core point
		:param COdata:
		:param min_sigma:
		:param min_pix:
		:param connectivity:
		:return:
		"""
		#pass


		minValue = min_sigma*self.rms



		Nz,Ny,Nx  = COdata.shape
		extendMask = np.zeros([Nz+2,Ny+2,Nx+2] ,dtype=int)

		goodValues= COdata>=minValue

		extendMask[1:-1,1:-1,1:-1] =  goodValues  #[COdata>=minValue]=1

		s=generate_binary_structure(3,connectivity)


		if connectivity==1:
			coreArray=self.sumEdgeByCon1(extendMask)

		if connectivity==2:
			coreArray=self.sumEdgeByCon2(extendMask)

		if connectivity==3:
			coreArray=self.sumEdgeByCon3(extendMask)

		coreArray = coreArray>=min_pix
		coreArray[  ~goodValues ]=False  # nan could be, #remove falsely, there is a possibility that, a bad value may have lots of pixels around and clould be
		coreArray=coreArray+0

		labeled_core, num_features=label(coreArray,structure=s) #first label core, then expand, otherwise, the expanding would wrongly connected

		selectExpand= np.logical_and(labeled_core==0,  goodValues   )
		#expand labeled_core
		#coreLabelCopy=labeled_core.copy()

		expandTry = dilation(labeled_core , s  ) # first try to expand, then only keep those region that are not occupied previously
		#it is possible  that a molecular cloud may have less pixN than 8, because of the closeness of two

		labeled_core[  selectExpand  ] =  expandTry[ selectExpand  ]

		labeled_array = labeled_core

		if mimicDendro:
			print "Mimicing dendrogram.."
			extendedArray=labeled_array>0
			extendedArray=extendedArray+0
			labeled_array, num_features = label(extendedArray, structure=s)

		saveName="{}dbscanS{}P{}Con{}.fits".format( region,min_sigma,min_pix,connectivity )

		if getMask:

			return labeled_array>0 #actually return mask


		print num_features,"features found!"

		fits.writeto(savePath+saveName, labeled_array, header=COHead, overwrite=True)
		return savePath+saveName



	def maskByGrow(self,COFITS,peakSigma=3,minV=1.):

		COData,COHead=myFITS.readFITS( COFITS )
		markers=np.zeros_like(COData )

		COData[COData<minV* self.rms]=0

		markers[COData>peakSigma*self.rms] = 1

		labels=watershed(COData,markers)
		fits.writeto("growMaskPeak3Min1.fits",labels,header=COHead,overwrite=True)



	def myDilation(self,scimesFITS,rawCOFITS,startSigma=20,endSigma=2, saveName="", maskCOFITS=None,savePath="" ):
		"""
		#because SCIMES removes weak emissions in the envelop of clouds, we need to add them back
		#one possible way is to use svm to split the trunk, test this con the  /home/qzyan/WORK/myDownloads/MWISPcloud/ClusterAsgn_ComplicateVe.fits

		:return:
		"""

		#cloudData,cloudHead = myFITS.readFITS("/home/qzyan/WORK/myDownloads/MWISPcloud/ClusterAsgn_ComplicateVe.fits")

		cloudData,cloudHead = myFITS.readFITS(scimesFITS)

		#rawFITS= rawCOFITS #"/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"


		if rawCOFITS!=None:
			rawCO,rawHead=   myFITS.readFITS( rawCOFITS )

		if maskCOFITS!=None:

			maskData,maskHead=myFITS.readFITS(maskCOFITS)

		#the expansion should stars from high coValue, to low CO values, to avoid cloud cross wak bounarires
		#sCon=generate_binary_structure(3,2)
		print "Expanding clous..."

		sigmaSteps= np.arange(startSigma,endSigma-1,-1)

		if endSigma not in sigmaSteps:
			sigmaSteps=list(sigmaSteps)
			sigmaSteps.append(endSigma)
		print "Step of sigmas, ", sigmaSteps

		cloudData = cloudData + 1  # to keep noise reagion  as 0
		for sigmas in sigmaSteps:

			#produceMask withDBSCAN
			#if sigmas>2:
			if maskCOFITS==None:
				COMask = self.computeDBSCAN( rawCO,rawHead, min_sigma=sigmas, min_pix=8, connectivity=2 ,region="" , getMask=True ) #use all of them
			else:
				COMask=maskData>sigmas*self.rms
			#else:
				#COMask = self.computeDBSCAN(  rawCO,rawHead, min_sigma=sigmas, min_pix=16, connectivity=2 ,region="" , getMask=True )

			for i in range(2000):


				rawAssign=cloudData.copy()

				d1Try=dilation(cloudData  ) #expand with connectivity 1, by defaults

				assignRegion=  np.logical_and(cloudData==0 , COMask )
				cloudData[ assignRegion ] = d1Try[ assignRegion ]


				diff=cloudData -rawAssign
				sumAll=np.sum(diff )
				if sumAll==0:
					print  "Sigmas: {}, Loop:{},  all difference:{}, beak".format(sigmas, i, sumAll)

					break

				else:
					print  "Sigmas: {}, Loop:{}, all difference:{} continue".format(sigmas, i, sumAll)
					continue




		cloudData=cloudData-1

		fits.writeto( savePath+saveName+"_extend.fits",cloudData ,header=cloudHead,overwrite=True)
		return savePath+saveName+"_extend.fits"



	def directLabel(self,COFITS,DBMaskFITS,min_sigma=3,min_pix=8,calCat=True ,useMask=True, peakSigma=3. ):

		saveMarker=""
		COData,COHead=myFITS.readFITS( CO12FITS )

		if useMask:
			DBMaskData,_=  myFITS.readFITS(  DBMaskFITS )

			maskData=np.zeros_like( DBMaskData )

			maskData[COData>min_sigma*self.rms]=1
			maskData[DBMaskData==0]=0
			saveLabel= "LabelSigma_{}_P{}.fits".format( min_sigma,min_pix )

		else:

			#use peak sigma to grow a mask

			maskData=np.zeros_like( COData )
			maskData[COData>min_sigma*self.rms]=1


			saveLabel= "NoMaskLabelSigma_{}_P{}.fits".format( min_sigma,min_pix )

			saveMarker="growMask"

		labels=measure.label(maskData,connectivity=1)
		fits.writeto(  saveLabel, labels,header=COHead, overwrite=True)


		if calCat  :

			self.getCatFromLabelArray(COFITS,saveLabel,self.TBModel,saveMarker=saveMarker,  minPix=min_pix,rms= min_sigma  )


	def getCatFromLabelArray(self,  CO12FITS,labelFITS,TBModel,minPix=8,rms=2 ,saveMarker="", peakSigma=3,region=""):
		"""
		Extract catalog from
		:param labelArray:
		:param head:
		:return:
		"""

		#do not make any selection here, because of the closeness of many clouds, some cloud may have pixels less than 8, we should keep them
		#they are usefull to mask edge sources..., and to clean fits

		if saveMarker=="":

			saveName= region+"DBSCAN{}_P{}Cat.fit".format(rms,minPix)

		else:
			saveName=saveMarker+".fit"

		clusterTBOld=Table.read( TBModel )

		###
		dataCO, headCO = myFITS.readFITS( CO12FITS )

		#dataCO=np.nan_to_num(dataCO), should not have Nan values


		dataCluster , headCluster=myFITS.readFITS( labelFITS )

		minV=np.nanmin(dataCluster)
		wcsCloud=WCS( headCluster )

		clusterIndex1D= np.where( dataCluster>minV )
		clusterValue1D=  dataCluster[clusterIndex1D ]

		Z0,Y0,X0 = clusterIndex1D

		newTB= Table( clusterTBOld[0])
		newTB["sum"]=newTB["flux"]

		newTB["l_rms"]=newTB["v_rms"]
		newTB["b_rms"]=newTB["v_rms"]

		newTB["pixN"]=newTB["v_rms"]
		newTB["peak"]=newTB["v_rms"]
		newTB["peak2"]=newTB["v_rms"]  #the second largest peak
		dataClusterNew=np.zeros_like( dataCluster)

		# in the newCluster, number stars from 1, not zero

		idCol="_idx"



		#count all clusters

		#ids,count=np.unique(dataCluster,return_counts=True )
		ids,count=np.unique(clusterValue1D,return_counts=True )

		#GoodIDs=  ids[count>=minPix ]

		#GoodCount = count[ count>=minPix  ]

		GoodIDs=ids
		GoodCount=count
		print "Total number of turnks? ",len(ids)
		#print "Total number of Good Trunks? ",len(GoodIDs)

		#dataCO,headCO=doFITS.readFITS( CO12FITS )
		widgets = ['Recalculating cloud parameters: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),  ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

		pbar = ProgressBar(widgets=widgets, maxval=len(GoodIDs))
		pbar.start()

		catTB=newTB.copy()
		catTB.remove_row(0)



		for i in  range(len(ids)) :

			#i would be the newID
			newID= GoodIDs[i]

			if newID==minV: #this value is the masked values, for DBSCAN, is 0, and for dendrogram is 1.
				continue

			pixN=GoodCount[i]

			newRow=newTB[0]


			newRow[idCol] = newID

			cloudIndex=self.getIndices(Z0,Y0,X0,clusterValue1D,newID)

			coValues=  dataCO[ cloudIndex ]


			sortedCO=np.sort(coValues)
			peak = sortedCO[-1] #np.max( coValues)
			peak2=sortedCO[-2]
			cloudV=cloudIndex[0]
			cloudB=cloudIndex[1]
			cloudL=cloudIndex[2]



			sumCO=np.sum( coValues )

			Vcen,Vrms= weighted_avg_and_std(cloudV, coValues )
			Bcen,Brms= weighted_avg_and_std(cloudB, coValues )
			Lcen,Lrms= weighted_avg_and_std(cloudL, coValues )

			#calculate the exact area

			LBcore=zip(  cloudB ,    cloudL   )

			pixelsN= {}.fromkeys(LBcore).keys() #len( set(LBcore) )
			area_exact=len(pixelsN)*0.25 #arc mins square


			#dataClusterNew[cloudIndex] =newID

			#save values
			newRow["pixN"]= pixN
			newRow["peak"]= peak
			newRow["peak2"]= peak2

			newRow["sum"]= sumCO
			newRow["area_exact"]= area_exact

			newRow["x_cen"],  newRow["y_cen"], newRow["v_cen"]= wcsCloud.wcs_pix2world( Lcen, Bcen,Vcen ,0)
			newRow["v_cen"]= newRow["v_cen"]/1000.
			dv=headCluster["CDELT3"]/1000. #km/s

			dl= abs( headCluster["CDELT1"] ) #deg

			newRow["v_rms"] = Vrms*dv

			newRow["l_rms"] = Lrms*dl
			newRow["b_rms"] = Brms*dl

			catTB.add_row(newRow)

			pbar.update(i)



		pbar.finish()
		#save the clouds

		#fits.writeto(self.regionName+"NewCloud.fits", dataClusterNew,header=headCluster,overwrite=True   )
		catTB.write( saveName ,overwrite=True)

	def getSumToFluxFactor(self):

		theta =  np.deg2rad(0.5/60)
		omega = theta * theta
		f=115.271202000
		waveLength =299792458/(f*1e9)
		k= 1.38064852e3 #has converted to jansky
		factorSumToFlux=  2*k*omega/waveLength/waveLength
		
		return factorSumToFlux
	def converSumToFlux(self,sumRow):

		factorSumToFlux=self.getSumToFluxFactor(factorSumToFlux)

		return sumRow* factorSumToFlux #jansky

	def converFluxToSum(self, fluxRow):
		factorSumToFlux=self.getSumToFluxFactor(factorSumToFlux)

		return fluxRow/factorSumToFlux



	def getIndices(self,Z0,Y0,X0,values1D,choseID):


		cloudIndices = np.where(values1D==choseID )

		cX0=X0[cloudIndices ]
		cY0=Y0[cloudIndices ]
		cZ0=Z0[cloudIndices ]

		return tuple( [ cZ0, cY0, cX0 ]  )



	def getIndices2D(self, Y0,X0,values1D,choseID):



		cloudIndices = np.where(values1D==choseID )

		cX0=X0[cloudIndices ]
		cY0=Y0[cloudIndices ]


		return tuple( [   cY0, cX0 ]  )




	def draw(self ):
		"""
		#draw compare of
		:return:
		"""



		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		drawTB=Table.read( "Sigma1_P25FastDendro.fit" )


		axNumber=fig.add_subplot(1,2,1)




		axArea= fig.add_subplot(1,2,2)

		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )


		totalTB=  [drawTB] #TBList1+TBList2

		for i in range( len(totalTB) ):

			eachTB = totalTB[i]

			binN,binEdges=np.histogram(eachTB["area_exact"]/3600., bins=areaEdges  )


			axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8  ,alpha= 0.5 )


		axArea.set_yscale('log')
		axArea.set_xscale('log')


		axArea.legend()



		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Number of trunks")


		plt.savefig( "compareDendroParaDBMask.pdf" ,  bbox_inches='tight')
		plt.savefig( "compareDendroParaDBMask.png" ,  bbox_inches='tight',dpi=300)


	def getEdgeCenter(self,edges):

		areaCenters= ( edges[1:] + edges[0:-1] )/2.

		return  areaCenters

	def cleanDBTB(self,dbTB,pixN=8,minV=3,minDelta=3):

		"""
		The minimum Peak, should be relative to the minValue
		:param dbTB:
		:param pixN:
		:param peak:
		:return:
		"""
		peakV=(minV + minDelta)*self.rms



		if type(dbTB)==list:
			newList=[]
			for eachT in dbTB:

				goodT=eachT.copy()
				goodT=goodT[ goodT["pixN"] >= pixN ]
				goodT=goodT[ goodT["peak"] >= peakV ]

				newList.append(goodT)

			return newList



		else:

			goodT=dbTB.copy()
			goodT=goodT[ goodT["pixN"] >= pixN ]
			goodT=goodT[ goodT["peak"] >= peakV ]

			return goodT





	def drawDBSCANArea(self):

		TB2_16= "G2650CO12DBCatS2P16Con2.fit"
		#TB2_16= "DBSCAN2_9Sigma1_P1FastDBSCAN.fit"
		TB25_9="G2650CO12DBCatS2.5P9Con2.fit"
		TB35_9="G2650CO12DBCatS3.5P9Con2.fit"
		TB45_9="G2650CO12DBCatS4.5P9Con2.fit"
		TB55_9="G2650CO12DBCatS5.5P9Con2.fit"
		TB65_9="G2650CO12DBCatS6.5P9Con2.fit"
		TB75_9="G2650CO12DBCatS7.5P9Con2.fit"

		TB3_9= "G2650CO12DBCatS3.0P9Con2.fit"
		TB4_9= "G2650CO12DBCatS4.0P9Con2.fit"
		TB5_9= "G2650CO12DBCatS5.0P9Con2.fit"
		TB6_9= "G2650CO12DBCatS6.0P9Con2.fit"
		TB7_9= "G2650CO12DBCatS7.0P9Con2.fit"



		TBFiles=[TB2_16,TB25_9,TB3_9, TB35_9, TB4_9, TB45_9,TB5_9, TB55_9, TB6_9 , TB65_9, TB7_9, TB75_9   ]



		sigmas=[2,2.5,3,3.5,4,4.5, 5, 5.5, 6, 6.5,7,7.5]

		labelStr=[  r"2$\sigma$, P16" ,   r"2.5$\sigma$, P16" ,  r"3$\sigma$, P16" ,  r"3.5$\sigma$, P16" ,   r"4$\sigma$, P16" ,  r"4.5$\sigma$, P16" , \
		            r"5$\sigma$, P16" ,  r"5.5$\sigma$, P16" , r"6$\sigma$, P16"  , r"6.5$\sigma$, P16"  , r"7$\sigma$, P16", r"7.5 $\sigma$, P16"   ]


		TBList=[]



		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )
		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })



		axArea=fig.add_subplot(1,2,1)



		for eachTBF,eachLab in zip(TBFiles,labelStr):
			tb=Table.read(eachTBF)

			tb=self.removeWrongEdges(tb)

			TBList.append( tb )


			goodT=tb


			goodT=goodT[ goodT["pixN"]>=16 ]

			goodT=goodT[ goodT["peak"]>=1.5 ]

			#
			binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )




			axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,label=eachLab ,alpha= 0.5 )


		axArea.set_yscale('log')
		axArea.set_xscale('log')

		axArea.legend()
		axArea.set_title("Plot of Area distribution with DBSCAN")

		plt.savefig( "dbscanArea.png" ,  bbox_inches='tight',dpi=300)


	def drawDBSCANNumber(self):

		minPix=8

		TB2_16="G2650CO12DBCatS2.0P{}Con2.fit".format(minPix)
		TB25_9="G2650CO12DBCatS2.5P{}Con2.fit".format(minPix)
		TB35_9="G2650CO12DBCatS3.5P{}Con2.fit".format(minPix)
		TB45_9="G2650CO12DBCatS4.5P{}Con2.fit".format(minPix)
		TB55_9="G2650CO12DBCatS5.5P{}Con2.fit".format(minPix)
		TB65_9="G2650CO12DBCatS6.5P{}Con2.fit".format(minPix)
		TB75_9="G2650CO12DBCatS7.5P{}Con2.fit".format(minPix)

		TB3_9= "G2650CO12DBCatS3.0P{}Con2.fit".format(minPix)
		TB4_9= "G2650CO12DBCatS4.0P{}Con2.fit".format(minPix)
		TB5_9= "G2650CO12DBCatS5.0P{}Con2.fit".format(minPix)
		TB6_9= "G2650CO12DBCatS6.0P{}Con2.fit".format(minPix)
		TB7_9= "G2650CO12DBCatS7.0P{}Con2.fit".format(minPix)


		TBFiles=[TB2_16,TB25_9,TB3_9, TB35_9, TB4_9, TB45_9,TB5_9, TB55_9, TB6_9 , TB65_9, TB7_9, TB75_9   ]
		TBList=[]

		Nlist=[]

		sigmas=[2,2.5,3,3.5,4,4.5, 5, 5.5, 6, 6.5,7,7.5]

		for eachTBF in TBFiles:
			tb=Table.read(eachTBF)
			TBList.append( tb )
			goodT=tb
			goodT=goodT[ goodT["pixN"]>=16 ]

			goodT=goodT[ goodT["peak"]>=1.5 ]

			Nlist.append(len(goodT) )

		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		drawTB=Table.read( "Sigma1_P25FastDendro.fit" )


		axNumber=fig.add_subplot(1,2,1)

		axNumber.plot(sigmas,Nlist,'o-',color='blue')

		axNumber.set_ylabel(r"Total number of trunks")
		axNumber.set_xlabel(r"CO cutoff ($\sigma$)")
		axNumber.set_title("Plot of total trunk numbers with DBSCAN")

		plt.savefig( "dbscanNumber.png" ,  bbox_inches='tight',dpi=300)

	def getRealArea(self,TB):
		#print TB.colnames
		araeList=[]

		#print TB.colnames
		for eachR in TB:
			v = eachR["v_cen"]
			dis= ( 0.037*v + 0.0249)*1000 # pc

			if dis<0  or dis> 1500 :
				continue

			N=eachR["area_exact"]/0.25


			#print N,eachR["pixN"]
			length= dis*np.deg2rad(0.5/60) # square pc
			trueArea=length**2*N #eachR["pixN"]  #*10000
			#print N,  trueArea

			araeList.append(  trueArea )

		return np.asarray(araeList)



	def drawAreaDistribute(self,TBName,region="",algorithm='Dendrogram'):
		"""

		:return:
		"""

		TB=Table.read( TBName )

		TBLOcal=Table.read("DBSCAN35_9Sigma1_P1FastDBSCAN.fit")
		TBAll=vstack([TB,TBLOcal ])

		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )



		fig=plt.figure(figsize=(8,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })
		axArea=fig.add_subplot(1,1,1)

		##########
		goodT=TB

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>= self.rms*5. ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5, label= region)#   r"SCIMES,min3$\sigma$P16 12CO"  )
		print region
		self.getAlphaWithMCMC(  goodT["area_exact"] , minArea= 0.00065, maxArea=None , physicalArea=False )

		#a=np.linspace(1,3000,6000)

		#trueArea=1./a**2
		#self.getAlphaWithMCMC(  trueArea , minArea= 1e-7, maxArea=None , physicalArea=True )

		#print "Above???-2??"


		###############
 		goodT=TBLOcal

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Velocity range (0-30 km/s)12CO Raw"  )


		areaEdges=np.linspace(0,100,1000)
		areaCenter=self.getEdgeCenter( areaEdges )



		realArea=self.getRealArea(goodT)
		binN,binEdges=np.histogram( realArea  , bins=areaEdges  )

		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Velocity range (0-30 km/s)12CO, distance Corrected"  )
		print min(realArea),"The minimum area?"
		self.getAlphaWithMCMC(  realArea  ,minArea= 0.42836824657505895 , maxArea=None,  physicalArea=True)

		areaEdges=np.linspace(0,6,1000)
		areaCenter=self.getEdgeCenter( areaEdges )


		############### Perseus
 		goodT=  Table.read("Local13DBSCAN3_9.fit")

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label=r"(26$^\circ$-50$^\circ$)13CO"  )


		############### Perseus
 		goodT=  Table.read("G210DBSCAN3_9.fit")

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label=r"G210(210$^\circ$-220$^\circ$)12CO"  )

		###############
 		goodT=  Table.read("DBSCAN3_9.fit")

		if "pixN" in goodT.colnames:

			goodT=goodT[ goodT["pixN"]>=16 ]
			goodT=goodT[ goodT["peak"]>=1.5 ]
		binN,binEdges=np.histogram(goodT["area_exact"]/3600., bins=areaEdges  )
		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Velocity range (30-60 km/s)12CO"  )









		###############

		axArea.set_yscale('log')
		axArea.set_xscale('log')
		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Bin number of trunks ")



		axArea.legend()
		axArea.set_title("Plot of Area distribution with DBSCAN")

		plt.savefig( region+"dbscanArea.png" ,  bbox_inches='tight',dpi=300)
		plt.savefig( region+"dbscanArea.pdf" ,  bbox_inches='tight' )


	def getAlphaWithMCMC(self,areaArray,minArea=0.03,maxArea=1.,physicalArea=False,verbose=True,plotTest=False,saveMark="" ):
		"""
		areaArray should be in square armin**2
		:param areaArray:
		:param minArea:
		:param maxArea:
		:return:
		"""

		print "Fitting index with MCMC..."

		if not physicalArea:
			areaArray=areaArray/3600.

		if maxArea!=None:
			select=np.logical_and( areaArray>minArea, areaArray<maxArea)

		else:
			select= areaArray>minArea

		rawArea =   areaArray[ select ]

		if verbose:
			print "Run first chain for {} molecular clouds.".format( len( rawArea ) )
		part1=doG210.fitPowerLawWithMCMCcomponent1(rawArea, minV=minArea, maxV=maxArea)
		if verbose:
			print "Run second chain for {} molecular clouds.".format( len(rawArea) )

		part2=doG210.fitPowerLawWithMCMCcomponent1(rawArea, minV=minArea, maxV=maxArea)

		allSample=np.concatenate(  [ part1 , part2 ]    )



		#test plot
		if plotTest:
			fig = plt.figure(figsize=(12, 6))
			ax0 = fig.add_subplot(1, 1, 1)
			# fig, axs = plt.subplots(nrows=1, ncols=2,  figsize=(12,6),sharex=True)
			rc('text', usetex=True)
			rc('font', **{'family': 'sans-serif', 'size': 13, 'serif': ['Helvetica']})

			ax0.scatter(part1,part2,s=10 )

			plt.savefig("mcmcSampleTest.pdf"  , bbox_inches='tight')
			aaaaaa

		meanAlpha= np.mean( allSample)
		stdAlpha=  np.std(allSample,ddof=1)
		if verbose:
			print "Alpha Mean: {:.2f}; std: {:.2f}".format( meanAlpha,  stdAlpha)

		return round(meanAlpha,2) , round(stdAlpha,2)

	def drawSumDistribute(self,TBName,region=""):
		"""
		:return:
		"""

		TB=Table.read( TBName )

		TBLOcal=Table.read("DBSCAN35_9Sigma1_P1FastDBSCAN.fit")
		TBAll=vstack([TB,TBLOcal ])



		goodT=TB





		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })
		axArea=fig.add_subplot(1,1,1)

		##########
		pixNCol =goodT["flux"]

		logPixN=np.log10( pixNCol  )

		print min( logPixN  ),max( logPixN  )

		areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,100)
		areaCenter=self.getEdgeCenter( areaEdges )

		binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


		drawBind=binN[binN>0]
		drawCenter= areaCenter[binN>0]

		axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Flux"  )

		select=np.logical_and( drawCenter<6, drawCenter>4 )
		x=drawCenter[ select ]   #np.log(drawCenter)
		y=np.log10(drawBind  )[ select]

		#print np.polyfit(x,y,1)

		###########################



		if 0:
			tbVox=goodT[ goodT["pixN"]>16  ]
			pixNCol =tbVox["pixN"]

			logPixN=np.log10( pixNCol  )

			print min( logPixN  ),max( logPixN  )

			areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,100)
			areaCenter=self.getEdgeCenter( areaEdges )

			binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


			drawBind=binN[binN>0]
			drawCenter= areaCenter[binN>0]

			axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="Voxel"  )

			select=np.logical_and( drawCenter<4, drawCenter>1.5 )
			x=drawCenter[ select ]   #np.log(drawCenter)
			y=np.log10(drawBind  )[ select]

			print np.polyfit(x,y,1)

		###########################




		##########
		tbVox=goodT
		pixNCol =tbVox["area_exact"]

		logPixN=np.log10( pixNCol  )
		print "draw areas"
		print min( logPixN  ),max( logPixN  )

		areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,15)

		print  areaEdges

		areaCenter=self.getEdgeCenter( areaEdges )

		binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


		drawBind=binN[binN>0]
		drawCenter= areaCenter[binN>0]

		axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label="area exact aa "  )

		select=np.logical_and( drawCenter<4, drawCenter>1.5  )


		x=drawCenter[ select ]   #np.log(drawCenter)
		y=np.log10(drawBind  )[ select]

		a= np.polyfit(x,y,1)
		p=np.poly1d(a)
		axArea.plot(  drawCenter ,  p(drawCenter) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5   )

		print a
		###########################








		axArea.set_xlabel(r"Voxel Number")
		axArea.set_ylabel(r"Bin number of trunks ")



		axArea.legend()
		axArea.set_title("Plot of Pixel distribution with DBSCAN")

		plt.savefig( region+"dbscanTotalPixel.png" ,  bbox_inches='tight',dpi=300)





	def drawPixNDistribute(self,TBName,region=""):
		"""

		:return:
		"""

		TB=Table.read( TBName )

		TBLOcal=Table.read("DBSCAN35_9Sigma1_P1FastDBSCAN.fit")
		TBAll=vstack([TB,TBLOcal ])



		goodT=TB
		goodT=goodT[ goodT["pixN"]>=16 ]

		pixNCol =goodT["pixN"]

		logPixN=np.log10( pixNCol  )

		print logPixN



		areaEdges=np.linspace( min( logPixN  ),max( logPixN  ) ,100)
		areaCenter=self.getEdgeCenter( areaEdges )



		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })
		axArea=fig.add_subplot(1,1,1)

		##########

		binN,binEdges=np.histogram( logPixN , bins=areaEdges  )


		drawBind=binN[binN>0]
		drawCenter= areaCenter[binN>0]

		axArea.plot(  drawCenter , np.log10(drawBind) , 'o-'  , markersize=1, lw=0.8,  alpha= 0.5 ,label=" ??"  )

		select=np.logical_and( drawCenter<4, drawCenter>1.5 )
		x=drawCenter[ select ]   #np.log(drawCenter)
		y=np.log(drawBind  )[ select]

		print np.polyfit(x,y,1)

		###############


		axArea.set_xlabel(r"Voxel Number")
		axArea.set_ylabel(r"Bin number of trunks ")



		axArea.legend()
		axArea.set_title("Plot of Pixel distribution with DBSCAN")

		plt.savefig( region+"dbscanTotalPixel.png" ,  bbox_inches='tight',dpi=300)





	def roughFit(self,centers,bins ):
		"""

		:return:
		"""
		y= bins[bins>0 ]  # areaCenter[binN>0]
		x=  centers[ bins>0  ]




		x1= x[x<= 0.1]  # areaCenter[binN>0]
		y1=  y[x<= 0.1 ]


		x2= x1[x1>=0.005 ]  # areaCenter[binN>0]
		y2=  y1[x1>= 0.005 ]

		x=np.log10(x2)
		y=np.log10(y2)


		print x
		print y

		print np.polyfit(x,y,1)


		return




	def drawTrueArea(self):

		goodTB=Table.read( "/home/qzyan/WORK/projects/maddalena/dendroDisPath/G2650/G2650goodDisTB.fit"  )
		dendroTB=Table.read( "/home/qzyan/WORK/myDownloads/testScimes/mosaicV1NewTB.fit" )

		areas=[]

		for eachG in goodTB:

			d=eachG["distance"]
			ID=int( eachG["sourceName"].split('oud')[1]  )
			dendroRow=  dendroTB[ID-1 ]

			area=dendroRow["area_exact"]

			area/0.25*(d*np.radians( 0.5/60. ) )**2
			#print area


			areas.append(area )
		#plot
		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		drawTB=Table.read( "Sigma1_P25FastDendro.fit" )

		bins=np.linspace( np.min(areas),np.max(areas),5  )
		areaCenter=self.getEdgeCenter( bins )

		ax=fig.add_subplot(1,2,1)

		binN,binEdges=np.histogram(areas, bins=bins  )

		ax.scatter(areaCenter,binN   )

		ax2=fig.add_subplot(1,2,2)


		ax2.scatter( goodTB["vlsr"], goodTB["distance"]  )


		plt.savefig( "exactArea.png" ,  bbox_inches='tight',dpi=300)




	def removeWrongEdges(self,TB):



		processTB=TB.copy()

		#remove cloudsThat touches the noise edge of the fits


		#part1= processTB[ np.logical_and( processTB["x_cen"]>=2815 ,processTB["y_cen"]>= 1003  )   ] #1003, 3.25

		#part2= processTB[ np.logical_and( processTB["x_cen"]<= 55 ,processTB["y_cen"]>= 1063  )   ] #1003, 3.25

		if "peak" in TB.colnames: #for db scan table

			part1= processTB[ np.logical_or( processTB["x_cen"]>26.25 ,processTB["y_cen"] < 3.25  )   ] #1003, 3.25

			part2= part1[ np.logical_or( part1["x_cen"]<49.25 ,part1["y_cen"]<  3.75 )   ] #1003, 3.25

			return part2
		else: #dendrogram tb

			part1= processTB[ np.logical_or( processTB["x_cen"]< 2815 ,processTB["y_cen"] < 1003  )   ] #1003, 3.25

			part2= part1[ np.logical_or( part1["x_cen"]>  55 ,part1["y_cen"]< 1063  )   ] #1003, 3.25

			return part2



	def removeAllEdges(self,TBList):
		"""

		:param TBList:
		:return:
		"""
		newList=[]

		for eachTB in TBList:
			newList.append( self.removeWrongEdges(eachTB) )

			
		return newList

	def getNList(self,TBList):

		Nlist=[]

		for eachTB in TBList:
			Nlist.append( len(eachTB) )
		return Nlist



	def getTotalFluxList(self,TBList):

		fluxlist=[]
		omega =1  # np.deg2rad(0.5 / 60) * np.deg2rad(0.5 / 60) / 4. / np.log(2)

		for eachTB in TBList:

			if "sum" in eachTB.colnames:
				toalFlux=np.nansum( eachTB["sum"]  )*0.2*omega # K km/s

			else:
				toalFlux=np.nansum( eachTB["flux"]  )*0.2/self.getSumToFluxFactor()*omega # K km/s

			fluxlist.append(  toalFlux  )
		return fluxlist

	def fluxAlphaDistribution(self):
		"""

		# draw alph distribution of flux, for each DBSCAN,

		:return:
		"""

		algDendro = "Dendrogram"
		tb8Den, tb16Den, label8Den, label16Den, sigmaListDen = self.getTBList(algorithm=algDendro)
		tb8Den = self.removeAllEdges(tb8Den)
		tb16Den = self.removeAllEdges(tb16Den)

		algDB = "DBSCAN"
		tb8DB, tb16DB, label8DB, label16DB, sigmaListDB = self.getTBList(algorithm=algDB)
		tb8DB = self.removeAllEdges(tb8DB)
		tb16DB = self.removeAllEdges(tb16DB)


		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)
		






		fig = plt.figure(figsize=(8, 6))
		rc('text', usetex=True)
		rc('font', **{'family': 'sans-serif', 'size': 13, 'serif': ['Helvetica']})

		#############   plot dendrogram
		axDendro = fig.add_subplot(1, 1, 1)

		alphaDendro, alphaDendroError = self.getFluxAlphaList(tb16Den, sigmaListDen)

		ebDendro=axDendro.errorbar(sigmaListDen, alphaDendro, yerr=alphaDendroError, c='g', marker='D', capsize=3 , elinewidth=1.3, lw=1, label=algDendro)



		axDendro.set_ylabel(r"$\alpha$ (flux)")
		axDendro.set_xlabel(r"CO cutoff ($\sigma$)")

		##############   plot DBSCAN

		alphaDB, alphaDBError =   self.getFluxAlphaList(tb16DB, sigmaListDB)
		ebDBSCAN = axDendro.errorbar(sigmaListDB, alphaDB, yerr=alphaDBError, c='b', marker='^',linestyle=":" , capsize=3 , elinewidth=1.0, lw=1, label=algDB ,  markerfacecolor='none' )

		ebDBSCAN[-1][0].set_linestyle(':')

		##########plot SCIMES

		alphaSCI, alphaDBError =   self.getFluxAlphaList(tb16SCI, sigmaListSCI)
		ebSCISCAN = axDendro.errorbar(sigmaListSCI, alphaSCI, yerr=alphaDBError, c='purple', marker='s',linestyle="--", capsize=3 , elinewidth=1.0, lw=1, label=algSCI,  markerfacecolor='none' )

		ebSCISCAN[-1][0].set_linestyle('--')


		if 1: #plot average alpha
			allAlpha = alphaDB + alphaDendro
			allAlphaError = alphaDBError + alphaDendroError

			print "Average error, ", np.mean(allAlphaError)

			errorAlpha = np.mean(allAlphaError) ** 2 + np.std(allAlpha, ddof=1) ** 2
			errorAlpha = np.sqrt(errorAlpha)

			alphaMean = np.mean(allAlpha)

			print "The mean alpha of flux distribution is {:.2f}, error is {:.2f}".format(alphaMean, errorAlpha)

			axDendro.plot([min(sigmaListDB), max(sigmaListDB)], [alphaMean, alphaMean], '--', color='black', lw=1)

		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )


		axDendro.legend(loc=1)

		fig.tight_layout()
		plt.savefig("compareParaFluxAlpha.pdf", bbox_inches='tight')
		plt.savefig("compareParaFluxAlpha.png", bbox_inches='tight', dpi=300)

	def alphaDistribution(self):
		"""

		# draw alph distribution, for each DBSCAN,

		:return:
		"""

		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)

		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)


		fig=plt.figure(figsize=(8, 6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#############   plot dendrogram
		axDendro=fig.add_subplot(1,1,1)

		alphaDendro , alphaDendroError = self.getAlphaList( tb16Den )

		axDendro.errorbar(sigmaListDen, alphaDendro, yerr= alphaDendroError , c='g', marker='D', capsize=3, elinewidth=1.3, lw=1,label= algDendro  )


		axDendro.set_ylabel(r"$\alpha$ (area)")
		axDendro.set_xlabel(r"CO cutoff ($\sigma$)")

 


		##############   plot DBSCAN

		#alphaDB,  alphaDBError = self.drawAlpha( axDB,tb8DB,tb16DB, label8DB,  label16DB ,sigmaListDB)

		alphaDB , alphaDBError = self.getAlphaList( tb16DB )

		ebDBSCAN=axDendro.errorbar(sigmaListDB, alphaDB, yerr= alphaDBError , c='b', marker='^',linestyle=":", capsize=3, elinewidth=1.0, lw=1,label= algDB , markerfacecolor='none' )
		ebDBSCAN[-1][0].set_linestyle(':')

		##########plot SCIMES

		alphaSCI , alphaSCIError = self.getAlphaList( tb16SCI )

		ebSCISCAN=axDendro.errorbar(sigmaListSCI, alphaSCI, yerr= alphaSCIError , c='purple', marker='s', linestyle="--", capsize=3, elinewidth=1.0, lw=1,label= algSCI,  markerfacecolor='none' )
		ebSCISCAN[-1][0].set_linestyle('--')
		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )

		#plot average
		allAlpha = alphaDB+alphaDendro
		allAlphaError = alphaDBError+alphaDendroError

		print "Average error, ",  np.mean(allAlphaError )

		errorAlpha= np.mean(allAlphaError )**2+  np.std(allAlpha,ddof=1)**2
		errorAlpha=np.sqrt( errorAlpha )

		alphaMean= np.mean( allAlpha)

		print "The mean alpha is {:.2f}, error is {:.2f}".format(alphaMean,errorAlpha )

		axDendro.plot([min(sigmaListDB),max(sigmaListDB)],  [alphaMean, alphaMean],'--', color='black', lw=1 )

		axDendro.legend(loc=1 )

		fig.tight_layout()
		plt.savefig( "compareParaAlpha.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaAlpha.png"  ,  bbox_inches='tight',dpi=300)






	def fluxDistribution(self):
		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)



		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)







		compoleteFluxa= 9*(1500./250)**2*0.2*1.5 # 2 sigma


		#aaaaaa

		fig=plt.figure(figsize=(20,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,3,1)



		at = AnchoredText(algDendro, loc=3, frameon=False)
		axDendro.add_artist(at)

		self.drawFlux(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)

		strXlabel=  r"Flux ($\rm K\ km\ s$$^{-1}$ $\Omega_{\rm A}$)"

		axDendro.set_xlabel( strXlabel )
		axDendro.set_ylabel(r"Number of trunks")

		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )

		axDendro.set_yscale('log')
		axDendro.set_xscale('log')
		#axDendro.plot( [compoleteFluxa,compoleteFluxa],[2,800],'--',color='black', lw=1  )



		axDendro.legend(loc=1, ncol=2 )
		#plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro )

		self.drawFlux(axDB,tb8DB,tb16DB, label8DB,label16DB, sigmaListDB)


		at = AnchoredText(algDB, loc=3, frameon=False)
		axDB.add_artist(at)

		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )

		axDB.set_xlabel( strXlabel )
		#axDB.set_ylabel(r"Bin number of trunks ")

		axDB.set_yscale('log')
		axDB.set_xscale('log')

		axDB.legend(loc=1, ncol=2 )

		axDB.set_ylabel(r"Number of trunks")



		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )

		#plot SCIMES

		axSCI=fig.add_subplot(1,3,3,sharex=axDendro,sharey=axDendro )

		self.drawFlux(axSCI,tb8SCI,tb16SCI, label8SCI,label16SCI, sigmaListSCI )


		at = AnchoredText(algSCI, loc=3, frameon=False)
		axSCI.add_artist(at)

		axSCI.set_xlabel( strXlabel )
		#axSCI.set_ylabel(r"Bin number of trunks ")

		axSCI.set_yscale('log')
		axSCI.set_xscale('log')

		axSCI.legend(loc=1, ncol=2 )

		axSCI.set_ylabel(r"Number of clusters")



		fig.tight_layout()
		plt.savefig( "compareParaFlux.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaFlux.png"  ,  bbox_inches='tight',dpi=300)







	def areaDistribution(self):

		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)

		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)



		#aaaaaa

		fig=plt.figure(figsize=(18,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,3,1)

		#self.drawNumber(axDendro,tb8Den,tb16Den,sigmaListDen)
		at = AnchoredText(algDendro, loc=3, frameon=False)
		axDendro.add_artist(at)

		self.drawArea(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)

		axDendro.set_xlabel(r"Angular area (deg$^2$)")
		axDendro.set_ylabel(r"Number of trunks")


		axDendro.set_yscale('log')
		axDendro.set_xscale('log')

		compoleteArea= 9*(1500./250)**2*0.25/3600. #0.0225




		axDendro.plot( [compoleteArea,compoleteArea],[2,2000],'--',color='black', lw=1  )


		axDendro.legend(loc=1, ncol=2 )
		#plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro )
		#self.drawNumber(axDB,tb8DB,tb16DB,sigmaListDB)
		self.drawArea(axDB,tb8DB,tb16DB, label8DB,label16DB, sigmaListDB)


		at = AnchoredText(algDB, loc=3, frameon=False)
		axDB.add_artist(at)
		axDB.plot( [compoleteArea,compoleteArea],[2,2000],'--',color='black', lw=1  )
		axDB.set_xlabel(r"Angular area (deg$^2$)")
		#axDB.set_ylabel(r"Bin number of trunks ")
		axDB.set_ylabel(r"Number of trunks")

		axDB.set_yscale('log')
		axDB.set_xscale('log')

		axDB.legend(loc=1, ncol=2 )

		#plot SCIMES

		axSCI=fig.add_subplot(1,3,3,sharex=axDendro,sharey=axDendro )
		#self.drawNumber(axDB,tb8DB,tb16DB,sigmaListDB)
		self.drawArea(axSCI,tb8SCI,tb16SCI, label8SCI,label16SCI, sigmaListSCI )


		at = AnchoredText(algSCI, loc=3, frameon=False)
		axSCI.add_artist(at)
		axSCI.plot( [compoleteArea,compoleteArea],[2,2000],'--',color='black', lw=1  )
		axSCI.set_xlabel(r"Angular area (deg$^2$)")
		#axSCI.set_ylabel(r"Bin number of trunks ")
		axSCI.set_ylabel(r"Number of clusters")

		axSCI.set_yscale('log')
		axSCI.set_xscale('log')

		axSCI.legend(loc=1, ncol=2 )

		########




		fig.tight_layout()
		plt.savefig( "compareParaArea.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaArea.png"  ,  bbox_inches='tight',dpi=300)




	def totaFluxDistribution(self):
		"""
		Compare the change of molecular cloud numbers with
		:return:
		"""
		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)


		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)




		fig=plt.figure(figsize=(8,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 10,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,1,1)

		#self.drawTotalFlux(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)

		#Nlist8Den=self.getTotalFluxList(tb8List)
		fluxList16Den=self.getTotalFluxList(tb16Den)
		axDendro.plot(sigmaListDen,fluxList16Den,'D-' , color="green",markersize=4, lw=1.0,label="Total flux (dendrogram)" )


		axDendro.set_ylabel(r"Total flux ($\rm K\ km\ s$$^{-1}$ $\Omega_\mathrm{A}$)")
		axDendro.set_xlabel(r"CO cutoff ($\sigma$)")



		#plot DBSCAN
		fluxList16DB=self.getTotalFluxList( tb16DB )

		axDendro.plot(sigmaListDB,fluxList16DB,'^--' ,linestyle=':', color="blue",markersize=3, lw=1.0,label= "Total flux (DBSCAN)", markerfacecolor='none' )


		#plot SCIMES
		fluxList16SCI=self.getTotalFluxList( tb16SCI )

		axDendro.plot(sigmaListSCI,fluxList16SCI,'s--' ,  color="purple",markersize=3, lw=1.0,label= "Total flux (SCIMES)", markerfacecolor='none' )



		#drawTotal Flux

		#axDendro.set_yscale('log')
		maskCO,_=myFITS.readFITS("G2650CO12MaskedCO.fits")
		totalFluxList=[]

		omega = 1 #np.deg2rad(0.5 / 60) * np.deg2rad(0.5 / 60) / 4. / np.log(2)

		for eachS in sigmaListDen:
			maskCO[maskCO<eachS*self.rms]=0
			sumCO= np.sum( maskCO)*0.2* omega
			totalFluxList.append( sumCO )

		totalFluxRatioDendro=np.asarray( fluxList16Den ) /np.asarray( totalFluxList  )
		totalFluxRatioDB=np.asarray( fluxList16DB ) /np.asarray( totalFluxList  )

		totalFluxRatioSCI=np.asarray( fluxList16SCI ) /np.asarray( totalFluxList  )

		tabBlue='tab:blue'
		axDendro.plot(sigmaListDB,totalFluxList,'o-' ,   color="black",markersize=3, lw=1.0,label= "Total flux(cutoff mask)", markerfacecolor='none' )

		z=np.polyfit(sigmaListDB, totalFluxList ,1 )

		p=np.poly1d(  z )
		print z
		print p(0.0)/p(3), p(0.0)/p(2)
		print p(3)/p(0), p(2)/p(0)

		#axDendro.plot(sigmaListDB, p( sigmaListDB )  ,'r-' ,   color="red",markersize=3, lw=1.0,label= "Total flux(cutoff mask)", markerfacecolor='none' )


		axRatio = axDendro.twinx()  # instantiate a second axes that shares the same x-axis

		axRatio.plot(sigmaListDen, totalFluxRatioDendro,'D-',color=tabBlue ,  markersize=3, lw=1.0, label= "Ratio to cutoff mask (dendrogram)"   )

		axRatio.plot(sigmaListDB, totalFluxRatioDB,'^--',linestyle=':',color= tabBlue ,  markersize=3, lw=1.0,label= "Ratio to cutoff mask (DBSCAN)",  markerfacecolor='none' )

		axRatio.plot(sigmaListSCI, totalFluxRatioSCI,'s--', color= tabBlue ,  markersize=3, lw=1.0,label= "Ratio to cutoff mask (SCIMES)",  markerfacecolor='none' )


		axRatio.set_ylabel('Ratio to cutoff mask', color= tabBlue )

		#draw



		axDendro.legend(loc=6)
		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )

		leg=axRatio.legend(loc=7)
		for text in leg.get_texts():
			plt.setp(text, color=tabBlue)

		axRatio.set_ylim([0.4, 1.06])


		axRatio.yaxis.label.set_color( tabBlue )
		axRatio.spines["right"].set_edgecolor( tabBlue )
		axRatio.tick_params(axis='y', colors= tabBlue )


		fig.tight_layout()
		plt.savefig( "compareParaTotalFlux.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaTotalFlux.png"  ,  bbox_inches='tight',dpi=300)


	def numberDistribution(self):
		"""
		Compare the change of molecular cloud numbers with
		:return:
		"""
		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)



		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)





		fig=plt.figure(figsize=(18,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 14.5,  'serif': ['Helvetica'] })

		#plot dendrogram
		axDendro=fig.add_subplot(1,3,1)

		self.drawNumber(axDendro,tb8Den,tb16Den, label8Den,label16Den, sigmaListDen)
		at = AnchoredText(algDendro, loc=1, frameon=False)
		axDendro.add_artist(at)
		axDendro.set_ylabel(r"Total number of trunks")
		axDendro.set_xlabel(r"CO cutoff ($\sigma$)")
		axDendro.legend(loc=2)
		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )


		#plot DBSCAN
		axDB=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro)
		self.drawNumber(axDB,tb8DB,tb16DB, label8DB,  label16DB ,sigmaListDB)
		at = AnchoredText(algDB, loc=1, frameon=False)
		axDB.add_artist(at)

		#axDB.set_ylabel(r"Total number of clusters")

		axDB.set_xlabel(r"CO cutoff ($\sigma$)")
		axDB.set_ylabel(r"Total number of trunks")

		axDB.legend(loc=2)
		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )

		#plot SCIMES

		axSCI=fig.add_subplot(1,3,3,sharex=axDendro,sharey=axDendro)
		self.drawNumber(axSCI,tb8SCI,tb16SCI, label8SCI,  label16SCI ,sigmaListSCI)
		at = AnchoredText(algSCI, loc=1, frameon=False)
		axSCI.add_artist(at)
		plt.xticks( [2,3,4,5,6,7],["2 (1.0 K)", "3 (1.5 K)" , "4 (2.0 K)" , "5 (2.5 K)" , "6 (3.0 K)" , "7 (3.5 K)"    ] )

		#axSCI.set_ylabel(r"Total number of clusters")

		axSCI.set_xlabel(r"CO cutoff ($\sigma$)")
		axSCI.set_ylabel(r"Total number of clusters")

		axSCI.legend(loc=3)


		fig.tight_layout()
		plt.savefig( "compareParaNumber.pdf"  ,  bbox_inches='tight')
		plt.savefig( "compareParaNumber.png"  ,  bbox_inches='tight',dpi=300)


	def drawFlux(self,ax,tb8List,tb16List, label8,label16, sigmaListDen ):

		#areaEdges=np.linspace(0.25/3600.,150,10000)
		#areaCenter=self.getEdgeCenter( areaEdges )

		areaEdges=np.linspace(8,1e5,1000)
		areaCenter=self.getEdgeCenter( areaEdges )

		NUM_COLORS = 12
		clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples

		for i in range( len(tb8List) ):

			eachTB8 = tb8List[i]
			eachTB16 = tb16List[i]

			if "sum" not in  eachTB8.colnames:
				#dendrogra
				sum8=eachTB8["flux"]/self.getSumToFluxFactor()*0.2 # K km/s
				sum16=eachTB16["flux"]/self.getSumToFluxFactor()*0.2 # K km/s

			else: #dbscan

				sum8 = eachTB8["sum"]*0.2 # K km/s
				sum16 = eachTB16["sum"]*0.2 # K km/s


			binN8,binEdges8 = np.histogram( sum8 , bins=areaEdges  )
			binN16,binEdges16 = np.histogram( sum16 , bins=areaEdges  )

			plot8=ax.plot( areaCenter[binN8>0],binN8[binN8>0], 'o-'  , markersize=1, lw=0.8,label=label8[i] ,alpha= 0.5,color=  clrs[i])

			ax.plot( areaCenter[binN16>0],binN16[binN16>0], '^--'  , markersize=1, lw=0.8,label=label16[i] ,alpha= 0.5,color= clrs[i] )


	def drawArea(self,ax,tb8List,tb16List, label8,label16, sigmaListDen ):


		areaEdges=np.linspace(0.25/3600.,150,10000)
		areaCenter=self.getEdgeCenter( areaEdges )


		NUM_COLORS = 12
		clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples



		for i in range( len(tb8List) ):

			eachTB8 = tb8List[i]
			eachTB16 = tb16List[i]

			binN8 , binEdges8 =np.histogram(eachTB8["area_exact"]/3600., bins=areaEdges  )
			binN16 , binEdges16 =np.histogram(eachTB16["area_exact"]/3600., bins=areaEdges  )


			ax.plot( areaCenter[binN8>0],binN8[binN8>0], 'o-'  , markersize=1, lw=0.8,label=label8[i] ,alpha= 0.5,color=clrs[i] )
			ax.plot( areaCenter[binN16>0],binN16[binN16>0], '^--'   , markersize=1, lw=0.8,label=label16[i] ,alpha= 0.5,color=clrs[i] )


	def drawPhysicalAreaSingle(self, ax, tbDendro2, physicalEdges, physicalCenter,completeArea,  label=None ):

		# physicalEdges  physicalCenter
		#areaEdges=np.linspace(0,100,1000)
		#areaCenter=self.getEdgeCenter( areaEdges )

		realArea=self.getRealArea(tbDendro2)
		binN,binEdges=np.histogram( realArea , bins=physicalEdges )

		#calculate alpha

		meanA, stdA = self.getAlphaWithMCMC(realArea, minArea=completeArea, maxArea=None, physicalArea=True)

		if label!=None:
			label=label+r": $\alpha={:.2f}\pm{:.2f}$".format(meanA,stdA)

		stepa=ax.plot(physicalCenter[binN > 0], binN[binN > 0], 'o-', markersize=1, lw=0.8, label=label, alpha=0.5 )



		return stepa[-1].get_color()
	
	
	def drawNumber(self,ax,tb8List,tb16List,label8,label16,sigmaListDen ):
		Nlist8Den=self.getNList(tb8List)
		Nlist16Den=self.getNList(tb16List)


		ax.plot(sigmaListDen,Nlist8Den,'o-',label="min\_nPix = 8",color="blue",lw=1 , markersize=3 )
		ax.plot(sigmaListDen,Nlist16Den,'o-',label="min\_nPix = 16",color="green", lw=1, markersize=3 )



	def drawTotalFlux(self,ax,tb8List,tb16List,label8,label16,sigmaListDen ):
		#Nlist8Den=self.getTotalFluxList(tb8List)
		Nlist16Den=self.getTotalFluxList(tb16List)


		#ax.plot(sigmaListDen,Nlist8Den,'o-',label="min\_nPix = 8",color="blue",lw=0.5)
		ax.plot(sigmaListDen,Nlist16Den,'o-' , color="green", lw=0.5)




	def getAlphaList(self,tbList, minArea=0.0225 ):
		# calculate alpha and  error for each alpha for each tb

		alphaList=[]
		errorList=[]

		for eachTB in tbList:

			area= eachTB["area_exact"]

			meanA,stdA=self.getAlphaWithMCMC(  area ,  minArea= minArea ,  maxArea=None , physicalArea=False )

			alphaList.append(meanA)
			errorList.append( stdA)

		return  alphaList,  errorList


	def getFluxCol(self,TB):

		if "sum" in TB.colnames:
			return TB["sum"]*0.2 #K km/s


		else:

			return TB["flux"]*0.2/self.getSumToFluxFactor()


	def getFluxAlphaList(self,tbList, sigmaList  ):
		# calculate alpha and  error for each alpha for each tb

		alphaList=[]
		errorList=[]
		for i in range( len(sigmaList) ):
			eachTB = tbList[i]

			eachSigma = sigmaList[i]

			flux= self.getFluxCol(eachTB  )
			minFlux=324*self.rms*0.2*eachSigma*3    # K km/s, the last 2 is the two channels

			meanA,stdA=self.getAlphaWithMCMC(  flux ,  minArea= minFlux ,  maxArea=None , physicalArea=True )

			alphaList.append(meanA)
			errorList.append( stdA)

		return  alphaList,  errorList







	def drawAlpha(self,ax,tb8List,tb16List, label8, label16,sigmaListDen ): #


		#fitting alpha and draw

		alpha8List, alpha8ErrorList = self.getAlphaList(tb16List)
		#alpha16List, alpha16ErrorList = self.getAlphaList(tb16List)

		#ax.plot(sigmaListDen,alpha8List,'o-',label="MinPix = 8",color="blue", markersize= 3, lw=1)
		#ax.plot(sigmaListDen,alpha16List,'o-',label="MinPix = 16",color="green", lw=0.5, markersize= 2.5  ,  alpha=0.8 )

		ax.errorbar(sigmaListDen, alpha8List, yerr= alpha8ErrorList , c='b', marker='o', capsize=1.5, elinewidth=0.8, lw=1,label=r"min\_nPix = 16" )

		return alpha8List,alpha8ErrorList

	def areaAndNumberDistribution(self, algorithm="Dendrogram" ):
		"""
		#draw the area the
		:return:
		"""


		#first, get TBList

		tb8,tb16,label8,label16,sigmaList=self.getTBList(algorithm=algorithm)



		#need to constrain the minP and PeakN, PeakSigma=lower sigma cut + 3 sigma, as the minDelta,



		fig=plt.figure(figsize=(12,6))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		axNumber=fig.add_subplot(1,2,1)

		Nlist8=self.getNList(tb8)
		Nlist16=self.getNList(tb16)


		axNumber.plot(sigmaList,Nlist8,'o-',label="MinPix = 8",color="blue",lw=0.5)
		axNumber.plot(sigmaList,Nlist16,'o-',label="MinPix = 16",color="green", lw=0.5)





		#axArea.set_xlabel(r"Area (deg$^2$)")
		axNumber.set_ylabel(r"Total number of trunks")
		axNumber.set_xlabel(r"CO cutoff ($\sigma$)")

		axNumber.legend()

		at = AnchoredText(algorithm, loc=3, frameon=False)
		#at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
		axNumber.add_artist(at)


		################ Area ##############


		axArea= fig.add_subplot(1,2,2)



		areaEdges=np.linspace(0,150,10000)
		areaCenter=self.getEdgeCenter( areaEdges )


		totalTB=tb8+tb16
		labelStr=label8+label16

		for i in range( len(totalTB) ):

			eachTB = totalTB[i]

			binN,binEdges=np.histogram(eachTB["area_exact"]/3600., bins=areaEdges  )


			axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  , markersize=1, lw=0.8,label=labelStr[i] ,alpha= 0.5 )



		#set ticikes of Area


		axArea.set_yscale('log')
		axArea.set_xscale('log')

		axArea.set_xlim( [ 0.005,150 ] )


		if algorithm=="DBSCAN":
			axArea.set_ylim( [ 0.8,50000 ] )

		else:
			axArea.set_ylim( [ 0.8,10000 ] )





		axArea.set_xlabel(r"Area (deg$^2$)")
		axArea.set_ylabel(r"Bin number of trunks ")


		axArea.legend(ncol=2)


		#draw scimes

		scimesTB=Table.read("ClusterCat_3_16Ve20.fit")

		binN,binEdges=np.histogram(scimesTB["area_exact"]/3600., bins=areaEdges  )


		axArea.plot( areaCenter[binN>0],binN[binN>0], 'o-'  ,  color='red',  markersize=1, lw=0.8,label=r"Scimes,3.0$\sigma$, P16" ,alpha= 0.5 )

		at = AnchoredText("Red: SCIMES,3.0$\sigma$, P16", loc=4, frameon=False)
		#at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
		axArea.add_artist(at)






		plt.savefig( "comparePara_{}.pdf".format(algorithm) ,  bbox_inches='tight')

		plt.savefig( "comparePara_{}.png".format(algorithm) ,  bbox_inches='tight',dpi=300)



	def getTBList(self, algorithm="DBSCAN"):
		"""
		return a list of table,
		:param minP:
		:param algorithm:
		:return:
		"""

		if algorithm=="DBSCAN":
			#ignore minP, only has 8


			TBList=[]
			TBList16=[]

			TBLabelsP8=[]
			TBLabelsP16=[]
			minPix=8

			#DbscanSigmaList= np.arange(2,6.5,0.5)
			DbscanSigmaList= np.arange(2,7.5,0.5)

			for sigmas in DbscanSigmaList:
				tbName= "G2650CO12DBCatS{:.1f}P{}Con2.fit".format(sigmas, minPix)
				ttt8=Table.read(tbName)
				ttt8=self.cleanDBTB(ttt8,pixN=8,minV=sigmas,minDelta=3)
				
				TBList.append(ttt8  )
				ttt16=ttt8[ttt8["pixN"]>=16]
				ttt16=self.cleanDBTB(ttt16,pixN=16,minV=sigmas,minDelta=3)


				TBList16.append(ttt16  )
				TBLabelsP8.append(  r"{:.1f}$\sigma$, P8".format( sigmas)   )
				TBLabelsP16.append( r"{:.1f}$\sigma$, P16".format( sigmas)   )

			


			return TBList,TBList16,TBLabelsP8,TBLabelsP16,DbscanSigmaList


		elif algorithm=="dendrogram" or algorithm=="Dendrogram" :

			TBListP8=[]
			TBListP16=[]

			TBLabelsP8=[]
			TBLabelsP16=[]

			#dendroSigmaList=[2,2.5 , 3, 3.5, 4,4.5,5, 5.5, 6]

			dendroSigmaList=[2,2.5 , 3, 3.5, 4,4.5,5, 5.5, 6,6.5,7 ]

			for sigmas in dendroSigmaList:
				tbName8= "minV{}minP{}_dendroCatTrunk.fit".format(sigmas, 8)
				tbName16= "minV{}minP{}_dendroCatTrunk.fit".format(sigmas, 16)

				TBListP8.append(Table.read(tbName8)  )
				TBListP16.append(Table.read(tbName16)  )


				TBLabelsP8.append(  r"{:.1f}$\sigma$, P8".format( sigmas)   )
				TBLabelsP16.append( r"{:.1f}$\sigma$, P16".format( sigmas)   )



			return TBListP8,TBListP16,TBLabelsP8,TBLabelsP16,dendroSigmaList

		elif algorithm=="SCIMES":

			TBListP8=[]
			TBListP16=[]

			TBLabelsP8=[]
			TBLabelsP16=[]

			#dendroSigmaList=[2,2.5 , 3, 3.5, 4,4.5,5, 5.5, 6,6.5,7]

			dendroSigmaList=[2,2.5 , 3, 3.5, 4,4.5,5, 5.5,6 ,6.5,7]
			path="./scimesG2650/"
			for sigmas in dendroSigmaList:
				tbName8= path+"ClusterCat_{}_{}Ve20.fit".format(sigmas, 8)
				tbName16= path+"ClusterCat_{}_{}Ve20.fit".format(sigmas, 16)

				TBListP8.append(Table.read(tbName8)  )
				TBListP16.append(Table.read(tbName16)  )


				TBLabelsP8.append(  r"{:.1f}$\sigma$, P8".format( sigmas)   )
				TBLabelsP16.append( r"{:.1f}$\sigma$, P16".format( sigmas)   )



			return TBListP8,TBListP16,TBLabelsP8,TBLabelsP16,dendroSigmaList



	def setMinVandPeak(self,cloudLabelFITS,COFITS, peakSigma=3,minP=8):
		"""
		:param cloudLabelFITS:
		:param peakSigma:
		:return:
		"""
		#reject cloud that peak values area less than peakSigma, and total peak number are less then minP



		dataCluster, head= myFITS.readFITS(cloudLabelFITS)

		dataCO,headCO= myFITS.readFITS(COFITS)

		clusterIndex1D= np.where( dataCluster>0)
		clusterValue1D=  dataCluster[clusterIndex1D ]

		Z0,Y0,X0 = clusterIndex1D

		allClouds,counts=np.unique( clusterValue1D , return_counts=True)


		widgets = ['Fast Dendro WithDBSCAN: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),  ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

		pbar = ProgressBar(widgets=widgets, maxval=len(allClouds))
		pbar.start()


		for i in range( len(allClouds) ):


			pbar.update(i)

			cloudID =  allClouds[i]

			pixN=counts[i]
			cloudIndex=self.getIndices(Z0,Y0,X0,clusterValue1D,cloudID)

			if pixN<minP:#reject clouds

				dataCluster[cloudIndex]=0

				continue




			coValues=  dataCO[ cloudIndex ]

			if np.nanmax(coValues) <  peakSigma*self.rms: #reject
				cloudIndex=self.getIndices(Z0,Y0,X0,clusterValue1D,cloudID)
				dataCluster[cloudIndex]=0

				continue
		#relabelclouds
		pbar.finish()
		#dataCluster[dataCluster>0]=1
		s=generate_binary_structure(3,1)

		newDataCluster= dataCluster>0



		labeled_redo, num_features=label(newDataCluster, structure=s) #first label core, then expand, otherwise, the expanding would wrongly connected

		print "Total number of clouds? ",  num_features

		tbDendro=Table.read( "minV5minP8_dendroCatTrunk.fit" )
		print "The dendrogramN is ",len(tbDendro)

		#save the fits

		fits.writeto("relabelFastDendrominPeak{}_P{}.fits".format( peakSigma, minP ), labeled_redo, header=headCO,overwrite=True)


	def fastDendro(self,COFITS,minDelta=3,minV=3,minP=8):

		COData,COHead=myFITS.readFITS( COFITS)

		print np.max(COData)
		#first create dendrogram
		self.computeDBSCAN(COData,COHead, min_sigma=minV,min_pix=3,connectivity=1,region="fastDendroTest")

		dbFITS="fastDendroTestdbscanS{}P3Con1.fits".format(minV)



		self.setMinVandPeak(dbFITS,COFITS, peakSigma=minDelta+minV,minP=minP)

	def clearnDBAssign(self,DBLabelFITS,DBTableFile	,pixN=8,minDelta=3,minV=2 ,prefix="" ):

		minPeak=(minV+minDelta)*self.rms
		saveName=prefix+"DBCLEAN{}_{}Label.fits".format( minV, pixN )
		saveNameTB=prefix+"DBCLEAN{}_{}TB.fit".format( minV, pixN )

		DBTable=Table.read( DBTableFile )

		dataCluster,headCluster=myFITS.readFITS(DBLabelFITS )

		clusterIndex1D= np.where( dataCluster>0 )
		clusterValue1D=  dataCluster[clusterIndex1D ]

		Z0,Y0,X0 = clusterIndex1D
		#cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, newID)

		emptyTB= Table( DBTable[0] )
		emptyTB.remove_row(0)
		print "Cleaning DBSCAN table..."

		widgets = ['Recalculating cloud parameters: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),  ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options

		pbar = ProgressBar(widgets=widgets, maxval=len(DBTable))
		pbar.start()

		indexRun=0
		for eachDBRow in DBTable:
			indexRun=indexRun+1
			pbar.update(indexRun)
			cloudID=  eachDBRow["_idx"]

			pixNCloud=int(  eachDBRow["pixN"]  )
			peakCloud= eachDBRow["peak"]

			if peakCloud < minPeak or pixNCloud< pixN : # set as zero

				cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, cloudID)
				dataCluster[cloudIndex] = 0

				continue

			#edge


			emptyTB.add_row( eachDBRow  )
		pbar.finish()
		#save
		fits.writeto(saveName,dataCluster,header=headCluster,overwrite=True)

		emptyTB.write( saveNameTB,overwrite=True  )

		return saveName, saveNameTB


	def getCropDataAndHead(self,rawFITSFile,drawChannel,lRange,bRange):

		tmpFITS="checkCloudTmp.fits"

		tmpData, tmpHead = myFITS.readFITS(rawFITSFile)
		save2D=tmpData[drawChannel]

		fits.writeto(tmpFITS, save2D,header=tmpHead ,overwrite=True)


		cropTMP = "checkCloudTmpCrop.fits"

		doFITS.cropFITS2D(tmpFITS,cropTMP, Lrange=lRange, Brange=bRange , overWrite=True   )



		return myFITS.readFITS(cropTMP)


	def drawCloudMap(self,drawChannel=98,lRange=None,bRange=None ):
		"""
		#draw small clouds to check if the are real...

		one color for DBSCAN
		one color for dendrogram,

		draw 2sigma, because they would provide the smallest area of clouds,

		:return:
		"""

		xRange=[]
		yRange=[]

		#first

		#axCO.set_xlim( [1000, 1700] )
		#axCO.set_ylim( [350, 950 ] )


		COFITS="G2650Local30.fits"

		#data,head=myFITS.readFITS(COFITS)

		data,head=self.getCropDataAndHead(   COFITS,drawChannel=drawChannel,lRange=lRange,bRange= bRange )


		WCSCO=WCS(head)

		channelRawCO=data #[drawChannel]

		DBLabelFITS = "DBCLEAN2.0_8Label.fits"
		DBTableFile= "DBCLEAN2.0_8TB.fit"
		drawDBSCANtb=Table.read( DBTableFile )


		#relabelDB,newDBTable= self.clearnDBAssign( DBLabelFITS,DBTableFile	,pixN=16, minDelta=3, minV=2  )


		drawDBSCANtb=self.cleanDBTB( drawDBSCANtb, minDelta=3,minV=2,pixN=8)


		drawDBSCANtb=self.removeWrongEdges(drawDBSCANtb)



		drawDBSCANData,drawDBSCANHead=self.getCropDataAndHead(   DBLabelFITS,drawChannel=drawChannel,lRange=lRange,bRange= bRange )
		#drawDBSCANData,drawDBSCANHead = myFITS.readFITS(DBLabelFITS)
		WCSCrop = WCS( drawDBSCANHead )

		channelDBSCAN =drawDBSCANData  #drawDBSCANData[drawChannel]



		dbClouds=np.unique(channelDBSCAN)



		drawDENDROtb=Table.read("minV2minP8_dendroCatTrunk.fit")

		drawDENDROtb=self.removeWrongEdges(drawDENDROtb)


		#drawDENDROData,drawDENDROHead = myFITS.readFITS("minV2minP16_TrunkAsign.fits")
		#drawDENDROData=drawDENDROData-1
		#channelDENDRO =  drawDENDROData[drawChannel]


		drawDendrogram, drawDendrogramHead = self.getCropDataAndHead(   "minV2minP16_TrunkAsign.fits",drawChannel=drawChannel,lRange=lRange,bRange= bRange )


		channelDENDRO=drawDendrogram-1


		dendroClouds=np.unique(channelDENDRO)



		#scimes
		#drawSCIMESData, drawSCIMESHead = myFITS.readFITS("./scimesG2650/ClusterAsgn_2_8Ve20.fits")
		#channelSCIMES =  drawSCIMESData[drawChannel]

		drawSCIMES, drawSCIMESHead =self.getCropDataAndHead(   "./scimesG2650/ClusterAsgn_2_16Ve20.fits", drawChannel=drawChannel,lRange=lRange,bRange= bRange )
		channelSCIMES = drawSCIMES  #drawSCIMESData[drawChannel]

		maximumArea= 144 *0.25 #arcmin^2

		#
		#print drawDBSCANtb.colnames



		fig = plt.figure(1, figsize=(10, 8) )
		rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
		#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

		rc('text', usetex=True)

		axCO= pywcsgrid2.subplot(221, header=   WCSCrop  )
		axCO.imshow(channelRawCO,origin='lower',cmap="bone",vmin=0 ,vmax=3,interpolation='none')



		#draw Dendrogram.............
		#the trunk assign of Dendrogrom is wong, the cloud0 ara all missed, so we ignore them

		runIndex=0

		for eachDRC in dendroClouds:
			break
			if runIndex == 0:
				labelDendro = "Dendrogram"


			else:
				labelDendro = None



			eachDRC=int(eachDRC)
			if eachDRC==0:
				continue

			cRow=  drawDENDROtb[drawDENDROtb["_idx"]==eachDRC  ]

			area=cRow["area_exact"]

			if area>maximumArea:
				continue

			else:
				#draw

				if np.isnan(cRow["x_cen"] ):
					continue
				#print eachDRC

				axCO.scatter(cRow["x_cen"], cRow["y_cen"], s=13,facecolors='none',edgecolors='r', linewidths=0.3,  label= labelDendro  )
				runIndex=runIndex+1


		#draw DBSCAN.............
		runIndex=0
		for eachDBC in dbClouds:
			break
			if runIndex == 0:
				labelDB = "DBSCAN"


			else:
				labelDB = None

			if eachDBC==0:
				continue



			cRow=  drawDBSCANtb[drawDBSCANtb["_idx"]==eachDBC  ]



			if len(cRow)==0: #may be edge sources
				continue

			area=cRow["area_exact"]


			if area>maximumArea:
				continue

			else:
				#draw

				if np.isnan(cRow["x_cen"] ):
					continue


				axCO["gal"].scatter(cRow["x_cen"], cRow["y_cen"]  , s=10,facecolors='none',edgecolors='b',linewidths=0.3, label= labelDB )
				runIndex=runIndex+1

		axCO.legend(loc=3)

		axCO.set_ticklabel_type("absdeg", "absdeg")
		axCO.axis[:].major_ticks.set_color("w")

		#draw DBSCAN labels

		cmap = plt.cm.gist_rainbow

		cmap = plt.cm.jet

		cmap.set_bad('black', 1.)

		axDBSCAN= pywcsgrid2.subplot(222, header=   WCSCO  ,sharex=axCO,sharey=axCO )

		newLabels=self.showLabels(axDBSCAN, channelDBSCAN  )

		self.contourLabels(axDBSCAN, channelDBSCAN)


		at = AnchoredText("DBSCAN", loc=1, frameon=False,prop={"color":"w"} )
		axDBSCAN.add_artist(at)
		#draw Dendrogram labels
		axDendrogram= pywcsgrid2.subplot(223, header=   WCSCO ,sharex=axCO,sharey=axCO )


		#draw dnedrogram contours
		newLabels=self.showLabels(axDendrogram, channelDENDRO  )
		self.contourLabels(axDendrogram, channelDENDRO)

		#self.contourLabels(axDendrogram,newLabels )


		at = AnchoredText("Dendrogram", loc=1, frameon=False,prop={"color":"w"} )
		axDendrogram.add_artist(at)

		#draw SCIMES

		axSCIMES= pywcsgrid2.subplot(224, header=   WCSCO ,sharex=axCO,sharey=axCO )

		newLabels=self.showLabels(axSCIMES, channelSCIMES  )

		self.contourLabels(axSCIMES,channelSCIMES )


		at = AnchoredText("SCIMES", loc=1, frameon=False,prop={"color":"w"} )
		axSCIMES.add_artist(at)




		fig.tight_layout()
		plt.savefig("checkCloud.pdf", bbox_inches="tight")

		plt.savefig("checkCloud.png", bbox_inches="tight",dpi=600)

	def showLabels(self,ax,labelFITS2DArray):
		"""
		Use jet to show labels, and
		:param ax:
		:param labelFITS2DArray:
		:return:
		"""

		labelFITS2DArray=np.nan_to_num(labelFITS2DArray)

		minV=np.nanmin(labelFITS2DArray )
		#labelFITS2DArray = labelFITS2DArray.astype(np.float)
		#labelFITS2DArray[labelFITS2DArray==minV]= np.NaN # only float is allowed to have NaN values

		cmap = plt.cm.jet
		cmap.set_bad('black', 1. )

		dbClouds = np.unique( labelFITS2DArray )
		newLabels = np.arange( len(dbClouds) )

		np.random.shuffle(newLabels)

		#reassign fits

		newChannelMap = labelFITS2DArray.copy()


		#
		index1D=np.where(labelFITS2DArray > minV)
		values1D=labelFITS2DArray[index1D]

		Y0, X0 = index1D



		for i in np.arange(len(dbClouds)):

			oldID = dbClouds[i]
			newID= newLabels[i]

			if oldID==minV:
				continue

			indicesL=self.getIndices2D(Y0,X0,values1D,oldID)

			newChannelMap[  indicesL  ] = newID

		newChannelMap = newChannelMap.astype(np.float)
		newChannelMap[   labelFITS2DArray == minV   ] = np.NaN


		ax.imshow(newChannelMap,origin='lower',cmap=cmap, vmin=0,vmax=len(newLabels),  interpolation='none')


		return newChannelMap






	def contourLabels(self,ax ,  labelFITS2DArray ):
		"""

		:param ax:
		:param labelFITS2D:
		:return:
		"""

		dbClouds=np.unique( labelFITS2DArray )

		noiseV=np.min( dbClouds)

		for eachDendroC in dbClouds:
			#pass
			if eachDendroC == noiseV:
				continue

			cloudIndex= labelFITS2DArray == eachDendroC
			cloudIndex=cloudIndex.astype(int)


			ax.contour(cloudIndex, colors="white", linewidths=0.2, origin="lower", levels=[1] )







	def getLVFITSByDBMASK(self,DBlabel,CO12FITS,PVHeadTempFITS):
		dataDB, headDB = myFITS.readFITS( DBlabel )
		dataCO,headCO= myFITS.readFITS( CO12FITS )

		pvData,pvHead= myFITS.readFITS( PVHeadTempFITS )

		mask=dataDB>0
		mask=mask+0
		coMask=dataCO*mask
		Nz,Ny,Nx=dataCO.shape
		pvData=np.nansum(coMask, axis=1 )/Ny

		fits.writeto("G2650PV_DBMASK.fits",pvData,header=pvHead, overwrite=True)

	def cleanAllDBfits(self):

		DbscanSigmaList = np.arange(2, 7.5, 0.5)


		for sigmas in DbscanSigmaList:

			for minPix in [8,16]:

				tbName = "G2650CO12DBCatS{:.1f}P{}Con2.fit".format(sigmas, 8)
				fitsName = "G2650CO12dbscanS{:.1f}P{}Con2.fits".format(sigmas, 8 )
				self.clearnDBAssign( fitsName,tbName, pixN=minPix, minV=sigmas, minDelta= 3 )



	def splitEdges(self,TB):
		"""

		:param TB:
		:return: good sources , and edge sources

		"""
 
		processTB=TB

		dbClusterTB=processTB
		if "peak" in dbClusterTB.colnames: #for db scan table

			select1=np.logical_and( processTB["x_cen"]<= 26.25 ,processTB["y_cen"] >= 3.25  )
			select2=np.logical_and( processTB["x_cen"]>=49.25 ,processTB["y_cen"]>=  3.75 )
			allSelect= np.logical_or( select1,select2 )

			badSource=dbClusterTB[allSelect]

			part1= processTB[ np.logical_or( processTB["x_cen"]>26.25 ,processTB["y_cen"] < 3.25  )   ] #1003, 3.25
			part2= part1[ np.logical_or( part1["x_cen"]<49.25 ,part1["y_cen"]<  3.75 )   ] #1003, 3.25


			return part2,badSource



		else: #dendrogram tb


			select1=np.logical_and( processTB["x_cen"]>= 2815 ,processTB["y_cen"] >= 1003  )
			select2=np.logical_and( processTB["x_cen"]<=  55 ,processTB["y_cen"]>= 1063  )
			allSelect= np.logical_or( select1,select2 )


			badSource=dbClusterTB[allSelect]
			part1= processTB[ np.logical_or( processTB["x_cen"]< 2815 ,processTB["y_cen"] < 1003  )   ] #1003, 3.25
			part2= part1[ np.logical_or( part1["x_cen"]>  55 ,part1["y_cen"]< 1063  )   ] #1003, 3.25



			return part2,badSource


		#used to remove edge sources, and




	def produceMask(self, COFITS, LabelFITS ,labelTB, region=""):

		#mask pixels that have been not ben labeled by other

		dataCluster, headCluster = myFITS.readFITS(LabelFITS)

		dataCO, headCO = myFITS.readFITS( COFITS )
		rmsData,rmsHead=myFITS.readFITS( "/home/qzyan/WORK/myDownloads/testScimes/RMS_G2650CO12.fits" )


		dbClusterTB=Table.read( labelTB )

		saveFITS=region+"MaskedCO.fits"
		minV= np.nanmin(dataCluster)
		print "The noise is maked with ",minV

		coGood=  dataCluster>minV


		clusterIndex1D= np.where( dataCluster>minV )
		clusterValue1D=  dataCluster[clusterIndex1D ]

		Z0,Y0,X0 = clusterIndex1D



		dataCO[ ~coGood ]=0#including nan


		#getedigeTBlist
		processTB=dbClusterTB
		
		goodSource,badSource=self.splitEdges( processTB )


		for eachBadC in badSource:

			badID=eachBadC["_idx"]
			cloudIndex = self.getIndices(Z0, Y0, X0, clusterValue1D, badID)
			
			dataCO[ cloudIndex ]=0




		fits.writeto(saveFITS,dataCO, header=headCO,   overwrite = True )

	def getDiffCO(self,maskedFITS,labelFITS, cutoff=3, ):


		dataCluster, headCluster = myFITS.readFITS(labelFITS)

		dataCO, headCO = myFITS.readFITS( maskedFITS )

		minV= np.nanmin(dataCluster)

		dataCO[dataCO<cutoff*self.rms]=0

		dataCO[dataCluster> minV]=0

		fits.writeto( "DiffTest.fits" , dataCO, header=headCluster,overwrite=True   )


	def extendAllScimes(self  ): #only extend,no others
		"""
		Extend the scimes Asgncube
		:return:
		"""
		#the SCIMES assign cubes stars from -1
		#scimesPath="./scimesG2650/"
		#minV=-1
		G2650MaskCO = "G2650CO12MaskedCO.fits"
		scimesPath="/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/"
		#dendroSigmaList = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], negative values found in fits, no idea why
		dendroSigmaList = [  3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

		for eachSigma  in dendroSigmaList :
			for eachPix in [8, 16]:
				scimesAsgnFITS=scimesPath+"ClusterAsgn_{}_{}Ve20.fits".format( eachSigma ,eachPix )

				if os.path.isfile( scimesAsgnFITS):

					saveName="ClusterAsgn_{}_{}Ve20_extend.fits".format( eachSigma ,eachPix )
					self.myDilation(scimesAsgnFITS, None,startSigma=15, saveName=saveName, endSigma=eachSigma,maskCOFITS=G2650MaskCO ,savePath= scimesPath)


	def testFluxOfUM(self):

		UMfitsDB="./ursaMajor/UMCO12dbscanS2P16Con2.fits"
		dataDB,headDB=myFITS.readFITS(UMfitsDB)
		UMCO= "/home/qzyan/WORK/projects/NewUrsaMajorPaper/OriginalFITS/myCut12CO.fits"
		dataCO, headCO =myFITS.readFITS( UMCO )


		dbMask= dataDB>0
		dataCO[~dbMask]=0

		fluxList=[]
		sigmaList=np.arange(2,8,0.5)
		for sigmas in sigmaList:
			print sigmas
			dataCO[dataCO<sigmas*self.rms]=0

			totalFlux=np.sum(dataCO )
			fluxList.append( totalFlux*0.16)

		#plot
		fig=plt.figure(figsize=(10, 8))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#############   plot dendrogram
		axUM=fig.add_subplot(1,1,1)

		axUM.plot(sigmaList, fluxList, 'o--' ,linestyle=':', color="blue", lw=1.5 )


		axUM.set_ylabel(r"Total flux ($\rm K\ km\ s$$^{-1}$ $\Omega_\mathrm{A}$)")
		axUM.set_xlabel(r"CO cutoff ($\sigma$)")

		plt.savefig( "ursaMajorFlux.pdf"  ,  bbox_inches='tight')
		plt.savefig( "ursaMajorFlux.png"  ,  bbox_inches='tight', dpi=300)



	def drawVeDistribution(self):
		"""
		:return:
		"""
		####
		#
		#first check if the trunk assign of dendrogram is right


		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)

		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)

		velEdges = np.linspace(0,7,250)
		areaCenter = self.getEdgeCenter( velEdges )


		fig=plt.figure(figsize=(16,5))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#Dendrogram....
		axDendro=fig.add_subplot(1,3,1)
		color2 = self.drawVelTBSingle(axDendro, tb16Den[2] , velEdges ,areaCenter,label= label16Den[2]  )
		color6 = self.drawVelTBSingle(axDendro,  tb16Den[6] , velEdges ,areaCenter,label= label16Den[6]   )
		color10 = self.drawVelTBSingle(axDendro,  tb16Den[10] , velEdges ,areaCenter,label= label16Den[10]   )
		l=axDendro.legend(loc=1)

		colorsDendro=[color2, color6, color10 ]

		for text,color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)





		at = AnchoredText("Dendrogram", loc=4, frameon=False ,pad=1)
		axDendro.add_artist(at)


		axDendro.set_ylabel(r"Number of trunks")

		axDendro.set_xlabel(r"Velocity dispersion $\sigma_V$ ($\rm km\ s$$^{-1}$)")

		#DBSCAN........
		axDBSCAN=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro)
		color2=self.drawVelTBSingle(axDBSCAN, tb16DB[2] , velEdges ,areaCenter,label= label16DB[2]  )
		color6=self.drawVelTBSingle(axDBSCAN,  tb16DB[6] , velEdges ,areaCenter,label= label16DB[6]   )
		color10=self.drawVelTBSingle(axDBSCAN,  tb16DB[10] , velEdges ,areaCenter,label= label16DB[10]   )
		l=axDBSCAN.legend(loc=1)
 
		colorsDendro=[color2, color6, color10 ]

		for text,color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)



		at = AnchoredText("DBSCAN", loc=4, frameon=False,pad=1)
		axDBSCAN.add_artist(at)
		axDBSCAN.set_xlabel(r"Velocity dispersion $\sigma_V$ ($\rm km\ s$$^{-1}$)")
		axDBSCAN.set_ylabel(r"Number of trunks")


		##SCIMES

		axSCIMES=fig.add_subplot(1,3,3,sharex=axDendro,sharey=axDendro)
		color2=self.drawVelTBSingle(axSCIMES, tb16SCI[2] , velEdges ,areaCenter,label= label16SCI[2]  )
		color6=self.drawVelTBSingle(axSCIMES,  tb16SCI[6] , velEdges ,areaCenter,label= label16SCI[6]   )
		color10=self.drawVelTBSingle(axSCIMES,  tb16SCI[10] , velEdges ,areaCenter,label= label16SCI[10]   )
		l=axSCIMES.legend(loc=1)
		colorsDendro=[color2, color6, color10 ]

		for text,color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)

		at = AnchoredText("SCIMES", loc=4, frameon=False ,pad=1)
		axSCIMES.add_artist(at)


		axSCIMES.set_xlabel(r"Velocity dispersion $\sigma_V$ ($\rm km\ s$$^{-1}$)")

		axSCIMES.set_ylabel(r"Number of clusters")


		axSCIMES.set_xlim([0,4])


		#axDBSCAN.set_xlabel(r"Velocity dispersion $\sigma_V$ ($\rm km\ s$$^{-1}$)")

		fig.tight_layout()
		plt.savefig( "velDistribute.pdf"  ,  bbox_inches='tight')
		plt.savefig( "velDistribute.png"  ,  bbox_inches='tight', dpi=300)


	def drawVelTBSingle(self,ax,testTB,velEdges,areaCenter,label=None):


		#testTB=self.removeWrongEdges(testTB)
		vDisperse = testTB["v_rms"]

		maxV= np.max(vDisperse)
		meanV= np.mean(vDisperse)
		medianV= np.median(vDisperse)


		binN8 , binEdges8 =np.histogram(vDisperse, bins=velEdges  )

		peakV=areaCenter[ binN8.argmax() ]

		suffix="\nPeak: {:.2f}, Mean: {:.2f}, Median: {:.2f}".format( peakV, meanV, medianV )


		stepa=ax.step( areaCenter, binN8, lw=1.0 , label=label+suffix )





		return stepa[-1].get_color()

	def drawPeakTBSingle(self,ax,testTB,velEdges,areaCenter,label=None):

		#testTB=self.removeWrongEdges(testTB)
		vDisperse = testTB["peak"]

		peakV= np.max(vDisperse)
		meanV= np.mean(vDisperse)
		medianV= np.median(vDisperse)

		suffix="\nMean: {:.1f}, Median: {:.1f}".format(   meanV, medianV )

		binN8 , binEdges8 =np.histogram(vDisperse, bins=velEdges  )
		stepa = ax.step( areaCenter, binN8, lw=1.0 , label=label+suffix )
		#ax.set_yscale('log')



		return stepa[-1].get_color()

	def drawPeakDistribution(self):
		"""
		:return:
		"""
		####

		# first check if the trunk assign of dendrogram is right

		algDendro="Dendrogram"
		tb8Den,tb16Den,label8Den,label16Den,sigmaListDen=self.getTBList(algorithm=algDendro)
		tb8Den=self.removeAllEdges(tb8Den)
		tb16Den=self.removeAllEdges(tb16Den)


		algDB="DBSCAN"
		tb8DB,tb16DB,label8DB,label16DB,sigmaListDB=self.getTBList(algorithm=algDB)
		tb8DB=self.removeAllEdges(tb8DB)
		tb16DB=self.removeAllEdges(tb16DB)

		algSCI="SCIMES"
		tb8SCI,tb16SCI,label8SCI,label16SCI,sigmaListSCI=self.getTBList(algorithm= algSCI )
		tb8SCI=self.removeAllEdges(tb8SCI)
		tb16SCI=self.removeAllEdges(tb16SCI)

		velEdges = np.linspace(0,40,200)
		areaCenter = self.getEdgeCenter( velEdges )


		fig=plt.figure(figsize=(16,5))
		rc('text', usetex=True )
		rc('font', **{'family': 'sans-serif',  'size'   : 13,  'serif': ['Helvetica'] })

		#Dendrogram....
		axDendro=fig.add_subplot(1,3,1)
		#self.drawVelTBSingle(axDendro, tb16Den[2] , velEdges ,areaCenter,label= label16Den[2]  )
		#self.drawVelTBSingle(axDendro,  tb16Den[6] , velEdges ,areaCenter,label= label16Den[6]   )
		#self.drawVelTBSingle(axDendro,  tb16Den[10] , velEdges ,areaCenter,label= label16Den[10]   )

		#to be modified
		tbDendro2=Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/Dendro_3_16Ve20ManualCat.fit")
		tbDendro6=Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/Dendro_5_16Ve20ManualCat.fit")
		tbDendro10=Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/Dendro_7_16Ve20ManualCat.fit")

		tbDendro2, tbDendro6,tbDendro10 =self.removeAllEdges( [ tbDendro2, tbDendro6,tbDendro10 ] )



		color2  = self.drawPeakTBSingle(axDendro,tbDendro2   , velEdges ,areaCenter,label= label16Den[2]  )
		color6  = self.drawPeakTBSingle(axDendro,  tbDendro6 , velEdges ,areaCenter,label= label16Den[6]   )
		color10 = self.drawPeakTBSingle(axDendro,  tbDendro10 , velEdges ,areaCenter,label= label16Den[10]   )


		l=axDendro.legend(loc=1)
		colorsDendro=[color2, color6, color10 ]
		for text,color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)



		at = AnchoredText("Dendrogram", loc=4, frameon=False,pad=1 )
		axDendro.add_artist(at)


		axDendro.set_ylabel(r"Number of trunks")

		
		axDendro.set_xlabel(r"Peak brightness temperature (K)")

		#DBSCAN........
		axDBSCAN=fig.add_subplot(1,3,2,sharex=axDendro,sharey=axDendro)
		color2= self.drawPeakTBSingle(axDBSCAN, tb16DB[2] , velEdges ,areaCenter,label= label16DB[2]  )
		color6= self.drawPeakTBSingle(axDBSCAN,  tb16DB[6] , velEdges ,areaCenter,label= label16DB[6]   )
		color10= self.drawPeakTBSingle(axDBSCAN,  tb16DB[10] , velEdges ,areaCenter,label= label16DB[10]   )



		l=axDBSCAN.legend(loc=1)
		colorsDendro=[color2, color6, color10 ]
		for text,color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)




		at = AnchoredText("DBSCAN", loc=4, frameon=False, pad=1 )
		axDBSCAN.add_artist(at)
		#axDBSCAN.set_xlabel(r"Peak values (K)")
		axDBSCAN.set_xlabel(r"Peak brightness temperature (K)")
		axDBSCAN.set_ylabel(r"Number of trunks")

		##SCIMES

		axSCIMES=fig.add_subplot(1,3,3,sharex=axDendro,sharey=axDendro)

		tbSCIMES2=Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/ClusterAsgn_3_16Ve20ManualCat.fit")
		tbSCIMES6=Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/ClusterAsgn_5_16Ve20ManualCat.fit")
		tbSCIMES10=Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/ClusterAsgn_7_16Ve20ManualCat.fit")

		tbSCIMES2, tbSCIMES6,tbSCIMES10 =self.removeAllEdges( [ tbSCIMES2, tbSCIMES6,tbSCIMES10 ] )



		color2=self.drawPeakTBSingle(axSCIMES, tbSCIMES2 , velEdges ,areaCenter,label= label16SCI[2]  )
		color6=self.drawPeakTBSingle(axSCIMES,  tbSCIMES6 , velEdges ,areaCenter,label= label16SCI[6]   )
		color10=self.drawPeakTBSingle(axSCIMES,  tbSCIMES10 , velEdges ,areaCenter,label= label16SCI[10]   )
		l=axSCIMES.legend(loc=1)


		colorsDendro=[color2, color6, color10 ]
		for text,color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)




		at = AnchoredText("SCIMES", loc=4, frameon=False, pad=1 )
		axSCIMES.add_artist(at)

		axSCIMES.set_ylabel(r"Number of clusters")

		axSCIMES.set_xlabel(r"Peak brightness temperature (K)")

		axSCIMES.set_xlim(0,15)
		fig.tight_layout()
		plt.savefig("peakDistribute.pdf", bbox_inches='tight')
		plt.savefig("peakDistribute.png", bbox_inches='tight', dpi=300)




	def produceSCIMECat(self):
		"""
		To get the peak of clusters
		:return:
		"""
		sicmesPath="/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/"
		labelFITS = sicmesPath+"ClusterAsgn_3_16Ve20.fits"
		savename = sicmesPath+"ClusterAsgn_3_16Ve20ManualCat"
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS, labelFITS, doDBSCAN.TBModel, minPix=16, rms=3,   saveMarker=savename)

		#######################
		sicmesPath="/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/"
		labelFITS = sicmesPath+"ClusterAsgn_5_16Ve20.fits"
		savename = sicmesPath+"ClusterAsgn_5_16Ve20ManualCat"
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS, labelFITS, doDBSCAN.TBModel, minPix=16, rms=5,   saveMarker=savename)

		#######################
		sicmesPath="/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/"
		labelFITS = sicmesPath+"ClusterAsgn_7_16Ve20.fits"
		savename = sicmesPath+"ClusterAsgn_7_16Ve20ManualCat"
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS, labelFITS, doDBSCAN.TBModel, minPix=16, rms=7,   saveMarker=savename)



	def produceDENDROCat(self):
		"""
		To get the peak of clusters
		:return:
		"""
		#sicmesPath="/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/"
		labelFITS =  "G2650minV7minP16_TrunkAsignMask0.fits"
		savename =  "Dendro_7_16Ve20ManualCat"
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS, labelFITS, doDBSCAN.TBModel, minPix=16, rms=3,   saveMarker=savename)

		#######################
		#sicmesPath="/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/"
		labelFITS =  "G2650minV5minP16_TrunkAsignMask0.fits"
		savename =  "Dendro_5_16Ve20ManualCat"
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS, labelFITS, doDBSCAN.TBModel, minPix=16, rms=5,   saveMarker=savename)

		#######################
		#sicmesPath="/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/"
		labelFITS =   "G2650minV3minP16_TrunkAsignMask0.fits"
		savename =  "Dendro_3_16Ve20ManualCat"
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS, labelFITS, doDBSCAN.TBModel, minPix=16, rms=7,   saveMarker=savename)




	def physicalAreaDistribution(self):
		"""
		#draw the physical Area Distribution distribution of molecular clouds
		:return:
		"""

		#9pixels at 1500 pc
		completeArea=0.428 #36824657505895 #pc^2 should equal to

		#use 3,5,7#because they all have observed area

		algDendro = "Dendrogram"
		tb8Den, tb16Den, label8Den, label16Den, sigmaListDen = self.getTBList(algorithm=algDendro)
		tb8Den = self.removeAllEdges(tb8Den)
		tb16Den = self.removeAllEdges(tb16Den)

		algDB = "DBSCAN"
		tb8DB, tb16DB, label8DB, label16DB, sigmaListDB = self.getTBList(algorithm=algDB)
		tb8DB = self.removeAllEdges(tb8DB)
		tb16DB = self.removeAllEdges(tb16DB)

		algSCI = "SCIMES"
		tb8SCI, tb16SCI, label8SCI, label16SCI, sigmaListSCI = self.getTBList(algorithm=algSCI)
		tb8SCI = self.removeAllEdges(tb8SCI)
		tb16SCI = self.removeAllEdges(tb16SCI)



		## physicalEdges  physicalCenter
		physicalEdges=np.linspace(0,100,1000) #square pc^2
		physicalCenter=self.getEdgeCenter( physicalEdges )


		velEdges = np.linspace(0, 40, 200)
		areaCenter = self.getEdgeCenter(velEdges)

		fig = plt.figure(figsize=(16, 5))
		rc('text', usetex=True)
		rc('font', **{'family': 'sans-serif', 'size': 13, 'serif': ['Helvetica']})

		# Dendrogram....
		axDendro = fig.add_subplot(1, 3, 1)
		# self.drawVelTBSingle(axDendro, tb16Den[2] , velEdges ,areaCenter,label= label16Den[2]  )
		# self.drawVelTBSingle(axDendro,  tb16Den[6] , velEdges ,areaCenter,label= label16Den[6]   )
		# self.drawVelTBSingle(axDendro,  tb16Den[10] , velEdges ,areaCenter,label= label16Den[10]   )

		# to be modified
		tbDendro2 = Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/Dendro_3_16Ve20ManualCat.fit")
		tbDendro6 = Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/Dendro_5_16Ve20ManualCat.fit")
		tbDendro10 = Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/Dendro_7_16Ve20ManualCat.fit")

		tbDendro2, tbDendro6, tbDendro10 = self.removeAllEdges([tbDendro2, tbDendro6, tbDendro10])

		color2 = self.drawPhysicalAreaSingle(axDendro, tbDendro2, physicalEdges, physicalCenter, completeArea,label=label16Den[2])
		color6 = self.drawPhysicalAreaSingle(axDendro, tbDendro6, physicalEdges, physicalCenter,completeArea, label=label16Den[6])
		color10 = self.drawPhysicalAreaSingle(axDendro, tbDendro10, physicalEdges, physicalCenter, completeArea, label=label16Den[10])



		l = axDendro.legend(loc=1)
		colorsDendro = [color2, color6, color10]
		for text, color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)


		#draw complete line
		axDendro.plot( [completeArea,completeArea],[2,2000],'--',color='black', lw=1  )


		at = AnchoredText("Dendrogram", loc=3, frameon=False, pad=0.2)
		axDendro.add_artist(at)


		xLabelStr= r"Phyiscal area ($\rm pc^{2}$)"
		axDendro.set_ylabel(r"Number of trunks")

		axDendro.set_xlabel( xLabelStr )


		axDendro.set_yscale('log')
		axDendro.set_xscale('log')


		# DBSCAN........
		axDBSCAN = fig.add_subplot(1, 3, 2, sharex=axDendro, sharey=axDendro)



		color2 = self.drawPhysicalAreaSingle(axDBSCAN, tb16DB[2], physicalEdges, physicalCenter, completeArea,label=label16DB[2])
		color6 = self.drawPhysicalAreaSingle(axDBSCAN, tb16DB[6], physicalEdges, physicalCenter,completeArea, label=label16DB[6])
		color10 = self.drawPhysicalAreaSingle(axDBSCAN, tb16DB[10], physicalEdges, physicalCenter, completeArea, label=label16DB[10])


		l = axDBSCAN.legend(loc=1)
		colorsDendro = [color2, color6, color10]
		for text, color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)

		at = AnchoredText("DBSCAN", loc=3, frameon=False, pad=0.2)
		axDBSCAN.add_artist(at)
		# axDBSCAN.set_xlabel(r"Peak values (K)")
		axDBSCAN.set_xlabel( xLabelStr )
		axDBSCAN.set_ylabel(r"Number of trunks")
		axDBSCAN.plot( [completeArea,completeArea],[2,2000],'--',color='black', lw=1  )


		##SCIMES

		axSCIMES = fig.add_subplot(1, 3, 3, sharex=axDendro, sharey=axDendro)

		tbSCIMES2 = Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/ClusterAsgn_3_16Ve20ManualCat.fit")
		tbSCIMES6 = Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/ClusterAsgn_5_16Ve20ManualCat.fit")
		tbSCIMES10 = Table.read("/home/qzyan/WORK/myDownloads/MWISPcloud/scimesG2650/ClusterAsgn_7_16Ve20ManualCat.fit")

		tbSCIMES2, tbSCIMES6, tbSCIMES10 = self.removeAllEdges([tbSCIMES2, tbSCIMES6, tbSCIMES10])


		color2 = self.drawPhysicalAreaSingle(axSCIMES, tbSCIMES2, physicalEdges, physicalCenter, completeArea,label=label16SCI[2])
		color6 = self.drawPhysicalAreaSingle(axSCIMES, tbSCIMES6 , physicalEdges, physicalCenter,completeArea, label=label16SCI[6])
		color10 = self.drawPhysicalAreaSingle(axSCIMES,  tbSCIMES10 , physicalEdges, physicalCenter, completeArea, label=label16SCI[10])



		l = axSCIMES.legend(loc=1)

		colorsDendro = [color2, color6, color10]
		for text, color in zip(l.get_texts(), colorsDendro):
			text.set_color(color)

		at = AnchoredText("SCIMES", loc=3, frameon=False, pad=0.2)
		axSCIMES.add_artist(at)



		axSCIMES.set_ylabel(r"Number of clusters")
		axSCIMES.set_xlabel(xLabelStr)
		axSCIMES.plot( [completeArea,completeArea],[2,2000],'--',color='black', lw=1  )
		axSCIMES.set_yscale('log')
		axSCIMES.set_xscale('log')



		fig.tight_layout()
		plt.savefig("physicalAreaDistribute.pdf", bbox_inches='tight')
		plt.savefig("physicalAreaDistribute.png", bbox_inches='tight', dpi=300)


	def drawOverallMoment(self):
		"""
		draw a simple map for over all moment of local molecular clouds
		:return:
		"""
		fitsFile="allM0G2650Local.fits"
		import matplotlib
		from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
		from mpl_toolkits.axes_grid1.axes_grid import AxesGrid

		dataCO,headCO=myFITS.readFITS(fitsFile)

		wcsCO=WCS(headCO)


		fig = plt.figure(1 , figsize=(16, 8) )
		rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
		#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

		rc('text', usetex=True)

		# grid helper
		grid_helper = pywcsgrid2.GridHelper(wcs=wcsCO)

		# AxesGrid to display tow images side-by-side
		fig = plt.figure(1, (6, 3.5))


		grid = ImageGrid(fig, (1, 1, 1), nrows_ncols=(1, 1),
						 cbar_mode="single", cbar_pad="0.5%",
						 cbar_location="top", cbar_size="2%",
						 axes_class=(pywcsgrid2.Axes, dict(header= wcsCO)))

		main_axes = grid[0]
		main_axes.locator_params(nbins=10)

		cb_axes = grid.cbar_axes[0]  # colorbar axes

		im = main_axes.imshow(np.sqrt(dataCO),origin='lower',cmap="jet",  vmin=0, vmax= 2.4 , interpolation='none')
		main_axes.axis[:].major_ticks.set_color("w")
		cb=cb_axes.colorbar(im)
		#cb_axes.axis["right"].toggle(ticklabels=True)
		cb_axes.set_xlabel("K")

 		#print dir(cb_axes.axes)

		tickesArray=np.asarray( [0,0.1,0.5,1,2,3,4,5] )

		cb.ax.set_xticks( np.sqrt( tickesArray)   )
		cb.ax.set_xticklabels( map(str,tickesArray) )

		#cbar.ax.set_xticksset_xticklabels(['Low', 'Medium', 'High'])


		fig.tight_layout()
		plt.savefig("localM0.pdf", bbox_inches='tight')
		plt.savefig("localM0.png", bbox_inches='tight', dpi=300)



	def drawCheckClouds(self):
		"""
		compare the result of molecular clouds
		:return:
		"""
		#drawLrange,drawBrange=gaiaDis.box(38.9496172,0.1091115,3263.632 ,2991.628 ,4.9024796e-06)
		#region 2
		#drawLrange,drawBrange=gaiaDis.box(44.8346022,0.8519293,2361.857 ,2141.563 ,4.9024796e-06)
		drawLrange,drawBrange=gaiaDis.box(42.8611667,0.1834138,3122.226 ,2901.286 ,4.9024796e-06)


		#vRange=[1.8,9.8] #km/s
		vRange = [ -6,  6	 ]  # km/s

		#first crop fits

		rawCO="G2650Local30.fits"
		
		rawCO="G2650CO12MaskedCO.fits"

		
		labelDendroFITS = "G2650minV3minP16_TrunkAsignMask0.fits"
		labelDBSCANFITS = "DBCLEAN3.0_16Label.fits"
		labelSCIMESFITS = "./scimesG2650/ClusterAsgn_3_16Ve20.fits"

		cropRawcoFITS =  self.tmpPath + "cropRawco.fits"
		cropDendroFITS = self.tmpPath + "cropDendro.fits"
		cropSCIMES = self.tmpPath + "cropScimes.fits"
		cropDBSCAN=  self.tmpPath + "cropDbscan.fits"


		doFITS.cropFITS(rawCO,outFITS=cropRawcoFITS,Vrange=vRange,Brange=drawBrange,Lrange=drawLrange , overWrite=True )

		#cropFITS 3D
		doFITS.cropFITS(labelDendroFITS,outFITS=cropDendroFITS,Vrange=vRange,Brange=drawBrange,Lrange=drawLrange , overWrite=True )
		doFITS.cropFITS(labelDBSCANFITS,outFITS=cropDBSCAN,Vrange=vRange,Brange=drawBrange,Lrange=drawLrange , overWrite=True )
		

		doFITS.cropFITS(labelSCIMESFITS,outFITS=cropSCIMES,Vrange=vRange,Brange=drawBrange,Lrange=drawLrange , overWrite=True )


		#check uniqueness along each points
		if 1:
			self.checkUniqueness(cropDendroFITS )
			self.checkUniqueness(cropDBSCAN )
			self.checkUniqueness(cropSCIMES )

		labelDendro,labelHead=self.getIntLabel(cropDendroFITS )

		WCSCrop=WCS(labelHead)

		fig = plt.figure(1, figsize=(10, 9) )
		rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica'], "size":13 })
		#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

		rc('text', usetex=True)

		axCO= pywcsgrid2.subplot(221, header=   WCSCrop  )


		dataCO,headCO=myFITS.readFITS(cropRawcoFITS)

		intData=np.sum(dataCO,axis=0)*0.2

		axCO.imshow(np.sqrt( intData),origin='lower',cmap="bone",vmin=0 ,vmax=4, interpolation='none')

		at = AnchoredText(r"$^{12}\mathrm{CO}~(J=1\rightarrow0)$", loc=1, frameon=False,  prop={"color": "w","size":13 })
		axCO.add_artist(at)

		axCO.set_ticklabel_type("absdeg", "absdeg")
		axCO.axis[:].major_ticks.set_color("w")


		###dendrogram

		axDendrogram= pywcsgrid2.subplot(222, header=   WCSCrop ,sharex=axCO,sharey=axCO )
		#self.showLabels(axDendrogram,labelDendro  )

		self.showLabels(axDendrogram,labelDendro  )
		#axDendrogram.imshow(labelDendro,origin='lower',cmap="jet", interpolation='none')


		at = AnchoredText("Dendrogram", loc=1, frameon=False,  prop={"color": "w","size":13 })
		axDendrogram.add_artist(at)
		axDendrogram.set_ticklabel_type("absdeg", "absdeg")
		axDendrogram.axis[:].major_ticks.set_color("w")

		###DBSCAN
		labelDBSCAN,labelHead=self.getIntLabel(cropDBSCAN )

		axDBSCAN = pywcsgrid2.subplot(223, header=WCSCrop, sharex=axCO, sharey=axCO)
		self.showLabels(axDBSCAN, labelDBSCAN)
		#axDBSCAN.imshow(labelDBSCAN,origin='lower',cmap="jet", interpolation='none')

		
		at = AnchoredText("DBSCAN", loc=1, frameon=False,  prop={"color": "w","size":13 })
		axDBSCAN.add_artist(at)
		axDBSCAN.set_ticklabel_type("absdeg", "absdeg")
		axDBSCAN.axis[:].major_ticks.set_color("w")


		##SCIMES
		labelSCIMES,labelHead=self.getIntLabel(cropSCIMES )

		axSCIMES = pywcsgrid2.subplot(224, header=WCSCrop, sharex=axCO, sharey=axCO)
		self.showLabels(axSCIMES, labelSCIMES)
		
		#axSCIMES.imshow(labelSCIMES,origin='lower',cmap="jet", interpolation='none')

		
		at = AnchoredText("SCIMES", loc=1, frameon=False, prop={"color": "w","size":13 })
		axSCIMES.add_artist(at)
		axSCIMES.set_ticklabel_type("absdeg", "absdeg")
		axSCIMES.axis[:].major_ticks.set_color("w")



		fig.tight_layout(pad=0.2)
		plt.savefig("checkCloud.pdf", bbox_inches="tight")

		plt.savefig("checkCloud.png", bbox_inches="tight",dpi=600)




	def getIntLabel(self,cropLabelFITS):
		labelDendroData, labelHead = doFITS.readFITS(cropLabelFITS)

		minVDendro = np.nanmin(labelDendroData)

		uniqueValues = np.unique(labelDendroData)
		# print 0 in uniqueValues

		labelDendroData[labelDendroData == minVDendro] = np.NaN

		meanLabel = np.nanmean(labelDendroData, axis=0)

		#meanLabel=np.nan_to_num(meanLabel)

		return meanLabel,labelHead



	def checkUniqueness(self,cropDendroFITS):

		labelDendroData,_ = doFITS.readFITS( cropDendroFITS )

		minVDendro = np.nanmin( labelDendroData )

		uniqueValues=np.unique(labelDendroData)
		#print 0 in uniqueValues
		 

		labelDendroData[ labelDendroData == minVDendro ] = np.NaN


		meanLabel=np.nanmean( labelDendroData,axis=0 )

		#meanLabel=np.nan_to_num(meanLabel)

		uniqueValuesV2=np.unique( meanLabel )
		uniqueValuesV2=uniqueValuesV2[uniqueValuesV2>minVDendro-1]


		print len( uniqueValues ), "==",  len( uniqueValuesV2 )+1



	def ZZZZZZ(self):
		pass



doDBSCAN=myDBSCAN()

G2650CO12FITS="/home/qzyan/WORK/myDownloads/testFellwalker/WMSIPDBSCAN/G2650Local30.fits"
DBMaskFITS= "/home/qzyan/WORK/myDownloads/testFellwalker/G2650DB_1_25.fits"
TaurusCO12FITS="/home/qzyan/WORK/dataDisk/Taurus/t12_new.fits"
PerCO12="/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650Per3060.fits"

localCO13="/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650Local30CO13.fits"

G210CO12="/home/qzyan/WORK/myDownloads/newMadda/data/G210CO12sm.fits"
G210CO13="/home/qzyan/WORK/myDownloads/newMadda/data/G210CO13sm.fits"

ursaMajor=""
G2650MaskCO = "G2650CO12MaskedCO.fits"
#veloicty distance, relation
# 13.46359868  4.24787753

if 0:

	doDBSCAN.drawCheckClouds()
	sys.exit()

if 0:
	doDBSCAN.drawOverallMoment()

	sys.exit()


if 0:
	#doDBSCAN.drawVeDistribution()
	doDBSCAN.drawPeakDistribution()

	#doDBSCAN.physicalAreaDistribution()
	sys.exit()



if 0:
	doDBSCAN.produceDENDROCat()
	sys.exit()
if 1:  # compare distributions
	#doDBSCAN.numberDistribution()
	#doDBSCAN.totaFluxDistribution() # add

	#doDBSCAN.fluxDistribution()

	#doDBSCAN.areaDistribution()
	#doDBSCAN.alphaDistribution()
	doDBSCAN.fluxAlphaDistribution()

	sys.exit()

##
if 0:#Scimes pipe line

	doDBSCAN.produceSCIMECat()
	#get catalog from scimes fits

	sys.exit()



if 0:
	#drawLrange,drawBrange=gaiaDis.box(38.4031840,0.3732480,22768.128 ,18363.802 ,0)
	drawLrange,drawBrange=gaiaDis.box(28.9977093,-0.0541066,22680.000 ,18662.400 ,0)

	doDBSCAN.drawCloudMap( lRange=drawLrange,bRange=drawBrange,  drawChannel= 91 )

	#doDBSCAN
	#doDBSCAN.drawAreaDistribute("ClusterCat_3_16Ve20.fit", region="scimes")
	#doDBSCAN.drawAreaDistribute("taurusDB3_8.fit" , region="scimes" )

	sys.exit()




if 0:#use DBSCAN to produe dendrotrunks
	COData, COHead = myFITS.readFITS(G2650CO12FITS)
	#DBLabelFITS="G2650CO12DendroByDBSCANdbscanS7P3Con1.fits"
	#DBTableFile="G2650CO12DendroByDBSCANCatS7P3Con1.fit"

	#doDBSCAN.clearnDBAssign(  DBLabelFITS, DBTableFile, pixN=16, minDelta=3, minV=7,   prefix="mimicDendro" )

	#DbscanSigmaList = np.arange(6)
	#for sigmas in [7]:
		#print "Calculating dengram with dBSCAN sigma:", sigmas
		#doDBSCAN.computeDBSCAN(COData, COHead, min_sigma=sigmas, min_pix= 3 , connectivity=1, region="G2650CO12DendroByDBSCAN" , mimicDendro= False )
	#sys.exit()
	if 1: #step2, calculate all catalog
		DbscanSigmaList = np.arange(2, 7.5, 0.5)
		for sigmas in [7]:
			labelFITS="G2650CO12DendroByDBSCANdbscanS7P3Con1.fits"
			savename="G2650CO12DendroByDBSCANCatS{}P{}Con1".format(sigmas,3)

			doDBSCAN.getCatFromLabelArray( G2650CO12FITS,  labelFITS  ,  doDBSCAN.TBModel, minPix=3, rms=sigmas, saveMarker= savename )

	sys.exit()

if 0: #small test
	pass

	TBListP8,TBListP16,TBLabelsP8,TBLabelsP16,dendroSigmaList=doDBSCAN.getTBList("SCIMES")

	print TBListP8
	sys.exit()







if 0: #high Galacticlatitudes, ursa major
	doDBSCAN.rms=0.16
	coFITS12UM="/home/qzyan/WORK/projects/NewUrsaMajorPaper/OriginalFITS/myCut12CO.fits"
	COData,COHead=myFITS.readFITS( coFITS12UM)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=2, min_pix=16, connectivity=2, region="UMCO12",savePath="./ursaMajor/")

	doDBSCAN.testFluxOfUM(  )

	#doDBSCAN.getCatFromLabelArray(coFITS12UM,"UMCO12dbscanS2P8Con2.fits",doDBSCAN.TBModel,saveMarker="UMCO12_2_8")
	#doDBSCAN.drawAreaDistribute("UMCO12_2_8.fit" , region="Taurus" )

	sys.exit()






if 0: #produce new mask
		maskCO="G2650CO12MaskedCO.fits"
		#doDBSCAN.computeDBSCAN(COData, COHead, min_sigma=2, min_pix=8, connectivity=2, region="G2650CO12Mask")

		doDBSCAN.produceMask( G2650CO12FITS, "G2650CO12dbscanS2.0P8Con2.fits", "G2650CO12DBCatS2.0P8Con2.fit",  region="G2650CO12")
		#doDBSCAN.getDiffCO("G2650CO12MaskedCO.fits","minV4minP16_TrunkAsign.fits",cutoff=4)


		sys.exit()







if 0: #DBSCAN PipeLine

	if 0: #produce all DBSCAN cases step1
		COData, COHead = myFITS.readFITS(G2650CO12FITS)

		DbscanSigmaList = np.arange(2, 7.5, 0.5)
		for sigmas in DbscanSigmaList:
			print "Calculating ",sigmas
			doDBSCAN.computeDBSCAN(  COData,COHead, min_sigma=sigmas, min_pix=8, connectivity=2, region="G2650CO12")

	if 0: #step2, calculate all catalog
		DbscanSigmaList = np.arange(2, 7.5, 0.5)
		for sigmas in DbscanSigmaList:
			labelFITS="G2650CO12dbscanS{}P8Con2.fits".format( sigmas )
			savename="G2650CO12DBCatS{}P{}Con2".format(sigmas,8)

			doDBSCAN.getCatFromLabelArray( G2650CO12FITS,  labelFITS  ,  doDBSCAN.TBModel, minPix=8, rms=sigmas, saveMarker= savename )




	if 0:

		doDBSCAN.cleanAllDBfits()
	sys.exit()


if 0:#distance pipeline

	#step1, extend 3,sigma,1000 pixels, to 2 sigma
	if 0: # use masked fits, a cutoff to
		dendroLabel= "ClusterAsgn_3_1000Ve20.fits"
		extendedLabel="G2650DisCloudVe20_extend.fits"

		doDBSCAN.myDilation( dendroLabel, G2650CO12FITS, startSigma=10,endSigma=2, saveName="G2650DisCloudVe20",maskCOFITS= G2650MaskCO )



	if 1: # get catalog
		dendro1000ExtendFITS = "G2650DisCloudVe20_extend.fits"
		savename="G2650CloudForDisCatDendro1000"
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS, dendro1000ExtendFITS, doDBSCAN.TBModel, minPix=1000, rms=3, saveMarker=savename)


	if 0:
		pass
		#then produce int figures, usemyScimes.py








if 0: #Taurus
	COData,COHead=myFITS.readFITS( TaurusCO12FITS)
	doDBSCAN.rms=0.3
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=3,min_pix=8,connectivity=2,region="Taurus")

	#doDBSCAN.getCatFromLabelArray(TaurusCO12FITS,"TaurusdbscanS3P8Con2.fits",doDBSCAN.TBModel,saveMarker="taurusDB3_8")
	#doDBSCAN.drawAreaDistribute("taurusDB3_8.fit" , region="Taurus" )

	


	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=1,min_pix=25,connectivity=3)

	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2)
	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=6,min_pix=9,connectivity=2)

	sys.exit()


if 0:


	doDBSCAN.drawDBSCANNumber()
	doDBSCAN.drawDBSCANArea()



if 0: #dilation SCIMES

	#scimesFITS= "/home/qzyan/WORK/myDownloads/MWISPcloud/ClusterAsgn_ComplicateVe.fits"
	#rawFITS="/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"

	scimesFITS= "ClusterAsgn_3_16Ve20.fits"
	rawFITS= G2650CO12FITS  #"/home/qzyan/WORK/myDownloads/testScimes/complicatedTest.fits"

	doDBSCAN.myDilation( scimesFITS , rawFITS, saveName="G2650SCIMES_3_16Ve20", startSigma=15 )

	sys.exit()

if 0:
	doDBSCAN.getLVFITSByDBMASK( "G2650CO12dbscanS2.0P8Con2.fits", G2650CO12FITS, "/home/qzyan/WORK/myDownloads/testScimes/G2650PV.fits"  )




if 0: #test Fast Dendrogram


	doDBSCAN.fastDendro("testDendro.fits",minV=5,minP=8)


	sys.exit()

if 0: #test Fast Dendrogram
	COData,COHead=myFITS.readFITS( G2650CO12FITS)


	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=3,connectivity=1,region="G2650CO12DBDendro")

	#should peak sigma be larger?
	doDBSCAN.setMinVandPeak( "G2650CO12DBDendrodbscanS5P3Con1.fits" ,G2650CO12FITS,minP=8,peakSigma=8 )

	sys.exit()






if 0: # get catalog from extended fits


	doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650DisCloudVe20_extend.fits",doDBSCAN.TBModel,saveMarker="G2650CloudForDisCat")
	sys.exit()







if 0:
	#doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650CO12dbscanS2P16Con2.fits",doDBSCAN.TBModel, saveMarker="G2650CO12DBCatS2P16Con2" )
	for i in np.arange(2 ,8,0.5):
		savename="G2650CO12DBCatS{}P{}Con2".format(i,8)
		doDBSCAN.getCatFromLabelArray(G2650CO12FITS,"G2650CO12dbscanS{}P8Con2.fits".format(i),doDBSCAN.TBModel,saveMarker=savename)










if 0: #
	G214COFITS="G214CO12.fits"
	COData,COHead=myFITS.readFITS( G214COFITS)

	doDBSCAN.computeDBSCAN(COData, COHead,region="G214")
	doDBSCAN.slowDBSCAN(COData, COHead,region="G214")




	sys.exit()






if 0:# DBSCAN for G210
	region="G210CO13"

	processFITS=G210CO13

	doDBSCAN.rms=0.25


	if 1:#find clouds
		COData,COHead=myFITS.readFITS( processFITS)

		for sigmas in [3,4,5]:
			saveName=doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=sigmas, min_pix=8,connectivity=2, region=region)

			doDBSCAN.getCatFromLabelArray(processFITS, saveName , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN{}_8".format(sigmas)   )

	sys.exit()




if 0:# DBSCAN for CO13
	region="Local13"
	doDBSCAN.rms=0.25


	if 0:#find clouds
		COData,COHead=myFITS.readFITS( localCO13)

		for sigmas in [3,4,5]:
			doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=sigmas, min_pix=9,connectivity=2, region=region)

		sys.exit()

	else:#calcatelog

		doDBSCAN.getCatFromLabelArray(localCO13,  "Local13dbscanS3P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN3_9"  )
		doDBSCAN.getCatFromLabelArray(localCO13,  "Local13dbscanS4P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN4_9"  )
		doDBSCAN.getCatFromLabelArray(localCO13,  "Local13dbscanS5P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker=region+"DBSCAN5_9"  )



if 0:
	#draw perseus
	#doDBSCAN.drawAreaDistribute("DBSCAN3_9.fit"  )

	doDBSCAN.drawAreaDistribute("ClusterCat_3_16Ve20.fit" , region="scimes" )


	#doDBSCAN.drawAreaDistribute("minV3minP16_dendroCatTrunk.fit" , region="Perseus" )

	#doDBSCAN.drawSumDistribute("DBSCAN3_9Sigma1_P1FastDBSCAN.fit"  )


	#doDBSCAN.drawSumDistribute("DBSCAN3_9.fit"  )



	#doDBSCAN.drawSumDistribute("minV3minP16_dendroCatTrunk.fit"  )
	#doDBSCAN.drawDBSCANArea()

	sys.exit()



if 0:# DBSCAN for perseus
	region="PerG2650"
	PerCO12="/home/qzyan/WORK/dataDisk/MWISP/G2650/merge/G2650Per3060.fits"

	if 0:#find clouds


		COData,COHead=myFITS.readFITS( PerCO12)
		doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2, region=region)
		doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2, region=region)

		sys.exit()
	else:#calcatelog

		doDBSCAN.getCatFromLabelArray(PerCO12,  "PerG2650dbscanS3P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker="DBSCAN3_9"  )
		doDBSCAN.getCatFromLabelArray(PerCO12,  "PerG2650dbscanS4P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker="DBSCAN4_9"  )
		doDBSCAN.getCatFromLabelArray(PerCO12,  "PerG2650dbscanS5P9Con2.fits" , doDBSCAN.TBModel  ,  rms=1,minPix=1 , saveMarker="DBSCAN5_9"  )

if 0:#Example
	doDBSCAN.rms=0.5
	COData,COHead=myFITS.readFITS( CO12FITS)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=i,min_pix=9,connectivity=2)




if 0:
	ModelTB="minV3minP16_dendroCatTrunk.fit"

	#doDBSCAN.getCatFromLabelArray(CO12FITS, "dbscanS1P25Con3.fits",  ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN1_25"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS, "dbscanS2P16Con2.fits",  ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN2_16"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS3P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN3_9"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS4P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN4_9"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN5_9"  )
	#doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS6P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN6_9"  )

	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS2.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN25_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS3.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN35_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS4.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN45_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS5.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN55_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS6.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN65_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS7.5P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN75_9"  )
	doDBSCAN.getCatFromLabelArray(CO12FITS,  "dbscanS7P9Con2.fits" ,   ModelTB,  rms=1,minPix=1 , saveMarker="DBSCAN7_9"  )

	import sys
	sys.exit()

if 0:
	COData,COHead=myFITS.readFITS( CO12FITS)

	#doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=1,min_pix=25,connectivity=3)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=2,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=3,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=4,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=5,min_pix=9,connectivity=2)
	doDBSCAN.computeDBSCAN(COData,COHead, min_sigma=6,min_pix=9,connectivity=2)







if 0:

	ModelTB="minV3minP16_dendroCatTrunk.fit"

	doDBSCAN.getCatFromLabelArray(CO12FITS,DBMaskFITS,  ModelTB,  rms=1,minPix=25 )